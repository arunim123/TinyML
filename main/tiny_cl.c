#include "tiny_cl.h"
#include "weights.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static TinyCLModel s_model;
static StaticReplayBuffer s_replay;

// Helper: ReLU activation
static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f; }

// Helper: Comparator for sparse gradient sorting (descending order)
static int cmp_sparse_desc(const void *a, const void *b) {
    const SparseGradEntry *ea = (const SparseGradEntry *)a;
    const SparseGradEntry *eb = (const SparseGradEntry *)b;
    return (ea->mag < eb->mag) - (ea->mag > eb->mag);
}

// 4-bit Quantization Helpers
static void quantize_latent_4bit(const float latent[LATENT_DIM], uint8_t packed_out[LATENT_DIM / 2], float *out_min, float *out_max) {
    float min_v = latent[0];
    float max_v = latent[0];
    for (int i = 1; i < LATENT_DIM; ++i) {
        if (latent[i] < min_v) min_v = latent[i];
        if (latent[i] > max_v) max_v = latent[i];
    }

    float range = max_v - min_v;
    if (range < 1e-6f) range = 1e-6f;
    float scale = 15.0f / range;

    *out_min = min_v;
    *out_max = max_v;

    for (int i = 0; i < LATENT_DIM / 2; ++i) {
        float val0 = latent[2 * i];
        float val1 = latent[2 * i + 1];

        uint8_t q0 = (uint8_t)roundf((val0 - min_v) * scale);
        uint8_t q1 = (uint8_t)roundf((val1 - min_v) * scale);

        if (q0 > 15) q0 = 15;
        if (q1 > 15) q1 = 15;

        packed_out[i] = (q0 & 0x0F) | ((q1 & 0x0F) << 4);
    }
}

static void dequantize_latent_4bit(const uint8_t packed[LATENT_DIM / 2], float min_v, float max_v, float latent_out[LATENT_DIM]) {
    float range = max_v - min_v;
    float step = range / 15.0f;

    for (int i = 0; i < LATENT_DIM / 2; ++i) {
        uint8_t b = packed[i];
        uint8_t q0 = b & 0x0F;
        uint8_t q1 = (b >> 4) & 0x0F;

        latent_out[2 * i] = min_v + (float)q0 * step;
        latent_out[2 * i + 1] = min_v + (float)q1 * step;
    }
}

void tiny_cl_init(bool use_sparsity) {
    memset(&s_model, 0, sizeof(s_model));
    memset(&s_replay, 0, sizeof(s_replay));

    s_model.use_sparsity = use_sparsity;
    s_model.base_sparsity = BASE_SPARSITY;
    s_model.current_sparsity = BASE_SPARSITY;
    s_model.total_adaptation_steps = 0;

    // Load FC layer initial weights from weights.h
    memcpy(s_model.clf_w, INITIAL_FC_WEIGHTS, sizeof(INITIAL_FC_WEIGHTS));
    memcpy(s_model.clf_b, INITIAL_FC_BIAS, sizeof(INITIAL_FC_BIAS));

    s_replay.capacity = BUFFER_CAPACITY;
    s_replay.size = 0;
    s_replay.samples_seen = 0;
}

void tiny_cl_extract_features(const float input_window[SEQ_LEN], float latent_out[LATENT_DIM]) {
    static float conv1[CONV1_OUT][SEQ_LEN];
    static float pool1[CONV1_OUT][POOL1_LEN];
    static float conv2[CONV2_OUT][POOL1_LEN];
    static float pool2[CONV2_OUT][POOL2_LEN];

    // Conv1 Layer (1 channel in -> CONV1_OUT channels out, kernel CONV1_KERNEL)
    int pad1 = CONV1_KERNEL / 2;
    for (int c = 0; c < CONV1_OUT; ++c) {
        for (int i = 0; i < SEQ_LEN; ++i) {
            float sum = CONV1_BIAS[c];
            for (int k = 0; k < CONV1_KERNEL; ++k) {
                int in_idx = i + k - pad1;
                float val = (in_idx >= 0 && in_idx < SEQ_LEN) ? input_window[in_idx] : 0.0f;
                sum += val * CONV1_WEIGHTS[c * CONV1_KERNEL + k];
            }
            conv1[c][i] = relu(sum);
        }
    }

    // MaxPool1 (stride 2)
    for (int c = 0; c < CONV1_OUT; ++c) {
        for (int i = 0; i < POOL1_LEN; ++i) {
            float v1 = conv1[c][2 * i];
            float v2 = (2 * i + 1 < SEQ_LEN) ? conv1[c][2 * i + 1] : v1;
            pool1[c][i] = v1 > v2 ? v1 : v2;
        }
    }

    // Conv2 Layer (CONV1_OUT in -> CONV2_OUT out, kernel CONV2_KERNEL)
    int pad2 = CONV2_KERNEL / 2;
    for (int c_out = 0; c_out < CONV2_OUT; ++c_out) {
        for (int i = 0; i < POOL1_LEN; ++i) {
            float sum = CONV2_BIAS[c_out];
            for (int c_in = 0; c_in < CONV1_OUT; ++c_in) {
                for (int k = 0; k < CONV2_KERNEL; ++k) {
                    int in_idx = i + k - pad2;
                    float val = (in_idx >= 0 && in_idx < POOL1_LEN) ? pool1[c_in][in_idx] : 0.0f;
                    int weight_idx = c_out * (CONV1_OUT * CONV2_KERNEL) + c_in * CONV2_KERNEL + k;
                    sum += val * CONV2_WEIGHTS[weight_idx];
                }
            }
            conv2[c_out][i] = relu(sum);
        }
    }

    // MaxPool2 (stride 2) & Flatten into latent_out
    int idx = 0;
    for (int c = 0; c < CONV2_OUT; ++c) {
        for (int i = 0; i < POOL2_LEN; ++i) {
            float v1 = conv2[c][2 * i];
            float v2 = (2 * i + 1 < POOL1_LEN) ? conv2[c][2 * i + 1] : v1;
            latent_out[idx++] = v1 > v2 ? v1 : v2;
        }
    }
}

int tiny_cl_predict(const float latent[LATENT_DIM], float logits_out[NUM_CLASSES], float probs_out[NUM_CLASSES]) {
    float max_logit = -1e9f;
    for (int c = 0; c < NUM_CLASSES; ++c) {
        float sum = s_model.clf_b[c];
        for (int d = 0; d < LATENT_DIM; ++d) {
            sum += latent[d] * s_model.clf_w[c][d];
        }
        logits_out[c] = sum;
        if (sum > max_logit) max_logit = sum;
    }

    // Softmax calculation
    float exp_sum = 0.0f;
    for (int c = 0; c < NUM_CLASSES; ++c) {
        probs_out[c] = expf(logits_out[c] - max_logit);
        exp_sum += probs_out[c];
    }

    int best_class = 0;
    float max_prob = 0.0f;
    for (int c = 0; c < NUM_CLASSES; ++c) {
        probs_out[c] /= exp_sum;
        if (probs_out[c] > max_prob) {
            max_prob = probs_out[c];
            best_class = c;
        }
    }

    return best_class;
}

void tiny_cl_adapt_sparse(const float latent[LATENT_DIM], int target_label) {
    if (target_label < 0 || target_label >= NUM_CLASSES) return;

    float logits[NUM_CLASSES];
    float probs[NUM_CLASSES];
    tiny_cl_predict(latent, logits, probs);

    // Compute cross-entropy gradients for output linear layer
    static float grad_w[NUM_CLASSES][LATENT_DIM];
    static float grad_b[NUM_CLASSES];

    for (int c = 0; c < NUM_CLASSES; ++c) {
        float target = (c == target_label) ? 1.0f : 0.0f;
        float err = probs[c] - target;
        grad_b[c] = err;
        for (int d = 0; d < LATENT_DIM; ++d) {
            grad_w[c][d] = err * latent[d];
        }
    }

    if (!s_model.use_sparsity) {
        // Full dense SGD update
        for (int c = 0; c < NUM_CLASSES; ++c) {
            s_model.clf_b[c] -= LEARNING_RATE * grad_b[c];
            for (int d = 0; d < LATENT_DIM; ++d) {
                s_model.clf_w[c][d] -= LEARNING_RATE * grad_w[c][d];
            }
        }
    } else {
        // Sparse magnitude-based gradient update (Touch only top % of gradients)
        static SparseGradEntry entries[NUM_CLASSES * LATENT_DIM];
        int total_params = NUM_CLASSES * LATENT_DIM;
        int count = 0;

        for (int c = 0; c < NUM_CLASSES; ++c) {
            for (int d = 0; d < LATENT_DIM; ++d) {
                entries[count].mag = fabsf(grad_w[c][d]);
                entries[count].idx = c * LATENT_DIM + d;
                count++;
            }
        }

        qsort(entries, total_params, sizeof(SparseGradEntry), cmp_sparse_desc);

        int top_k = (int)(s_model.base_sparsity * (float)total_params);
        if (top_k < 1) top_k = 1;

        for (int i = 0; i < top_k; ++i) {
            int idx = entries[i].idx;
            int c = idx / LATENT_DIM;
            int d = idx % LATENT_DIM;
            s_model.clf_w[c][d] -= LEARNING_RATE * grad_w[c][d];
        }

        for (int c = 0; c < NUM_CLASSES; ++c) {
            s_model.clf_b[c] -= LEARNING_RATE * grad_b[c];
        }
    }

    s_model.total_adaptation_steps++;
}

void tiny_cl_replay_store(const float latent[LATENT_DIM], int label) {
    s_replay.samples_seen++;
    int slot = -1;

    if (s_replay.size < s_replay.capacity) {
        slot = s_replay.size++;
    } else {
        // Reservoir sampling random replacement
        long long r = rand() % s_replay.samples_seen;
        if (r < s_replay.capacity) {
            slot = (int)r;
        }
    }

    if (slot >= 0) {
        quantize_latent_4bit(latent, s_replay.quantized[slot], &s_replay.min_vals[slot], &s_replay.max_vals[slot]);
        s_replay.labels[slot] = (int8_t)label;
    }
}

void tiny_cl_replay_adapt_step(void) {
    if (s_replay.size == 0) return;

    static float dequant_latent[LATENT_DIM];
    int batch_size = BATCH_SIZE < s_replay.size ? BATCH_SIZE : s_replay.size;

    for (int i = 0; i < batch_size; ++i) {
        int idx = rand() % s_replay.size;
        dequantize_latent_4bit(s_replay.quantized[idx], s_replay.min_vals[idx], s_replay.max_vals[idx], dequant_latent);
        tiny_cl_adapt_sparse(dequant_latent, s_replay.labels[idx]);
    }
}

void tiny_cl_get_stats(TinyCLStats *stats) {
    if (!stats) return;

    stats->model_bytes = sizeof(s_model);
    stats->buffer_bytes = sizeof(s_replay);
    stats->total_static_bytes = stats->model_bytes + stats->buffer_bytes;
    stats->buffer_occupancy = s_replay.size;
    stats->current_sparsity = s_model.current_sparsity;
}
