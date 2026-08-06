#include "tiny_cl.h"
#include "weights.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

static TinyCLModel s_model;
static StaticReplayBuffer s_replay;
static TinyCLUpdateRule s_rule = TINY_CL_UPDATE_SORT;
static bool s_replay_enabled = true;
/* Running estimate of the top-k magnitude cutoff, used by the
   THRESHOLD rule to avoid any selection pass at all. */
static float s_grad_threshold = 0.0f;

// Helper: ReLU activation
static inline float relu(float x) {
    return x > 0.0f ? x : 0.0f; }

/*
 * Per-window standardisation. Must match standardise() in
 * tools/train_vibration_model.py exactly (population std, eps 1e-8),
 * otherwise host-trained weights will not transfer.
 */
void tiny_cl_standardise_window(const float in[SEQ_LEN], float out[SEQ_LEN]) {
    float sum = 0.0f;
    for (int i = 0; i < SEQ_LEN; ++i) sum += in[i];
    float mean = sum / (float)SEQ_LEN;
    float acc = 0.0f;
    for (int i = 0; i < SEQ_LEN; ++i) {
        float d = in[i] - mean;
        acc += d * d;
    }
    float sd = sqrtf(acc / (float)SEQ_LEN);
    float inv = 1.0f / (sd + 1e-8f);
    for (int i = 0; i < SEQ_LEN; ++i) out[i] = (in[i] - mean) * inv;
}

/*
 * Quickselect (Hoare partition) placing the k-th largest magnitude at
 * index k-1, so that entries[0..k-1] are the top-k in arbitrary order.
 * Expected O(n) versus O(n log n) for the comparison sort it replaces.
 * Median-of-three pivoting bounds the adversarial case in practice.
 */
static void introselect_topk(SparseGradEntry *a, int n, int k) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        int mid = lo + ((hi - lo) >> 1);
        /* median-of-three -> a[lo] holds the pivot */
        if (a[mid].mag > a[lo].mag)  { SparseGradEntry t=a[mid]; a[mid]=a[lo]; a[lo]=t; }
        if (a[hi].mag  > a[lo].mag)  { SparseGradEntry t=a[hi];  a[hi]=a[lo];  a[lo]=t; }
        if (a[mid].mag > a[hi].mag)  { SparseGradEntry t=a[mid]; a[mid]=a[hi]; a[hi]=t; }
        float pivot = a[lo].mag;
        int i = lo, j = hi + 1;
        for (;;) {
            do { ++i; } while (i <= hi && a[i].mag > pivot);
            do { --j; } while (a[j].mag < pivot);
            if (i >= j) break;
            SparseGradEntry t = a[i]; a[i] = a[j]; a[j] = t;
        }
        SparseGradEntry t = a[lo]; a[lo] = a[j]; a[j] = t;
        if (j >= k) hi = j - 1;
        else        lo = j + 1;
    }
}

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

void tiny_cl_init(TinyCLUpdateRule rule) {
    memset(&s_model, 0, sizeof(s_model));
    memset(&s_replay, 0, sizeof(s_replay));

    s_rule = rule;
    s_replay_enabled = true;
    s_grad_threshold = 0.0f;
    s_model.use_sparsity = (rule != TINY_CL_UPDATE_DENSE);
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

void tiny_cl_set_replay_enabled(bool enabled) {
    s_replay_enabled = enabled;
}

void tiny_cl_extract_features(const float input_window[SEQ_LEN], float latent_out[LATENT_DIM]) {
    static float norm_window[SEQ_LEN];
    static float conv1[CONV1_OUT][SEQ_LEN];
    static float pool1[CONV1_OUT][POOL1_LEN];
    static float conv2[CONV2_OUT][POOL1_LEN];
    static float pool2[CONV2_OUT][POOL2_LEN];

    tiny_cl_standardise_window(input_window, norm_window);

    // Conv1 Layer (1 channel in -> CONV1_OUT channels out, kernel CONV1_KERNEL)
    int pad1 = CONV1_KERNEL / 2;
    for (int c = 0; c < CONV1_OUT; ++c) {
        for (int i = 0; i < SEQ_LEN; ++i) {
            float sum = CONV1_BIAS[c];
            for (int k = 0; k < CONV1_KERNEL; ++k) {
                int in_idx = i + k - pad1;
                float val = (in_idx >= 0 && in_idx < SEQ_LEN) ? norm_window[in_idx] : 0.0f;
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

    const int total_params = NUM_CLASSES * LATENT_DIM;
    int top_k = (int)(s_model.base_sparsity * (float)total_params);
    if (top_k < 1) top_k = 1;

    switch (s_rule) {

    case TINY_CL_UPDATE_DENSE:
        for (int c = 0; c < NUM_CLASSES; ++c) {
            for (int d = 0; d < LATENT_DIM; ++d) {
                s_model.clf_w[c][d] -= LEARNING_RATE * grad_w[c][d];
            }
        }
        break;

    case TINY_CL_UPDATE_SORT: {
        /* Baseline: full comparison sort to find the top-k. O(n log n)
           plus an indirect-call comparator on every comparison. */
        static SparseGradEntry entries[NUM_CLASSES * LATENT_DIM];
        int count = 0;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            for (int d = 0; d < LATENT_DIM; ++d) {
                entries[count].mag = fabsf(grad_w[c][d]);
                entries[count].idx = c * LATENT_DIM + d;
                count++;
            }
        }
        qsort(entries, total_params, sizeof(SparseGradEntry), cmp_sparse_desc);
        for (int i = 0; i < top_k; ++i) {
            int idx = entries[i].idx;
            int c = idx / LATENT_DIM, d = idx % LATENT_DIM;
            s_model.clf_w[c][d] -= LEARNING_RATE * grad_w[c][d];
        }
        break;
    }

    case TINY_CL_UPDATE_INTROSELECT: {
        /* Same exact top-k set as SORT, but selected by partitioning
           rather than ordering: expected O(n), no comparator indirection. */
        static SparseGradEntry entries[NUM_CLASSES * LATENT_DIM];
        int count = 0;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            for (int d = 0; d < LATENT_DIM; ++d) {
                entries[count].mag = fabsf(grad_w[c][d]);
                entries[count].idx = c * LATENT_DIM + d;
                count++;
            }
        }
        introselect_topk(entries, total_params, top_k);
        for (int i = 0; i < top_k; ++i) {
            int idx = entries[i].idx;
            int c = idx / LATENT_DIM, d = idx % LATENT_DIM;
            s_model.clf_w[c][d] -= LEARNING_RATE * grad_w[c][d];
        }
        break;
    }

    case TINY_CL_UPDATE_THRESHOLD: {
        /* Approximate: keep a running magnitude cutoff and update only
           coordinates exceeding it, in a single pass with no selection
           structure at all. The cutoff adapts multiplicatively toward the
           target sparsity, so no sort or partition is ever performed and
           no O(n) scratch is required. */
        int written = 0;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            for (int d = 0; d < LATENT_DIM; ++d) {
                if (fabsf(grad_w[c][d]) >= s_grad_threshold) {
                    s_model.clf_w[c][d] -= LEARNING_RATE * grad_w[c][d];
                    written++;
                }
            }
        }
        /* Multiplicative control loop toward top_k writes per step. */
        if (written > top_k)      s_grad_threshold *= 1.05f;
        else if (written < top_k) s_grad_threshold *= 0.95f;
        if (s_grad_threshold < 1e-12f) s_grad_threshold = 1e-12f;
        s_model.last_writes = written;
        break;
    }
    }

    if (s_rule != TINY_CL_UPDATE_DENSE) {
        s_model.current_sparsity = s_model.base_sparsity;
    }
    if (s_rule == TINY_CL_UPDATE_DENSE)  s_model.last_writes = total_params;
    else if (s_rule != TINY_CL_UPDATE_THRESHOLD) s_model.last_writes = top_k;

    for (int c = 0; c < NUM_CLASSES; ++c) {
        s_model.clf_b[c] -= LEARNING_RATE * grad_b[c];
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
    if (!s_replay_enabled) return;
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

    stats->head_bytes       = (uint32_t)sizeof(s_model);
    stats->reservoir_bytes  = (uint32_t)sizeof(s_replay);

    /* Function-local static scratch in this translation unit. These were
       omitted by the previous revision of this function, which reported
       only head+reservoir and thereby understated the true SRAM cost. */
    stats->activation_bytes = (uint32_t)(
          sizeof(float) * CONV1_OUT * SEQ_LEN      /* conv1        */
        + sizeof(float) * CONV1_OUT * POOL1_LEN    /* pool1        */
        + sizeof(float) * CONV2_OUT * POOL1_LEN    /* conv2        */
        + sizeof(float) * CONV2_OUT * POOL2_LEN    /* pool2        */
        + sizeof(float) * SEQ_LEN);                /* norm_window  */

    /* The gradient accumulator is always live. The selection scratch is
       only instantiated by the SORT and INTROSELECT rules; THRESHOLD and
       DENSE need none, which is a real memory saving and not just a
       speed one. */
    uint32_t grad = (uint32_t)(sizeof(float) * NUM_CLASSES * LATENT_DIM
                             + sizeof(float) * NUM_CLASSES);
    if (s_rule == TINY_CL_UPDATE_SORT || s_rule == TINY_CL_UPDATE_INTROSELECT) {
        grad += (uint32_t)(sizeof(SparseGradEntry) * NUM_CLASSES * LATENT_DIM);
    }
    stats->gradient_bytes = grad;

    stats->dequant_bytes  = (uint32_t)(sizeof(float) * LATENT_DIM);

    stats->total_sram_bytes = stats->head_bytes
                            + stats->reservoir_bytes
                            + stats->activation_bytes
                            + stats->gradient_bytes
                            + stats->dequant_bytes;

    /* static const, placed in memory-mapped flash on ESP32, not SRAM. */
    stats->flash_const_bytes = (uint32_t)(
          sizeof(CONV1_WEIGHTS) + sizeof(CONV1_BIAS)
        + sizeof(CONV2_WEIGHTS) + sizeof(CONV2_BIAS)
        + sizeof(INITIAL_FC_WEIGHTS) + sizeof(INITIAL_FC_BIAS));

    stats->buffer_occupancy = s_replay.size;
    stats->current_sparsity = s_model.current_sparsity;
    stats->update_rule      = s_rule;
}
