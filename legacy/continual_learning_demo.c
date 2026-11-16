#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define SEQ_LEN 100
#define CONV1_OUT 8
#define CONV2_OUT 16
#define POOL1_LEN (SEQ_LEN / 2)
#define POOL2_LEN (POOL1_LEN / 2)
#define LATENT_DIM (CONV2_OUT * POOL2_LEN)
#define NUM_CLASSES 2

#define TRAIN_SAMPLES 1000
#define TEST_SAMPLES 400
#define BASE_EPOCHS 10
#define EVOL_EPOCHS 10
#define BATCH_SIZE 32

#define BUFFER_CAPACITY 100

#define LEARNING_RATE 0.01f
#define BASE_SPARSITY 0.05f

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

typedef struct
{
    float x[SEQ_LEN];
    int y;
} Sample;

typedef struct
{
    int capacity;
    int size;
    long long samples_seen;

    int8_t *quantized; // [capacity][LATENT_DIM], values 0..15
    float *min_vals;   // [capacity]
    float *max_vals;   // [capacity]
    int *labels;       // [capacity]
} ReplayBuffer;

typedef struct
{
    // Frozen feature extractor parameters
    float conv1_w[CONV1_OUT][3];
    float conv1_b[CONV1_OUT];
    float conv2_w[CONV2_OUT][CONV1_OUT][3];
    float conv2_b[CONV2_OUT];

    // Trainable classifier
    float clf_w[NUM_CLASSES][LATENT_DIM];
    float clf_b[NUM_CLASSES];

    bool use_sparsity;
    float base_sparsity;
    float update_ratio_s;
} Model;

static float rand_uniform(void)
{
    return (float)rand() / (float)RAND_MAX;
}

static float rand_normal(float mean, float stddev)
{
    // Box-Muller transform
    float u1 = rand_uniform();
    float u2 = rand_uniform();
    if (u1 < 1e-7f)
    {
        u1 = 1e-7f;
    }
    float mag = sqrtf(-2.0f * logf(u1));
    float z0 = mag * cosf(2.0f * (float)M_PI * u2);
    return mean + stddev * z0;
}

static void shuffle_int_array(int *arr, int n)
{
    for (int i = n - 1; i > 0; --i)
    {
        int j = rand() % (i + 1);
        int tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

static void generate_synthetic_data(Sample *samples, int num_samples, bool is_evolved)
{
    float t[SEQ_LEN];
    for (int i = 0; i < SEQ_LEN; ++i)
    {
        t[i] = 10.0f * ((float)i / (float)(SEQ_LEN - 1));
    }

    for (int i = 0; i < num_samples; ++i)
    {
        int label = rand() % 2;
        float freq = (label == 0) ? 1.0f : 1.35f;
        float amp = (label == 0) ? 1.0f : 1.15f;
        float offset = rand_normal(0.0f, 0.08f);
        float phase = rand_uniform() * 2.0f * (float)M_PI;

        freq *= 1.0f + rand_normal(0.0f, 0.12f);
        amp *= 1.0f + rand_normal(0.0f, 0.10f);

        if (is_evolved)
        {
            freq *= 0.9f;
            amp *= 0.85f;
            offset -= 0.25f;
        }

        samples[i].y = label;
        for (int j = 0; j < SEQ_LEN; ++j)
        {
            float noise = rand_normal(0.0f, 0.28f);
            samples[i].x[j] = amp * sinf(2.0f * (float)M_PI * freq * t[j] + phase) + offset + noise;
        }
    }
}

static float init_weight_scale(int fan_in)
{
    return sqrtf(2.0f / (float)fan_in);
}

static void init_model(Model *m, bool use_sparsity)
{
    memset(m, 0, sizeof(*m));

    m->use_sparsity = use_sparsity;
    m->base_sparsity = BASE_SPARSITY;
    m->update_ratio_s = BASE_SPARSITY;

    float s1 = init_weight_scale(3);
    for (int oc = 0; oc < CONV1_OUT; ++oc)
    {
        for (int k = 0; k < 3; ++k)
        {
            m->conv1_w[oc][k] = rand_normal(0.0f, s1 * 0.5f);
        }
        m->conv1_b[oc] = 0.0f;
    }

    float s2 = init_weight_scale(CONV1_OUT * 3);
    for (int oc = 0; oc < CONV2_OUT; ++oc)
    {
        for (int ic = 0; ic < CONV1_OUT; ++ic)
        {
            for (int k = 0; k < 3; ++k)
            {
                m->conv2_w[oc][ic][k] = rand_normal(0.0f, s2 * 0.5f);
            }
        }
        m->conv2_b[oc] = 0.0f;
    }

    float sc = init_weight_scale(LATENT_DIM);
    for (int c = 0; c < NUM_CLASSES; ++c)
    {
        for (int d = 0; d < LATENT_DIM; ++d)
        {
            m->clf_w[c][d] = rand_normal(0.0f, sc);
        }
        m->clf_b[c] = 0.0f;
    }
}

static inline float relu(float x)
{
    return x > 0.0f ? x : 0.0f;
}

static void forward_feature_extractor(const Model *m, const float x[SEQ_LEN], float latent[LATENT_DIM])
{
    float conv1[CONV1_OUT][SEQ_LEN];
    float pool1[CONV1_OUT][POOL1_LEN];
    float conv2[CONV2_OUT][POOL1_LEN];
    float pool2[CONV2_OUT][POOL2_LEN];

    // Conv1 (padding=1, kernel=3) + ReLU
    for (int oc = 0; oc < CONV1_OUT; ++oc)
    {
        for (int t = 0; t < SEQ_LEN; ++t)
        {
            float v = m->conv1_b[oc];
            for (int k = 0; k < 3; ++k)
            {
                int in_t = t + k - 1;
                float in_v = 0.0f;
                if (in_t >= 0 && in_t < SEQ_LEN)
                {
                    in_v = x[in_t];
                }
                v += m->conv1_w[oc][k] * in_v;
            }
            conv1[oc][t] = relu(v);
        }
    }

    // MaxPool1d(k=2, stride=2)
    for (int c = 0; c < CONV1_OUT; ++c)
    {
        for (int i = 0; i < POOL1_LEN; ++i)
        {
            float a = conv1[c][2 * i];
            float b = conv1[c][2 * i + 1];
            pool1[c][i] = (a > b) ? a : b;
        }
    }

    // Conv2 (padding=1, kernel=3) + ReLU
    for (int oc = 0; oc < CONV2_OUT; ++oc)
    {
        for (int t = 0; t < POOL1_LEN; ++t)
        {
            float v = m->conv2_b[oc];
            for (int ic = 0; ic < CONV1_OUT; ++ic)
            {
                for (int k = 0; k < 3; ++k)
                {
                    int in_t = t + k - 1;
                    float in_v = 0.0f;
                    if (in_t >= 0 && in_t < POOL1_LEN)
                    {
                        in_v = pool1[ic][in_t];
                    }
                    v += m->conv2_w[oc][ic][k] * in_v;
                }
            }
            conv2[oc][t] = relu(v);
        }
    }

    // MaxPool1d(k=2, stride=2)
    for (int c = 0; c < CONV2_OUT; ++c)
    {
        for (int i = 0; i < POOL2_LEN; ++i)
        {
            float a = conv2[c][2 * i];
            float b = conv2[c][2 * i + 1];
            pool2[c][i] = (a > b) ? a : b;
        }
    }

    // Flatten
    int idx = 0;
    for (int c = 0; c < CONV2_OUT; ++c)
    {
        for (int t = 0; t < POOL2_LEN; ++t)
        {
            latent[idx++] = pool2[c][t];
        }
    }
}

static void classifier_forward(const Model *m, const float latent[LATENT_DIM], float logits[NUM_CLASSES])
{
    for (int c = 0; c < NUM_CLASSES; ++c)
    {
        float z = m->clf_b[c];
        for (int d = 0; d < LATENT_DIM; ++d)
        {
            z += m->clf_w[c][d] * latent[d];
        }
        logits[c] = z;
    }
}

static float softmax_cross_entropy(const float logits[NUM_CLASSES], int label, float probs[NUM_CLASSES])
{
    float max_logit = logits[0];
    for (int i = 1; i < NUM_CLASSES; ++i)
    {
        if (logits[i] > max_logit)
        {
            max_logit = logits[i];
        }
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < NUM_CLASSES; ++i)
    {
        probs[i] = expf(logits[i] - max_logit);
        sum_exp += probs[i];
    }
    for (int i = 0; i < NUM_CLASSES; ++i)
    {
        probs[i] /= sum_exp;
    }

    float p = probs[label];
    if (p < 1e-8f)
    {
        p = 1e-8f;
    }
    return -logf(p);
}

static void set_loss_magnitude(Model *m, float loss_val)
{
    float dynamic_ratio = m->base_sparsity + (loss_val * 0.02f);
    if (dynamic_ratio < 0.01f)
    {
        dynamic_ratio = 0.01f;
    }
    if (dynamic_ratio > 0.10f)
    {
        dynamic_ratio = 0.10f;
    }
    m->update_ratio_s = dynamic_ratio;
}

typedef struct
{
    float mag;
    int idx;
} MagEntry;

static int cmp_mag_desc(const void *a, const void *b)
{
    const MagEntry *ma = (const MagEntry *)a;
    const MagEntry *mb = (const MagEntry *)b;
    if (ma->mag < mb->mag)
        return 1;
    if (ma->mag > mb->mag)
        return -1;
    return 0;
}

static void apply_classifier_update(Model *m, const float grad_w[NUM_CLASSES][LATENT_DIM], const float grad_b[NUM_CLASSES], float lr)
{
    if (!m->use_sparsity)
    {
        for (int c = 0; c < NUM_CLASSES; ++c)
        {
            for (int d = 0; d < LATENT_DIM; ++d)
            {
                m->clf_w[c][d] -= lr * grad_w[c][d];
            }
            m->clf_b[c] -= lr * grad_b[c];
        }
        return;
    }

    const int total_w = NUM_CLASSES * LATENT_DIM;
    const int total_all = total_w + NUM_CLASSES;
    int k = (int)((float)total_all * m->update_ratio_s);

    if (k <= 0)
    {
        return;
    }
    if (k > total_all)
    {
        k = total_all;
    }

    MagEntry *entries = (MagEntry *)malloc((size_t)total_all * sizeof(MagEntry));
    if (!entries)
    {
        return;
    }

    int idx = 0;
    for (int c = 0; c < NUM_CLASSES; ++c)
    {
        for (int d = 0; d < LATENT_DIM; ++d)
        {
            entries[idx].mag = fabsf(grad_w[c][d]);
            entries[idx].idx = idx;
            idx++;
        }
    }
    for (int c = 0; c < NUM_CLASSES; ++c)
    {
        entries[idx].mag = fabsf(grad_b[c]);
        entries[idx].idx = idx;
        idx++;
    }

    qsort(entries, (size_t)total_all, sizeof(MagEntry), cmp_mag_desc);

    bool *active = (bool *)calloc((size_t)total_all, sizeof(bool));
    if (!active)
    {
        free(entries);
        return;
    }

    for (int i = 0; i < k; ++i)
    {
        active[entries[i].idx] = true;
    }

    idx = 0;
    for (int c = 0; c < NUM_CLASSES; ++c)
    {
        for (int d = 0; d < LATENT_DIM; ++d)
        {
            if (active[idx])
            {
                m->clf_w[c][d] -= lr * grad_w[c][d];
            }
            idx++;
        }
    }
    for (int c = 0; c < NUM_CLASSES; ++c)
    {
        if (active[idx])
        {
            m->clf_b[c] -= lr * grad_b[c];
        }
        idx++;
    }

    free(active);
    free(entries);
}

static void replay_buffer_init(ReplayBuffer *rb, int capacity)
{
    rb->capacity = capacity;
    rb->size = 0;
    rb->samples_seen = 0;

    rb->quantized = (int8_t *)malloc((size_t)capacity * LATENT_DIM * sizeof(int8_t));
    rb->min_vals = (float *)malloc((size_t)capacity * sizeof(float));
    rb->max_vals = (float *)malloc((size_t)capacity * sizeof(float));
    rb->labels = (int *)malloc((size_t)capacity * sizeof(int));

    if (!rb->quantized || !rb->min_vals || !rb->max_vals || !rb->labels)
    {
        fprintf(stderr, "Replay buffer allocation failed.\n");
        exit(1);
    }
}

static void replay_buffer_free(ReplayBuffer *rb)
{
    free(rb->quantized);
    free(rb->min_vals);
    free(rb->max_vals);
    free(rb->labels);

    rb->quantized = NULL;
    rb->min_vals = NULL;
    rb->max_vals = NULL;
    rb->labels = NULL;
    rb->size = 0;
}

static void quantize_4bit(const float latent[LATENT_DIM], int8_t q[LATENT_DIM], float *min_v, float *max_v)
{
    float mn = latent[0];
    float mx = latent[0];
    for (int i = 1; i < LATENT_DIM; ++i)
    {
        if (latent[i] < mn)
            mn = latent[i];
        if (latent[i] > mx)
            mx = latent[i];
    }

    float range = mx - mn;
    if (range == 0.0f)
    {
        range = 1e-5f;
    }

    for (int i = 0; i < LATENT_DIM; ++i)
    {
        float scaled = 15.0f * (latent[i] - mn) / range;
        int v = (int)lroundf(scaled);
        if (v < 0)
            v = 0;
        if (v > 15)
            v = 15;
        q[i] = (int8_t)v;
    }

    *min_v = mn;
    *max_v = mx;
}

static void dequantize_4bit(const int8_t q[LATENT_DIM], float min_v, float max_v, float out[LATENT_DIM])
{
    float range = max_v - min_v;
    for (int i = 0; i < LATENT_DIM; ++i)
    {
        out[i] = ((float)q[i] / 15.0f) * range + min_v;
    }
}

static void replay_buffer_add(ReplayBuffer *rb, const float latent[LATENT_DIM], int label)
{
    int8_t q[LATENT_DIM];
    float mn = 0.0f, mx = 0.0f;
    quantize_4bit(latent, q, &mn, &mx);

    rb->samples_seen += 1;
    int target_idx;

    if (rb->size < rb->capacity)
    {
        target_idx = rb->size;
        rb->size += 1;
    }
    else
    {
        long long j = rand() % rb->samples_seen;
        if (j >= rb->capacity)
        {
            return;
        }
        target_idx = (int)j;
    }

    memcpy(&rb->quantized[target_idx * LATENT_DIM], q, LATENT_DIM * sizeof(int8_t));
    rb->min_vals[target_idx] = mn;
    rb->max_vals[target_idx] = mx;
    rb->labels[target_idx] = label;
}

static int replay_buffer_sample(const ReplayBuffer *rb, int request, float *latents_out, int *labels_out)
{
    if (rb->size == 0 || request <= 0)
    {
        return 0;
    }

    int n = request < rb->size ? request : rb->size;
    int *idx = (int *)malloc((size_t)rb->size * sizeof(int));
    if (!idx)
    {
        return 0;
    }

    for (int i = 0; i < rb->size; ++i)
    {
        idx[i] = i;
    }
    shuffle_int_array(idx, rb->size);

    for (int i = 0; i < n; ++i)
    {
        int src = idx[i];
        const int8_t *q = &rb->quantized[src * LATENT_DIM];
        dequantize_4bit(q, rb->min_vals[src], rb->max_vals[src], &latents_out[i * LATENT_DIM]);
        labels_out[i] = rb->labels[src];
    }

    free(idx);
    return n;
}

static int argmax2(float a, float b)
{
    return (a >= b) ? 0 : 1;
}

static float train_model(Model *m,
                         const Sample *samples,
                         int n_samples,
                         int epochs,
                         float lr,
                         ReplayBuffer *rb,
                         bool use_replay,
                         const char *name)
{
    int *order = (int *)malloc((size_t)n_samples * sizeof(int));
    float *replay_latents = (float *)malloc((size_t)BATCH_SIZE * LATENT_DIM * sizeof(float));
    int *replay_labels = (int *)malloc((size_t)BATCH_SIZE * sizeof(int));
    if (!order || !replay_latents || !replay_labels)
    {
        fprintf(stderr, "Allocation failed in training.\n");
        exit(1);
    }

    printf("--- Training on %s ---\n", name);

    float last_acc = 0.0f;
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        for (int i = 0; i < n_samples; ++i)
        {
            order[i] = i;
        }
        shuffle_int_array(order, n_samples);

        double total_loss = 0.0;
        int correct = 0;

        for (int i = 0; i < n_samples; ++i)
        {
            const Sample *s = &samples[order[i]];

            float latent[LATENT_DIM];
            float logits[NUM_CLASSES];
            float probs[NUM_CLASSES];

            forward_feature_extractor(m, s->x, latent);
            classifier_forward(m, latent, logits);
            float loss_cur = softmax_cross_entropy(logits, s->y, probs);

            float grad_w[NUM_CLASSES][LATENT_DIM] = {{0}};
            float grad_b[NUM_CLASSES] = {0};

            for (int c = 0; c < NUM_CLASSES; ++c)
            {
                float dz = probs[c] - ((c == s->y) ? 1.0f : 0.0f);
                grad_b[c] += dz;
                for (int d = 0; d < LATENT_DIM; ++d)
                {
                    grad_w[c][d] += dz * latent[d];
                }
            }

            float total_loss_sample = loss_cur;

            if (use_replay && rb != NULL)
            {
                int n_rep = replay_buffer_sample(rb, BATCH_SIZE, replay_latents, replay_labels);
                if (n_rep > 0)
                {
                    float rep_grad_w[NUM_CLASSES][LATENT_DIM] = {{0}};
                    float rep_grad_b[NUM_CLASSES] = {0};
                    double rep_loss_sum = 0.0;

                    for (int r = 0; r < n_rep; ++r)
                    {
                        float rep_logits[NUM_CLASSES];
                        float rep_probs[NUM_CLASSES];
                        classifier_forward(m, &replay_latents[r * LATENT_DIM], rep_logits);
                        rep_loss_sum += softmax_cross_entropy(rep_logits, replay_labels[r], rep_probs);

                        for (int c = 0; c < NUM_CLASSES; ++c)
                        {
                            float dz = rep_probs[c] - ((c == replay_labels[r]) ? 1.0f : 0.0f);
                            rep_grad_b[c] += dz;
                            for (int d = 0; d < LATENT_DIM; ++d)
                            {
                                rep_grad_w[c][d] += dz * replay_latents[r * LATENT_DIM + d];
                            }
                        }
                    }

                    float inv_rep = 1.0f / (float)n_rep;
                    for (int c = 0; c < NUM_CLASSES; ++c)
                    {
                        rep_grad_b[c] *= inv_rep;
                        for (int d = 0; d < LATENT_DIM; ++d)
                        {
                            rep_grad_w[c][d] *= inv_rep;
                        }
                    }

                    float loss_rep = (float)(rep_loss_sum * inv_rep);
                    total_loss_sample = 0.5f * loss_cur + 0.5f * loss_rep;

                    for (int c = 0; c < NUM_CLASSES; ++c)
                    {
                        grad_b[c] = 0.5f * grad_b[c] + 0.5f * rep_grad_b[c];
                        for (int d = 0; d < LATENT_DIM; ++d)
                        {
                            grad_w[c][d] = 0.5f * grad_w[c][d] + 0.5f * rep_grad_w[c][d];
                        }
                    }
                }
            }

            set_loss_magnitude(m, total_loss_sample);
            apply_classifier_update(m, grad_w, grad_b, lr);

            if (rb != NULL && !use_replay)
            {
                replay_buffer_add(rb, latent, s->y);
            }

            total_loss += total_loss_sample;
            int pred = argmax2(logits[0], logits[1]);
            if (pred == s->y)
            {
                correct++;
            }
        }

        last_acc = 100.0f * (float)correct / (float)n_samples;
        printf("Epoch %d/%d | Loss: %.4f | Acc: %.2f%%\n",
               epoch + 1,
               epochs,
               (float)(total_loss / (double)n_samples),
               last_acc);
    }

    free(order);
    free(replay_latents);
    free(replay_labels);
    return last_acc;
}

static float evaluate_model(const Model *m, const Sample *samples, int n_samples, const char *name)
{
    int correct = 0;
    for (int i = 0; i < n_samples; ++i)
    {
        float latent[LATENT_DIM];
        float logits[NUM_CLASSES];

        forward_feature_extractor(m, samples[i].x, latent);
        classifier_forward(m, latent, logits);

        int pred = argmax2(logits[0], logits[1]);
        if (pred == samples[i].y)
        {
            correct++;
        }
    }

    float acc = 100.0f * (float)correct / (float)n_samples;
    printf("Accuracy on %s: %.2f%%\n", name, acc);
    return acc;
}

static void check_sparsity(const Model *m, float loss_hint)
{
    int total_all = NUM_CLASSES * LATENT_DIM + NUM_CLASSES;
    int k = (int)((float)total_all * m->update_ratio_s);
    float pct = 100.0f * (float)k / (float)total_all;

    printf("[Check] Sparse update target at loss %.4f: updating ~%d/%d params (%.2f%%).\n",
           loss_hint,
           k,
           total_all,
           pct);
}

int main(void)
{
    srand(42);

    printf("Generating Synthetic Data...\n");
    Sample *base_train = (Sample *)malloc(TRAIN_SAMPLES * sizeof(Sample));
    Sample *base_test = (Sample *)malloc(TEST_SAMPLES * sizeof(Sample));
    Sample *evol_train = (Sample *)malloc(TRAIN_SAMPLES * sizeof(Sample));
    Sample *evol_test = (Sample *)malloc(TEST_SAMPLES * sizeof(Sample));

    if (!base_train || !base_test || !evol_train || !evol_test)
    {
        fprintf(stderr, "Dataset allocation failed.\n");
        return 1;
    }

    generate_synthetic_data(base_train, TRAIN_SAMPLES, false);
    generate_synthetic_data(base_test, TEST_SAMPLES, false);
    generate_synthetic_data(evol_train, TRAIN_SAMPLES, true);
    generate_synthetic_data(evol_test, TEST_SAMPLES, true);

    printf("\n==================================================\n");
    printf("Test Pipeline 1: STANDARD TRAINING (100%% updates)\n");
    printf("==================================================\n");

    Model model_std;
    init_model(&model_std, false);

    train_model(&model_std,
                base_train,
                TRAIN_SAMPLES,
                BASE_EPOCHS,
                LEARNING_RATE,
                NULL,
                false,
                "Baseline Data (Standard)");

    float acc_base_std = evaluate_model(&model_std, base_test, TEST_SAMPLES, "Baseline Test Split");

    train_model(&model_std,
                evol_train,
                TRAIN_SAMPLES,
                EVOL_EPOCHS,
                LEARNING_RATE,
                NULL,
                false,
                "Evolved Data (Naive CL without Replay)");

    float acc_evol_std = evaluate_model(&model_std, evol_test, TEST_SAMPLES, "Evolved Test Split");
    float acc_base_cl_std = evaluate_model(&model_std, base_test, TEST_SAMPLES, "Baseline Test Split (After Naive CL)");

    printf("\n==================================================\n");
    printf("Test Pipeline 2: SPARSE TRAINING (~5%% updates) + QLR (100 sample capacity)\n");
    printf("==================================================\n");

    Model model_sparse;
    init_model(&model_sparse, true);

    ReplayBuffer rb;
    replay_buffer_init(&rb, BUFFER_CAPACITY);

    train_model(&model_sparse,
                base_train,
                TRAIN_SAMPLES,
                BASE_EPOCHS,
                LEARNING_RATE,
                &rb,
                false,
                "Baseline Data (w/ sparsity & buffer fill)");

    check_sparsity(&model_sparse, 1.0f);
    float acc_base_sparse = evaluate_model(&model_sparse, base_test, TEST_SAMPLES, "Baseline Test Split");

    train_model(&model_sparse,
                evol_train,
                TRAIN_SAMPLES,
                EVOL_EPOCHS,
                LEARNING_RATE,
                &rb,
                true,
                "Evolved Data (w/ Sparse updates & QLR)");

    check_sparsity(&model_sparse, 1.0f);
    float acc_evol_sparse = evaluate_model(&model_sparse, evol_test, TEST_SAMPLES, "Evolved Test Split");
    float acc_base_cl_sparse = evaluate_model(&model_sparse, base_test, TEST_SAMPLES, "Baseline Test Split (After Sparse CL + QLR)");

    printf("\n==================================================\n");
    printf("Final Output Metrics Report\n");
    printf("==================================================\n");
    printf("a) Standard Training vs. Sparse Training (Initial Learning):\n");
    printf("   Baseline Acc: Standard = %.2f%% | Sparse = %.2f%%\n", acc_base_std, acc_base_sparse);
    printf("\n");
    printf("b) Accuracy maintenance on old data (Catastrophic Forgetting test):\n");
    printf("   Naive Standard Continual Learning Maintained Acc : %.2f%%\n", acc_base_cl_std);
    printf("   Sparse Update + 4-bit QLR Maintained Acc         : %.2f%%\n", acc_base_cl_sparse);
    printf("\n");
    printf("c) New Evolved Data Classification:\n");
    printf("   Evolved Acc: Standard = %.2f%% | Sparse + QLR = %.2f%%\n", acc_evol_std, acc_evol_sparse);

    if (acc_evol_sparse > 85.0f && acc_base_cl_sparse > 85.0f)
    {
        printf("\nSUCCESS FLAG: Model achieved >85%% accuracy on both datasets utilizing the 5%% updates engine and QLR!\n");
    }
    else
    {
        printf("\nWARNING: Accuracy threshold not fully met. Hyperparameters might need tuning depending on initialized weight states.\n");
    }

    replay_buffer_free(&rb);
    free(base_train);
    free(base_test);
    free(evol_train);
    free(evol_test);

    return 0;
}
