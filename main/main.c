#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_adc/adc_oneshot.h"

#define SOIL_ADC_UNIT ADC_UNIT_1
#define SOIL_ADC_CHANNEL ADC_CHANNEL_6 // GPIO 34 on ESP32

#define SAMPLING_PERIOD_MS 50 // 50ms interval = 20Hz sample frequency

#define SEQ_LEN 100
#define CONV1_OUT 8
#define CONV2_OUT 16
#define POOL1_LEN (SEQ_LEN / 2)
#define POOL2_LEN (POOL1_LEN / 2)
#define LATENT_DIM (CONV2_OUT * POOL2_LEN)
#define NUM_CLASSES 2

#define BUFFER_CAPACITY 50  // Tighter bound for runtime memory safety
#define BATCH_SIZE 8        // Optimized mini-batch sampling for embedded pass
#define LEARNING_RATE 0.01f
#define BASE_SPARSITY 0.05f

#ifndef max
#define max(a,b) ((a) > (b) ? (a) : (b))
#endif

typedef struct {
    float x[SEQ_LEN];
    int y;
} RuntimeSample;

typedef struct {
    int capacity;
    int size;
    long long samples_seen;
    int8_t quantized[BUFFER_CAPACITY][LATENT_DIM]; 
    float min_vals[BUFFER_CAPACITY];
    float max_vals[BUFFER_CAPACITY];
    int labels[BUFFER_CAPACITY];
} StaticReplayBuffer;

typedef struct {
    float conv1_w[CONV1_OUT][3];
    float conv1_b[CONV1_OUT];
    float conv2_w[CONV2_OUT][CONV1_OUT][3];
    float conv2_b[CONV2_OUT];
    float clf_w[NUM_CLASSES][LATENT_DIM];
    float clf_b[NUM_CLASSES];
    bool use_sparsity;
    float base_sparsity;
    float update_ratio_s;
} Model;

typedef struct {
    float mag;
    int idx;
} MagEntry;

// Instantiations
Model nn_model;
StaticReplayBuffer memory_buffer;
float current_stream[SEQ_LEN];

static adc_oneshot_unit_handle_t adc1_handle;

// --- Embedded Static Memory Helpers ---
static int cmp_mag_desc(const void *a, const void *b) {
    const MagEntry *ma = (const MagEntry *)a;
    const MagEntry *mb = (const MagEntry *)b;
    return (ma->mag < mb->mag) - (ma->mag > mb->mag);
}

static void init_model_he(Model *m, bool use_sparsity) {
    memset(m, 0, sizeof(*m));
    m->use_sparsity = use_sparsity;
    m->base_sparsity = BASE_SPARSITY;
    m->update_ratio_s = BASE_SPARSITY;

    // Xavier/He-ish deterministic setup scaled for 1D signals
    float sc = sqrtf(2.0f / (float)LATENT_DIM);
    for (int c = 0; c < NUM_CLASSES; ++c) {
        for (int d = 0; d < LATENT_DIM; ++d) m->clf_w[c][d] = ((float)rand() / (float)RAND_MAX) * sc;
        m->clf_b[c] = 0.0f;
    }
}

// --- Algorithmic Pipeline Forward Passes ---
static inline float relu(float x) { return x > 0.0f ? x : 0.0f; }

static void forward_feature_extractor(const Model *m, const float x[SEQ_LEN], float latent[LATENT_DIM]) {
    static float conv1[CONV1_OUT][SEQ_LEN];
    static float pool1[CONV1_OUT][POOL1_LEN];
    static float conv2[CONV2_OUT][POOL1_LEN];
    static float pool2[CONV2_OUT][POOL2_LEN];

    for (int oc = 0; oc < CONV1_OUT; ++oc) {
        for (int t = 0; t < SEQ_LEN; ++t) {
            float v = m->conv1_b[oc];
            for (int k = 0; k < 3; ++k) {
                int in_t = t + k - 1;
                float in_v = (in_t >= 0 && in_t < SEQ_LEN) ? x[in_t] : 0.0f;
                v += m->conv1_w[oc][k] * in_v;
            }
            conv1[oc][t] = relu(v);
        }
    }

    for (int c = 0; c < CONV1_OUT; ++c) {
        for (int i = 0; i < POOL1_LEN; ++i) {
            float a = conv1[c][2 * i];
            float b = conv1[c][2 * i + 1];
            pool1[c][i] = (a > b) ? a : b;
        }
    }

    for (int oc = 0; oc < CONV2_OUT; ++oc) {
        for (int t = 0; t < POOL1_LEN; ++t) {
            float v = m->conv2_b[oc];
            for (int ic = 0; ic < CONV1_OUT; ++ic) {
                for (int k = 0; k < 3; ++k) {
                    int in_t = t + k - 1;
                    float in_v = (in_t >= 0 && in_t < POOL1_LEN) ? pool1[ic][in_t] : 0.0f;
                    v += m->conv2_w[oc][ic][k] * in_v;
                }
            }
            conv2[oc][t] = relu(v);
        }
    }

    for (int c = 0; c < CONV2_OUT; ++c) {
        for (int i = 0; i < POOL2_LEN; ++i) {
            float a = conv2[c][2 * i];
            float b = conv2[c][2 * i + 1];
            pool2[c][i] = (a > b) ? a : b;
        }
    }

    int idx = 0;
    for (int c = 0; c < CONV2_OUT; ++c) {
        for (int t = 0; t < POOL2_LEN; ++t) latent[idx++] = pool2[c][t];
    }
}

static void classifier_forward(const Model *m, const float latent[LATENT_DIM], float logits[NUM_CLASSES]) {
    for (int c = 0; c < NUM_CLASSES; ++c) {
        float z = m->clf_b[c];
        for (int d = 0; d < LATENT_DIM; ++d) z += m->clf_w[c][d] * latent[d];
        logits[c] = z;
    }
}

// --- Quantization & Backprop Layer Optimization ---
static void quantize_4bit(const float latent[LATENT_DIM], int8_t q[LATENT_DIM], float *min_v, float *max_v) {
    float mn = latent[0], mx = latent[0];
    for (int i = 1; i < LATENT_DIM; ++i) {
        if (latent[i] < mn) mn = latent[i];
        if (latent[i] > mx) mx = latent[i];
    }
    float range = (mx - mn == 0.0f) ? 1e-5f : (mx - mn);
    for (int i = 0; i < LATENT_DIM; ++i) {
        int v = (int)lroundf(15.0f * (latent[i] - mn) / range);
        q[i] = (int8_t)(v < 0 ? 0 : (v > 15 ? 15 : v));
    }
    *min_v = mn; *max_v = mx;
}

static void __attribute__((unused)) dequantize_4bit(const int8_t q[LATENT_DIM], float min_v, float max_v, float out[LATENT_DIM]) {
    float range = max_v - min_v;
    for (int i = 0; i < LATENT_DIM; ++i) out[i] = ((float)q[i] / 15.0f) * range + min_v;
}

static void apply_classifier_update_static(Model *m, const float grad_w[NUM_CLASSES][LATENT_DIM], const float grad_b[NUM_CLASSES], float lr) {
    const int total_all = (NUM_CLASSES * LATENT_DIM) + NUM_CLASSES;
    int k = (int)((float)total_all * m->update_ratio_s);
    if (k <= 0) return;

    // Eliminating malloc: Using statically declared buffer allocation safely reused on tasks
    static MagEntry entries[NUM_CLASSES * LATENT_DIM + NUM_CLASSES];
    static bool active_mask[NUM_CLASSES * LATENT_DIM + NUM_CLASSES];
    memset(active_mask, 0, sizeof(active_mask));

    int idx = 0;
    for (int c = 0; c < NUM_CLASSES; ++c) {
        for (int d = 0; d < LATENT_DIM; ++d) {
            entries[idx].mag = fabsf(grad_w[c][d]);
            entries[idx].idx = idx;
            idx++;
        }
    }
    for (int c = 0; c < NUM_CLASSES; ++c) {
        entries[idx].mag = fabsf(grad_b[c]);
        entries[idx].idx = idx;
        idx++;
    }

    qsort(entries, total_all, sizeof(MagEntry), cmp_mag_desc);
    for (int i = 0; i < k; ++i) active_mask[entries[i].idx] = true;

    idx = 0;
    for (int c = 0; c < NUM_CLASSES; ++c) {
        for (int d = 0; d < LATENT_DIM; ++d) {
            if (active_mask[idx]) m->clf_w[c][d] -= lr * grad_w[c][d];
            idx++;
        }
    }
    for (int c = 0; c < NUM_CLASSES; ++c) {
        if (active_mask[idx]) m->clf_b[c] -= lr * grad_b[c];
        idx++;
    }
}

// --- Reservoir Memory Manipulation ---
void buffer_add(StaticReplayBuffer *rb, const float latent[LATENT_DIM], int label) {
    int8_t q[LATENT_DIM];
    float mn, mx;
    quantize_4bit(latent, q, &mn, &mx);
    
    rb->samples_seen++;
    int target = 0;
    if (rb->size < BUFFER_CAPACITY) {
        target = rb->size;
        rb->size++;
    } else {
        int j = rand() % rb->samples_seen;
        if (j >= BUFFER_CAPACITY) return;
        target = j;
    }
    memcpy(rb->quantized[target], q, LATENT_DIM);
    rb->min_vals[target] = mn;
    rb->max_vals[target] = mx;
    rb->labels[target] = label;
}

static void init_adc(void) {
    adc_oneshot_unit_init_cfg_t init_config = {
        .unit_id = SOIL_ADC_UNIT,
        .ulp_mode = ADC_ULP_MODE_DISABLE,
    };
    ESP_ERROR_CHECK(adc_oneshot_new_unit(&init_config, &adc1_handle));

    adc_oneshot_chan_cfg_t config = {
        .bitwidth = ADC_BITWIDTH_12,
        .atten = ADC_ATTEN_DB_12,
    };
    ESP_ERROR_CHECK(adc_oneshot_config_channel(adc1_handle, SOIL_ADC_CHANNEL, &config));
}

// --- Setup Execution Blocks ---
void setup() {
    init_adc();
    vTaskDelay(pdMS_TO_TICKS(2000));
    
    printf("[System Boot] Initializing Neuromorphic Continual Learning Infrastructure...\n");
    init_model_he(&nn_model, true);
    
    memory_buffer.capacity = BUFFER_CAPACITY;
    memory_buffer.size = 0;
    memory_buffer.samples_seen = 0;
    
    printf("[System Ready] Beginning Active Sampling Loops.\n");
}

void loop() {
    printf("\n[Action] Recording target sequence window (100 sequential measurements)...\n");
    
    // Acquire a continuous window stream from the soil moisture sensor
    for (int i = 0; i < SEQ_LEN; i++) {
        int raw_in = 0;
        esp_err_t err = adc_oneshot_read(adc1_handle, SOIL_ADC_CHANNEL, &raw_in);
        if (err != ESP_OK) {
            raw_in = 0;
        }
        // Normalize 12-bit analog input (0-4095) to float spectrum (-1.0 to 1.0)
        current_stream[i] = ((float)raw_in / 2047.5f) - 1.0f;
        vTaskDelay(pdMS_TO_TICKS(SAMPLING_PERIOD_MS));
    }

    // Step 1: Compute Feature Pipeline Inference
    static float latent_features[LATENT_DIM];
    static float output_logits[NUM_CLASSES];
    forward_feature_extractor(&nn_model, current_stream, latent_features);
    classifier_forward(&nn_model, latent_features, output_logits);

    int prediction = (output_logits[0] >= output_logits[1]) ? 0 : 1;
    printf("[Prediction Inference Result] Deduced Operational Soil Category: %d\n", prediction);

    // Step 2: Simulate On-Device Edge Ground Truth Integration
    // In operation, substitute this mock target with anomalous boundaries or targeted state overrides
    int ground_truth_label = (current_stream[0] > 0.0f) ? 0 : 1; 

    // Step 3: Execute Localized Sparse-Gradient Update Backpropagation Sequence
    static float probs[NUM_CLASSES];
    float max_l = max(output_logits[0], output_logits[1]);
    float sum_exp = expf(output_logits[0] - max_l) + expf(output_logits[1] - max_l);
    probs[0] = expf(output_logits[0] - max_l) / sum_exp;
    probs[1] = expf(output_logits[1] - max_l) / sum_exp;

    static float grad_w[NUM_CLASSES][LATENT_DIM];
    memset(grad_w, 0, sizeof(grad_w));
    static float grad_b[NUM_CLASSES];
    memset(grad_b, 0, sizeof(grad_b));

    for (int c = 0; c < NUM_CLASSES; ++c) {
        float dz = probs[c] - ((c == ground_truth_label) ? 1.0f : 0.0f);
        grad_b[c] = dz;
        for (int d = 0; d < LATENT_DIM; ++d) grad_w[c][d] = dz * latent_features[d];
    }

    // Update weights with dynamic magnitude adjustments
    nn_model.update_ratio_s = BASE_SPARSITY + (-logf(max(probs[ground_truth_label], 1e-5f)) * 0.01f);
    if (nn_model.update_ratio_s > 0.10f) nn_model.update_ratio_s = 0.10f;

    apply_classifier_update_static(&nn_model, grad_w, grad_b, LEARNING_RATE);
    buffer_add(&memory_buffer, latent_features, ground_truth_label);

    printf("[Edge Learn System Update Log] Sparsity Update Ratio Cap Locked: %.2f%%\n", nn_model.update_ratio_s * 100.0f);
    printf("[Memory Status Metrics Log] Total Consolidated Reservoir Buffer Instances: %d / %d\n", memory_buffer.size, BUFFER_CAPACITY);
    
    vTaskDelay(pdMS_TO_TICKS(5000)); // Chill execution intervals between continuous tracking operations
}

void app_main(void) {
    setup();
    while (1) {
        loop();
    }
}
