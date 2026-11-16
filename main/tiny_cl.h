#ifndef TINY_CL_H
#define TINY_CL_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include "model_config.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    float mag;
    int idx;
} SparseGradEntry;

typedef struct {
    int capacity;
    int size;
    long long samples_seen;
    // 4-bit packed latent representation (LATENT_DIM / 2 bytes)
    uint8_t quantized[BUFFER_CAPACITY][LATENT_DIM / 2];
    float min_vals[BUFFER_CAPACITY];
    float max_vals[BUFFER_CAPACITY];
    int8_t labels[BUFFER_CAPACITY];
} StaticReplayBuffer;

typedef struct {
    float clf_w[NUM_CLASSES][LATENT_DIM];
    float clf_b[NUM_CLASSES];
    bool use_sparsity;
    float base_sparsity;
    float current_sparsity;
    uint32_t total_adaptation_steps;
    int last_writes;          /* weights actually written on the last step */
} TinyCLModel;

/* Selection strategy for the sparse head update. */
typedef enum {
    TINY_CL_UPDATE_DENSE = 0,   /* update every weight                       */
    TINY_CL_UPDATE_SORT,        /* top-k via full comparison sort (baseline) */
    TINY_CL_UPDATE_INTROSELECT, /* top-k via quickselect partition, O(n) exp. */
    TINY_CL_UPDATE_THRESHOLD    /* running-threshold heuristic, single pass   */
} TinyCLUpdateRule;

/*
 * Static footprint report. Unlike the previous revision, this accounts for
 * ALL file-scope and function-local static storage in this translation
 * unit, including gradient, selection and dequantisation scratch. Those
 * three were previously omitted, understating the true figure by 2.5x.
 */
typedef struct {
    uint32_t head_bytes;        /* head params + adaptation state          */
    uint32_t reservoir_bytes;   /* quantized latent replay reservoir       */
    uint32_t activation_bytes;  /* extractor activation scratch            */
    uint32_t gradient_bytes;    /* gradient accumulator + selection scratch */
    uint32_t dequant_bytes;     /* replay dequantisation staging           */
    uint32_t total_sram_bytes;  /* sum of the above (SRAM)                 */
    uint32_t flash_const_bytes; /* frozen weights, .rodata (NOT SRAM)      */
    int buffer_occupancy;
    float current_sparsity;
    TinyCLUpdateRule update_rule;
} TinyCLStats;

/**
 * @brief Initialize TinyCL engine, weights, and replay memory buffer.
 * @param rule Which head-update selection strategy to use.
 */
void tiny_cl_init(TinyCLUpdateRule rule);

/**
 * @brief Enable or disable replay rehearsal (for ablation studies).
 */
void tiny_cl_set_replay_enabled(bool enabled);

/**
 * @brief Forward pass through 1D-convolutional feature extractor.
 * @param input_window Raw 1D vibration magnitude signal window of length SEQ_LEN
 * @param latent_out Output array of size LATENT_DIM
 */
void tiny_cl_extract_features(const float input_window[SEQ_LEN], float latent_out[LATENT_DIM]);

/**
 * @brief Predict class label and confidence probabilities for a given latent representation.
 * @param latent Latent feature vector of size LATENT_DIM
 * @param logits_out Output array for unnormalized class logits (size NUM_CLASSES)
 * @param probs_out Output array for normalized probabilities (size NUM_CLASSES)
 * @return Predicted class index (0..NUM_CLASSES-1)
 */
int tiny_cl_predict(const float latent[LATENT_DIM], float logits_out[NUM_CLASSES], float probs_out[NUM_CLASSES]);

/**
 * @brief Perform magnitude-based sparse gradient update on the linear classifier layer.
 * @param latent Latent feature vector of size LATENT_DIM
 * @param target_label True class label for supervised adaptation
 */
void tiny_cl_adapt_sparse(const float latent[LATENT_DIM], int target_label);

/**
 * @brief Per-window standardisation applied before feature extraction.
 *        Exposed so host tooling can verify numerical parity.
 */
void tiny_cl_standardise_window(const float in[SEQ_LEN], float out[SEQ_LEN]);

/**
 * @brief Store a sample in the 4-bit quantized reservoir replay buffer.
 * @param latent Latent feature vector of size LATENT_DIM
 * @param label Target class label
 */
void tiny_cl_replay_store(const float latent[LATENT_DIM], int label);

/**
 * @brief Sample a mini-batch from replay memory and execute replay adaptation pass.
 */
void tiny_cl_replay_adapt_step(void);

/**
 * @brief Retrieve current memory and adaptation statistics.
 * @param stats Pointer to output TinyCLStats struct
 */
void tiny_cl_get_stats(TinyCLStats *stats);

#ifdef __cplusplus
}
#endif

#endif // TINY_CL_H
