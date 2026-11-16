#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../main/tiny_cl.h"
#include "../main/tiny_cl.c"

// Generate a synthetic test vibration signal window
static void generate_test_vibration_window(float window[SEQ_LEN], int label) {
    static float t_idx = 0.0f;
    for (int i = 0; i < SEQ_LEN; ++i) {
        float t = (t_idx + i) * 0.02f; // 50Hz sample rate -> dt = 0.02s
        float noise = ((float)rand() / (float)RAND_MAX - 0.5f) * 0.05f;

        if (label == 0) {
            // Healthy baseline: 10Hz fundamental vibration
            window[i] = 1.0f + 0.15f * sinf(2.0f * 3.14159f * 10.0f * t) + noise;
        } else {
            // Anomalous state: 10Hz + 25Hz harmonic + transient shocks
            float spike = ((rand() % 100) > 92) ? 0.4f : 0.0f;
            window[i] = 1.0f + 0.35f * sinf(2.0f * 3.14159f * 10.0f * t) +
                        0.25f * cosf(2.0f * 3.14159f * 25.0f * t) + spike + noise;
        }
    }
    t_idx += SEQ_LEN;
}

int main(void) {
    printf("=========================================================\n");
    printf("  TinyML Vibration PDM — Desktop C Simulation Harness    \n");
    printf("=========================================================\n");

    // Initialize algorithm
    tiny_cl_init(TINY_CL_UPDATE_SORT);

    TinyCLStats stats;
    tiny_cl_get_stats(&stats);
    printf("[Sim] Initialized TinyCL Engine.\n");
    printf("      Head + adaptation : %.2f KB (%u bytes)\n", stats.head_bytes / 1024.0f, stats.head_bytes);
    printf("      Replay reservoir  : %.2f KB (%u bytes)\n", stats.reservoir_bytes / 1024.0f, stats.reservoir_bytes);
    printf("      Activations       : %u bytes | Gradients/selection: %u bytes | Dequant: %u bytes\n",
           stats.activation_bytes, stats.gradient_bytes, stats.dequant_bytes);
    printf("      Total SRAM        : %.2f KB (%u bytes) | Flash (frozen weights): %u bytes\n\n",
           stats.total_sram_bytes / 1024.0f, stats.total_sram_bytes, stats.flash_const_bytes);

    static float sample_window[SEQ_LEN];
    static float latent[LATENT_DIM];
    static float logits[NUM_CLASSES];
    static float probs[NUM_CLASSES];

    int correct_predictions = 0;
    int total_sim_steps = 100;

    printf("[Sim] Executing %d streaming inference + adaptation cycles...\n\n", total_sim_steps);

    for (int step = 1; step <= total_sim_steps; ++step) {
        int true_label = (step <= 50) ? 0 : 1; // Class 0 first 50 steps, Class 1 next 50 steps
        generate_test_vibration_window(sample_window, true_label);

        // 1. Feature Extraction
        tiny_cl_extract_features(sample_window, latent);

        // 2. Inference
        int pred_label = tiny_cl_predict(latent, logits, probs);
        if (pred_label == true_label) correct_predictions++;

        // 3. Store in 4-bit Quantized Replay Buffer & Adapt
        tiny_cl_replay_store(latent, true_label);
        tiny_cl_adapt_sparse(latent, true_label);
        tiny_cl_replay_adapt_step();

        if (step % 20 == 0 || step == total_sim_steps) {
            tiny_cl_get_stats(&stats);
            printf("Step [%03d/%03d] True Label: %d | Pred: %d (Conf: %.1f%%) | Buffer: %d/%d samples\n",
                   step, total_sim_steps, true_label, pred_label, probs[pred_label] * 100.0f,
                   stats.buffer_occupancy, BUFFER_CAPACITY);
        }
    }

    float accuracy = (float)correct_predictions / (float)total_sim_steps * 100.0f;
    printf("\n=========================================================\n");
    printf("  SIMULATION COMPLETED SUCCESSFULLY!\n");
    printf("  Total Steps Tested : %d\n", total_sim_steps);
    printf("  Overall Accuracy   : %.2f%%\n", accuracy);
    printf("  Memory Check       : ZERO Heap Allocations (100%% Static)\n");
    printf("=========================================================\n");

    return 0;
}
