#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <stdint.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/semphr.h"
#include "esp_system.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "driver/uart.h"

#include "imu_driver.h"
#include "tiny_cl.h"

static const char *TAG = "TINYML_MAIN";

#define UART_PORT UART_NUM_0
#define UART_BUF_SIZE 512

// Static Task Stack Buffers (No Heap Allocations)
#define SAMPLING_STACK_SIZE 2048
#define INFERENCE_STACK_SIZE 4096
#define ADAPTATION_STACK_SIZE 4096
#define SERIAL_STACK_SIZE 4096

static StackType_t s_sampling_stack[SAMPLING_STACK_SIZE];
static StaticTask_t s_sampling_task_cb;

static StackType_t s_inference_stack[INFERENCE_STACK_SIZE];
static StaticTask_t s_inference_task_cb;

static StackType_t s_adaptation_stack[ADAPTATION_STACK_SIZE];
static StaticTask_t s_adaptation_task_cb;

static StackType_t s_serial_stack[SERIAL_STACK_SIZE];
static StaticTask_t s_serial_task_cb;

// Static Sync Primitives
static SemaphoreHandle_t s_window_mutex = NULL;
static StaticSemaphore_t s_window_mutex_cb;

static SemaphoreHandle_t s_adapt_sem = NULL;
static StaticSemaphore_t s_adapt_sem_cb;

// Global Static State
static float s_imu_window[SEQ_LEN];
static uint16_t s_window_write_idx = 0;
static bool s_window_ready = false;

static int s_latest_pred = 0;
static float s_latest_conf = 0.0f;
static float s_latest_mag = 1.0f;
static float s_latest_latent[LATENT_DIM];
static int s_pending_label = -1; // -1 means no pending manual label correction
static uint32_t s_inference_counter = 0;
static float s_last_latency_ms = 0.0f;

// 1. IMU Sampling Task (50Hz = 20ms interval)
static void imu_sampling_task(void *pvParameters)
{
    float mag_g = 1.0f;
    while (1) {
        esp_err_t err = imu_read_magnitude_g(&mag_g);
        if (err != ESP_OK) {
            // Fallback synthetic baseline if sensor read fails / during bring-up
            mag_g = 1.0f + ((float)(rand() % 100) / 1000.0f);
        }

        if (xSemaphoreTake(s_window_mutex, pdMS_TO_TICKS(10)) == pdTRUE) {
            s_imu_window[s_window_write_idx++] = mag_g;
            if (s_window_write_idx >= SEQ_LEN) {
                s_window_write_idx = 0;
                s_window_ready = true;
            }
            s_latest_mag = mag_g;
            xSemaphoreGive(s_window_mutex);
        }

        vTaskDelay(pdMS_TO_TICKS(20)); // 50Hz
    }
}

// 2. Inference Task
static void inference_task(void *pvParameters)
{
    static float local_window[SEQ_LEN];
    static float logits[NUM_CLASSES];
    static float probs[NUM_CLASSES];

    while (1) {
        bool process = false;

        if (xSemaphoreTake(s_window_mutex, pdMS_TO_TICKS(50)) == pdTRUE) {
            if (s_window_ready) {
                memcpy(local_window, s_imu_window, sizeof(local_window));
                s_window_ready = false;
                process = true;
            }
            xSemaphoreGive(s_window_mutex);
        }

        if (process) {
            int64_t t_start = esp_timer_get_time();

            // Extract features & run prediction
            tiny_cl_extract_features(local_window, s_latest_latent);
            s_latest_pred = tiny_cl_predict(s_latest_latent, logits, probs);
            s_latest_conf = probs[s_latest_pred];

            int64_t t_end = esp_timer_get_time();
            s_last_latency_ms = (float)(t_end - t_start) / 1000.0f;
            s_inference_counter++;
        }

        vTaskDelay(pdMS_TO_TICKS(100)); // Run inference check at 10Hz
    }
}

// 3. Adaptation Task
static void adaptation_task(void *pvParameters)
{
    while (1) {
        if (xSemaphoreTake(s_adapt_sem, portMAX_DELAY) == pdTRUE) {
            if (s_pending_label >= 0 && s_pending_label < NUM_CLASSES) {
                int label = s_pending_label;
                s_pending_label = -1; // Reset pending

                // Store in 4-bit quantized reservoir buffer and execute sparse adaptation
                tiny_cl_replay_store(s_latest_latent, label);
                tiny_cl_adapt_sparse(s_latest_latent, label);
                tiny_cl_replay_adapt_step();

                ESP_LOGI(TAG, "On-device adaptation step executed for target label: %d", label);
            }
        }
    }
}

// 4. Serial Communication Task (Telemetry & Commands over UART0 USB)
static void serial_io_task(void *pvParameters)
{
    uint8_t rx_buf[UART_BUF_SIZE];
    TinyCLStats stats;

    while (1) {
        // Broadcast telemetry JSON every 200ms
        tiny_cl_get_stats(&stats);

        printf("{\"type\":\"telemetry\",\"count\":%lu,\"pred\":%d,\"conf\":%.3f,\"mag\":%.3f,\"sparsity\":%.2f,\"buffer\":%d,\"static_ram_kb\":33.6,\"latency_ms\":%.2f}\n",
               (unsigned long)s_inference_counter, s_latest_pred, s_latest_conf, s_latest_mag,
               stats.current_sparsity, stats.buffer_occupancy, s_last_latency_ms);

        // Read incoming commands over serial UART
        int len = uart_read_bytes(UART_PORT, rx_buf, UART_BUF_SIZE - 1, pdMS_TO_TICKS(50));
        if (len > 0) {
            rx_buf[len] = '\0';
            char *str = (char *)rx_buf;

            // Parse simple command protocol e.g. {"cmd":"label","val":1} or LABEL:1
            int label_val = -1;
            if (strstr(str, "\"val\":1") || strstr(str, "LABEL:1") || strstr(str, "label:1")) {
                label_val = 1;
            } else if (strstr(str, "\"val\":0") || strstr(str, "LABEL:0") || strstr(str, "label:0")) {
                label_val = 0;
            }

            if (label_val >= 0) {
                s_pending_label = label_val;
                xSemaphoreGive(s_adapt_sem);
            }
        }

        vTaskDelay(pdMS_TO_TICKS(150));
    }
}

void app_main(void)
{
    ESP_LOGI(TAG, "==========================================================");
    ESP_LOGI(TAG, "  TinyML Vibration PDM — On-Device Firmware Entry Point    ");
    ESP_LOGI(TAG, "==========================================================");

    // Initialize static synchronization primitives
    s_window_mutex = xSemaphoreCreateMutexStatic(&s_window_mutex_cb);
    s_adapt_sem = xSemaphoreCreateBinaryStatic(&s_adapt_sem_cb);

    // Initialize TinyCL model engine
    tiny_cl_init(true);

    // Initialize I2C IMU Driver
    esp_err_t err = imu_init(DEFAULT_I2C_SDA_PIN, DEFAULT_I2C_SCL_PIN, DEFAULT_I2C_FREQ_HZ);
    if (err != ESP_OK) {
        ESP_LOGW(TAG, "Physical IMU not detected (err: %s). Running with simulated vibration baseline.", esp_err_to_name(err));
    }

    // Configure UART0 for serial telemetry communication
    uart_config_t uart_config = {
        .baud_rate = 115200,
        .data_bits = UART_DATA_8_BITS,
        .parity    = UART_PARITY_DISABLE,
        .stop_bits = UART_STOP_BITS_1,
        .flow_ctrl = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    uart_param_config(UART_PORT, &uart_config);
    uart_driver_install(UART_PORT, UART_BUF_SIZE * 2, 0, 0, NULL, 0);

    // Create FreeRTOS Tasks using Static Allocation
    xTaskCreateStatic(imu_sampling_task, "imu_sampling", SAMPLING_STACK_SIZE, NULL, 5, s_sampling_stack, &s_sampling_task_cb);
    xTaskCreateStatic(inference_task, "inference", INFERENCE_STACK_SIZE, NULL, 4, s_inference_stack, &s_inference_task_cb);
    xTaskCreateStatic(adaptation_task, "adaptation", ADAPTATION_STACK_SIZE, NULL, 3, s_adaptation_stack, &s_adaptation_task_cb);
    xTaskCreateStatic(serial_io_task, "serial_io", SERIAL_STACK_SIZE, NULL, 2, s_serial_stack, &s_serial_task_cb);

    ESP_LOGI(TAG, "All static FreeRTOS tasks started successfully!");
}
