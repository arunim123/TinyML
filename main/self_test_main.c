#include <stdio.h>
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_system.h"
#include "esp_log.h"
#include "imu_driver.h"

static const char *TAG = "IMU_SELF_TEST";

void app_main(void)
{
    ESP_LOGI(TAG, "=============================================");
    ESP_LOGI(TAG, "  TinyML Vibration PDM — Sensor Self-Test    ");
    ESP_LOGI(TAG, "=============================================");

    ESP_LOGI(TAG, "Initializing IMU (SDA: GPIO %d, SCL: GPIO %d)...", DEFAULT_I2C_SDA_PIN, DEFAULT_I2C_SCL_PIN);
    esp_err_t err = imu_init(DEFAULT_I2C_SDA_PIN, DEFAULT_I2C_SCL_PIN, DEFAULT_I2C_FREQ_HZ);

    if (err != ESP_OK) {
        ESP_LOGE(TAG, "IMU Initialization FAILED! Error code: %s", esp_err_to_name(err));
        ESP_LOGE(TAG, "Please check your physical wiring:");
        ESP_LOGE(TAG, "  - MPU6050 VCC  -> ESP32 3.3V");
        ESP_LOGE(TAG, "  - MPU6050 GND  -> ESP32 GND");
        ESP_LOGE(TAG, "  - MPU6050 SDA  -> ESP32 GPIO 21");
        ESP_LOGE(TAG, "  - MPU6050 SCL  -> ESP32 GPIO 22");
        return;
    }

    ESP_LOGI(TAG, "IMU Initialized successfully! Starting continuous reading loop (20Hz)...");

    int16_t rx, ry, rz;
    float ax, ay, az, mag;
    uint32_t sample_count = 0;

    while (1) {
        err = imu_read_raw(&rx, &ry, &rz);
        if (err == ESP_OK) {
            imu_read_accel_g(&ax, &ay, &az);
            imu_read_magnitude_g(&mag);

            sample_count++;
            ESP_LOGI(TAG, "[#%05lu] RAW: [%6d, %6d, %6d] | ACCEL(g): [X:%+.3f, Y:%+.3f, Z:%+.3f] | MAG: %.3fg",
                     sample_count, rx, ry, rz, ax, ay, az, mag);
        } else {
            ESP_LOGE(TAG, "I2C read failure: %s", esp_err_to_name(err));
        }

        vTaskDelay(pdMS_TO_TICKS(50)); // 50ms sampling interval
    }
}
