#ifndef IMU_DRIVER_H
#define IMU_DRIVER_H

#include <stdint.h>
#include <stdbool.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

// Default I2C pin configuration for ESP32
#define DEFAULT_I2C_SDA_PIN 21
#define DEFAULT_I2C_SCL_PIN 22
#define DEFAULT_I2C_FREQ_HZ 400000

// Target IMU Selection (MPU6050 by default)
#define CONFIG_IMU_TYPE_MPU6050 1
// #define CONFIG_IMU_TYPE_ADXL345 1

#if defined(CONFIG_IMU_TYPE_MPU6050)
    #define IMU_I2C_ADDR 0x68
#elif defined(CONFIG_IMU_TYPE_ADXL345)
    #define IMU_I2C_ADDR 0x53
#else
    #define IMU_I2C_ADDR 0x68
#endif

/**
 * @brief Initialize I2C master bus and wake up the IMU sensor.
 * @param sda_pin GPIO pin for SDA
 * @param scl_pin GPIO pin for SCL
 * @param i2c_freq_hz Clock speed (e.g. 400000 for 400kHz Fast Mode)
 * @return ESP_OK on success, esp_err_t failure code otherwise
 */
esp_err_t imu_init(int sda_pin, int scl_pin, uint32_t i2c_freq_hz);

/**
 * @brief Read raw 16-bit accelerometer measurements (X, Y, Z).
 * @param accel_x Pointer to output raw X acceleration
 * @param accel_y Pointer to output raw Y acceleration
 * @param accel_z Pointer to output raw Z acceleration
 * @return ESP_OK on success
 */
esp_err_t imu_read_raw(int16_t *accel_x, int16_t *accel_y, int16_t *accel_z);

/**
 * @brief Read acceleration in units of g (gravity, ~9.81 m/s^2).
 * @param ax_g Pointer to output X acceleration in g
 * @param ay_g Pointer to output Y acceleration in g
 * @param az_g Pointer to output Z acceleration in g
 * @return ESP_OK on success
 */
esp_err_t imu_read_accel_g(float *ax_g, float *ay_g, float *az_g);

/**
 * @brief Read 3-axis total vector magnitude in g: sqrt(ax^2 + ay^2 + az^2).
 * @param mag_g Pointer to output magnitude in g
 * @return ESP_OK on success
 */
esp_err_t imu_read_magnitude_g(float *mag_g);

#ifdef __cplusplus
}
#endif

#endif // IMU_DRIVER_H
