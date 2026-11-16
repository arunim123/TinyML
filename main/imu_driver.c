#include "imu_driver.h"
#include <math.h>
#include "esp_log.h"
#include "driver/i2c.h"

static const char *TAG = "IMU_DRIVER";

#define I2C_MASTER_NUM I2C_NUM_0
#define ACK_CHECK_EN 0x1
#define ACK_CHECK_DIS 0x0
#define ACK_VAL 0x0
#define NACK_VAL 0x1

// MPU6050 Register map
#define MPU6050_RA_PWR_MGMT_1   0x6B
#define MPU6050_RA_ACCEL_XOUT_H 0x3B
#define MPU6050_LSB_PER_G       16384.0f

// ADXL345 Register map
#define ADXL345_RA_POWER_CTL    0x2D
#define ADXL345_RA_DATAX0       0x32
#define ADXL345_LSB_PER_G       256.0f

static bool s_initialized = false;

static esp_err_t i2c_write_byte(uint8_t reg_addr, uint8_t data)
{
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (IMU_I2C_ADDR << 1) | I2C_MASTER_WRITE, ACK_CHECK_EN);
    i2c_master_write_byte(cmd, reg_addr, ACK_CHECK_EN);
    i2c_master_write_byte(cmd, data, ACK_CHECK_EN);
    i2c_master_stop(cmd);
    esp_err_t ret = i2c_master_cmd_begin(I2C_MASTER_NUM, cmd, pdMS_TO_TICKS(1000));
    i2c_cmd_link_delete(cmd);
    return ret;
}

static esp_err_t i2c_read_bytes(uint8_t reg_addr, uint8_t *data, size_t len)
{
    if (len == 0) return ESP_OK;
    i2c_cmd_handle_t cmd = i2c_cmd_link_create();
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (IMU_I2C_ADDR << 1) | I2C_MASTER_WRITE, ACK_CHECK_EN);
    i2c_master_write_byte(cmd, reg_addr, ACK_CHECK_EN);
    i2c_master_start(cmd);
    i2c_master_write_byte(cmd, (IMU_I2C_ADDR << 1) | I2C_MASTER_READ, ACK_CHECK_EN);
    if (len > 1) {
        i2c_master_read(cmd, data, len - 1, ACK_VAL);
    }
    i2c_master_read_byte(cmd, data + len - 1, NACK_VAL);
    i2c_master_stop(cmd);
    esp_err_t ret = i2c_master_cmd_begin(I2C_MASTER_NUM, cmd, pdMS_TO_TICKS(1000));
    i2c_cmd_link_delete(cmd);
    return ret;
}

esp_err_t imu_init(int sda_pin, int scl_pin, uint32_t i2c_freq_hz)
{
    i2c_config_t conf = {
        .mode = I2C_MODE_MASTER,
        .sda_io_num = (gpio_num_t)sda_pin,
        .sda_pullup_en = GPIO_PULLUP_ENABLE,
        .scl_io_num = (gpio_num_t)scl_pin,
        .scl_pullup_en = GPIO_PULLUP_ENABLE,
        .master.clk_speed = i2c_freq_hz,
    };

    esp_err_t err = i2c_param_config(I2C_MASTER_NUM, &conf);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "I2C param config failed: %s", esp_err_to_name(err));
        return err;
    }

    err = i2c_driver_install(I2C_MASTER_NUM, conf.mode, 0, 0, 0);
    if (err != ESP_OK && err != ESP_ERR_INVALID_STATE) { // allow if already installed
        ESP_LOGE(TAG, "I2C driver install failed: %s", esp_err_to_name(err));
        return err;
    }

#if defined(CONFIG_IMU_TYPE_MPU6050)
    // Wake up MPU6050 (clear sleep bit in PWR_MGMT_1)
    err = i2c_write_byte(MPU6050_RA_PWR_MGMT_1, 0x00);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "MPU6050 initialized successfully at I2C 0x%02X (SDA:%d, SCL:%d)", IMU_I2C_ADDR, sda_pin, scl_pin);
        s_initialized = true;
    } else {
        ESP_LOGE(TAG, "Failed to wake up MPU6050 at 0x%02X (err: %s)", IMU_I2C_ADDR, esp_err_to_name(err));
    }
#elif defined(CONFIG_IMU_TYPE_ADXL345)
    // Enable measure mode in ADXL345 POWER_CTL
    err = i2c_write_byte(ADXL345_RA_POWER_CTL, 0x08);
    if (err == ESP_OK) {
        ESP_LOGI(TAG, "ADXL345 initialized successfully at I2C 0x%02X (SDA:%d, SCL:%d)", IMU_I2C_ADDR, sda_pin, scl_pin);
        s_initialized = true;
    } else {
        ESP_LOGE(TAG, "Failed to initialize ADXL345 at 0x%02X (err: %s)", IMU_I2C_ADDR, esp_err_to_name(err));
    }
#endif

    return err;
}

esp_err_t imu_read_raw(int16_t *accel_x, int16_t *accel_y, int16_t *accel_z)
{
    if (!accel_x || !accel_y || !accel_z) return ESP_ERR_INVALID_ARG;
    if (!s_initialized) return ESP_ERR_INVALID_STATE;

    uint8_t raw_buf[6] = {0};

#if defined(CONFIG_IMU_TYPE_MPU6050)
    esp_err_t err = i2c_read_bytes(MPU6050_RA_ACCEL_XOUT_H, raw_buf, 6);
    if (err == ESP_OK) {
        *accel_x = (int16_t)((raw_buf[0] << 8) | raw_buf[1]);
        *accel_y = (int16_t)((raw_buf[2] << 8) | raw_buf[3]);
        *accel_z = (int16_t)((raw_buf[4] << 8) | raw_buf[5]);
    }
    return err;
#elif defined(CONFIG_IMU_TYPE_ADXL345)
    esp_err_t err = i2c_read_bytes(ADXL345_RA_DATAX0, raw_buf, 6);
    if (err == ESP_OK) {
        *accel_x = (int16_t)((raw_buf[1] << 8) | raw_buf[0]);
        *accel_y = (int16_t)((raw_buf[3] << 8) | raw_buf[2]);
        *accel_z = (int16_t)((raw_buf[5] << 8) | raw_buf[4]);
    }
    return err;
#else
    return ESP_ERR_NOT_SUPPORTED;
#endif
}

esp_err_t imu_read_accel_g(float *ax_g, float *ay_g, float *az_g)
{
    int16_t rx, ry, rz;
    esp_err_t err = imu_read_raw(&rx, &ry, &rz);
    if (err != ESP_OK) return err;

#if defined(CONFIG_IMU_TYPE_MPU6050)
    float lsb = MPU6050_LSB_PER_G;
#elif defined(CONFIG_IMU_TYPE_ADXL345)
    float lsb = ADXL345_LSB_PER_G;
#else
    float lsb = 16384.0f;
#endif

    if (ax_g) *ax_g = (float)rx / lsb;
    if (ay_g) *ay_g = (float)ry / lsb;
    if (az_g) *az_g = (float)rz / lsb;

    return ESP_OK;
}

esp_err_t imu_read_magnitude_g(float *mag_g)
{
    float ax, ay, az;
    esp_err_t err = imu_read_accel_g(&ax, &ay, &az);
    if (err != ESP_OK) return err;

    if (mag_g) {
        *mag_g = sqrtf(ax * ax + ay * ay + az * az);
    }
    return ESP_OK;
}
