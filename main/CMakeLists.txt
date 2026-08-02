idf_component_register(SRCS "main.c" "imu_driver.c" "tiny_cl.c"
                       INCLUDE_DIRS "."
                       PRIV_REQUIRES driver esp_driver_uart esp_driver_i2c esp_timer)


