// I2C Scanner utility - temporarily add to main.cpp setup() to debug

#include <Arduino.h>
#include "driver/i2c.h"
#include "esp_log.h"

void scan_i2c_bus() {
    Serial.println("\n=== I2C Bus Scanner ===");

    i2c_port_t i2c_port = I2C_NUM_0;

    int devices_found = 0;

    for (uint8_t addr = 1; addr < 127; addr++) {
        i2c_cmd_handle_t cmd = i2c_cmd_link_create();
        i2c_master_start(cmd);
        i2c_master_write_byte(cmd, (addr << 1) | I2C_MASTER_WRITE, true);
        i2c_master_stop(cmd);

        esp_err_t ret = i2c_master_cmd_begin(i2c_port, cmd, pdMS_TO_TICKS(50));
        i2c_cmd_link_delete(cmd);

        if (ret == ESP_OK) {
            Serial.printf("Device found at address 0x%02X\n", addr);
            devices_found++;
        }
    }

    if (devices_found == 0) {
        Serial.println("No I2C devices found!");
    } else {
        Serial.printf("\nTotal devices found: %d\n", devices_found);
    }

    Serial.println("======================\n");
}
