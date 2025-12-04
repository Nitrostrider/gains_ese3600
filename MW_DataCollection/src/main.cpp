#include <Arduino.h>
#include <Wire.h>

// ---------- I2C pins & ICM20600 config ----------
#define I2C_SDA 5     // working pins for your Xiao ESP32S3
#define I2C_SCL 6
#define ICM20600_ADDR 0x69

#define REG_PWR_MGMT_1   0x6B
#define REG_ACCEL_CONFIG 0x1C
#define REG_GYRO_CONFIG  0x1B
#define REG_ACCEL_XOUT_H 0x3B
#define REG_WHO_AM_I     0x75

// conversions for ±2g accel, ±250 dps gyro
static const float ACC_LSB_PER_G   = 16384.0f; // ±2g
static const float GYR_LSB_PER_DPS = 131.0f;   // ±250 dps

// ---------- Globals ----------
volatile bool streamOn = false;        // toggled by serial commands
const uint32_t PERIOD_MS = 25;         // ~40 Hz

// ---------- I2C helpers ----------
static bool i2cWrite(uint8_t reg, uint8_t val) {
  Wire.beginTransmission((uint8_t)ICM20600_ADDR);
  Wire.write(reg);
  Wire.write(val);
  return Wire.endTransmission() == 0;
}

static bool i2cReadBytes(uint8_t reg, uint8_t* buf, size_t len) {
  Wire.beginTransmission((uint8_t)ICM20600_ADDR);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;

  // force the uint8_t,uint8_t overload to avoid ambiguity
  if (Wire.requestFrom((uint8_t)ICM20600_ADDR, (uint8_t)len) != (int)len) return false;
  for (size_t i = 0; i < len; ++i) {
    buf[i] = Wire.read();
  }
  return true;
}

static bool icmInit(uint8_t& who) {
  who = 0;
  if (!i2cWrite(REG_PWR_MGMT_1, 0x01)) return false;  // wake, PLL
  delay(50);
  if (!i2cWrite(REG_ACCEL_CONFIG, 0x00)) return false; // ±2g
  if (!i2cWrite(REG_GYRO_CONFIG,  0x00)) return false; // ±250 dps
  delay(10);
  return i2cReadBytes(REG_WHO_AM_I, &who, 1);
}

static bool icmRead(float& ax,float& ay,float& az,
                    float& gx,float& gy,float& gz) {
  uint8_t raw[14];
  if (!i2cReadBytes(REG_ACCEL_XOUT_H, raw, sizeof(raw))) return false;

  auto s16 = [&](int i)->int16_t {
    return (int16_t)((raw[i] << 8) | raw[i+1]);
  };

  ax = s16(0)  / ACC_LSB_PER_G;
  ay = s16(2)  / ACC_LSB_PER_G;
  az = s16(4)  / ACC_LSB_PER_G;

  gx = s16(8)  / GYR_LSB_PER_DPS;
  gy = s16(10) / GYR_LSB_PER_DPS;
  gz = s16(12) / GYR_LSB_PER_DPS;

  return true;
}

void setup() {
  Serial.begin(115200);
  delay(1000);
  Serial.println("\n[BOOT] GAINS Pushup IMU Data Collector (Serial Mode)");
  Serial.println("Commands: START (begin streaming), STOP (end streaming)");
  Serial.println("Waiting for commands...\n");

  // I2C + IMU
  Wire.begin(I2C_SDA, I2C_SCL, 400000);
  uint8_t who = 0;
  if (icmInit(who)) {
    Serial.printf("[I2C] ICM20600 OK, WHO_AM_I=0x%02X (addr 0x%02X)\n",
                  who, ICM20600_ADDR);
  } else {
    Serial.println("[I2C] ICM init FAILED");
  }

  Serial.println("[READY] Send START to begin streaming IMU data");
}

void loop() {
  static uint32_t last = 0;

  // Check for serial commands
  if (Serial.available() > 0) {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();

    if (cmd == "START") {
      streamOn = true;
      Serial.println("[CMD] Streaming STARTED");
    } else if (cmd == "STOP") {
      streamOn = false;
      Serial.println("[CMD] Streaming STOPPED");
    }
  }

  if (!streamOn) {
    delay(5);
    return;
  }

  uint32_t now = millis();
  if (now - last < PERIOD_MS) {
    delay(1);
    return;
  }
  last = now;

  float ax, ay, az, gx, gy, gz;
  if (!icmRead(ax, ay, az, gx, gy, gz)) return;

  // Send data as binary: 6 floats (24 bytes)
  // Format: ax, ay, az, gx, gy, gz (all little-endian float32)
  Serial.write((uint8_t*)&ax, sizeof(float));
  Serial.write((uint8_t*)&ay, sizeof(float));
  Serial.write((uint8_t*)&az, sizeof(float));
  Serial.write((uint8_t*)&gx, sizeof(float));
  Serial.write((uint8_t*)&gy, sizeof(float));
  Serial.write((uint8_t*)&gz, sizeof(float));
}
