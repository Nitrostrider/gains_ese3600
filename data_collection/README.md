# GAINS-Pushup BLE Firmware

This firmware turns your XIAO ESP32S3 into a wireless IMU data streamer for the GAINS pushup classification system.

## Device Name

When flashed, your device will advertise as: **`GAINS-Pushup`**

This makes it easy to identify in the Python data collector GUI's device selection dialog.

## Hardware

- **Board**: Seeed Studio XIAO ESP32S3
- **IMU**: ICM20600 (6-axis: accelerometer + gyroscope)
- **I2C Pins**: SDA=GPIO5, SCL=GPIO6
- **I2C Address**: 0x69

## BLE Configuration

- **Device Name**: `GAINS-Pushup`
- **Service UUID**: `0xFF00`
- **IMU Characteristic**: `0xFF01` (Notify) - Sends 9 float32 values (36 bytes)
  - ax, ay, az (accelerometer in g)
  - gx, gy, gz (gyroscope in °/s)
  - mx, my, mz (magnetometer - set to 0)
- **Control Characteristic**: `0xFF02` (Write)
  - `0x01` - Start streaming
  - `0x00` - Stop streaming

## Flashing

From the `MW_DataCollection` directory:

```bash
# Build and upload
pio run -t upload

# Monitor serial output (115200 baud)
pio device monitor

# Upload and monitor
pio run -t upload && pio device monitor
```

## Serial Output

Expected output when powered on:

```
[BOOT] GAINS Pushup IMU Data Collector
[I2C] ICM20600 OK, WHO_AM_I=0x11 (addr 0x69)
[BLE] Advertising as GAINS-Pushup with svc 0xFF00
```

## Usage

1. Flash this firmware to your XIAO ESP32S3
2. Power on the device
3. Run the Python data collector GUI (`pushup_data_collector.py`)
4. Look for **"GAINS-Pushup"** in the device selection dialog
5. Connect and start collecting data

## Data Format

IMU data is sent at ~40Hz (25ms intervals):

- **Accelerometer Range**: ±2g (16384 LSB/g)
- **Gyroscope Range**: ±250°/s (131 LSB/°/s)
- **Byte Order**: Little-endian
- **Data Type**: float32

Each packet contains 36 bytes (9 floats), but only the first 6 values (24 bytes) are used by the data collector.

## Troubleshooting

### Device Not Appearing in BLE Scan

- Check serial monitor for "Advertising as GAINS-Pushup"
- Verify IMU initialization succeeded (WHO_AM_I check)
- Ensure Bluetooth is enabled on your computer
- Try restarting the device (unplug/replug power)

### IMU Initialization Failed

- Check I2C connections (SDA=GPIO5, SCL=GPIO6)
- Verify ICM20600 is at address 0x69
- Check power supply to expansion board
- Ensure I2C pins are not being used by other peripherals

### No Data Streaming

- Check if control characteristic is being written to
- Verify BLE connection is established
- Check serial monitor for error messages
- Ensure MTU is at least 128 bytes

## Modifications

To change the device name, edit `src/main.cpp`:

```cpp
// Line 120, 141, 143, 145
NimBLEDevice::init("YOUR-NAME-HERE");
scanResp.setName("YOUR-NAME-HERE");
adv->setName("YOUR-NAME-HERE");
```

Rebuild and upload after changes.
