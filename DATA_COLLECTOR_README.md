# Pushup IMU Data Collector

A Python GUI application for collecting IMU (accelerometer + gyroscope) data from the XIAO ESP32S3 device for pushup posture and phase classification using TensorFlow.

## Overview

This data collector is designed to gather training data for machine learning models that classify:
- **Pushup Phase**: top, moving-down, bottom, moving-up, not-in-pushup
- **Pushup Posture**: good-form, hips-sagging, hips-high (pike), partial-rom

The data is collected from a single IMU sensor worn on the body (recommended: upper back or sternum) and exported in JSON format for training in Jupyter notebooks.

## Features

- **BLE Connectivity**: Connects wirelessly to XIAO ESP32S3 via Bluetooth Low Energy
- **Real-time Visualization**: Live graphs showing accelerometer and gyroscope data
- **Labeled Recording**: Record sessions with phase and posture labels
- **Session Metadata**: Track participant ID, sensor placement, and notes
- **JSON Export**: Export data in a structured format for TensorFlow training
- **Multi-session Support**: Record multiple sessions before exporting

## Prerequisites

### Hardware
- Seeed Studio XIAO ESP32S3 with expansion board
- ICM20600 6-axis IMU sensor
- Firmware from `MW_DataCollection` uploaded to the device
  - Device will advertise as **"GAINS-Pushup"** over BLE

### Software
- Python 3.8 or higher
- Bluetooth adapter (built-in on most laptops)

## Installation

1. **Install Python dependencies**:
```bash
pip install -r requirements.txt
```

The requirements include:
- `bleak` - Python BLE library
- `matplotlib` - For real-time data visualization
- `numpy` - Numerical operations

2. **Ensure your ESP32 firmware is running**:
   - Upload the code from `MW_DataCollection/src/main.cpp` to your XIAO ESP32S3
   - The device should advertise as "MagicWand" over BLE

## Usage

### 1. Start the GUI

```bash
python pushup_data_collector.py
```

### 2. Connect to Device

1. Click **"Connect to Device"**
2. The app will scan for all available BLE devices (5 second scan)
3. A dialog will appear showing all discovered devices with their names and addresses
4. **Select your device** from the list (look for **"GAINS-Pushup"**)
5. Click **"Connect"** or double-click the device
6. Wait for the status to show "🟢 Connected"
7. You should see live IMU data in the graphs

**Tip**: If you don't see your device, make sure it's powered on and advertising, then try scanning again.

### 3. Configure Session Metadata

Before recording, fill in:
- **Participant ID**: Unique identifier (e.g., "P001", "participant_1")
- **IMU Placement**: Where the sensor is attached (e.g., "Upper Back", "Sternum")
- **Notes**: Any additional information about the session

### 4. Record Training Data

For each recording session:

1. **Select Phase Label**:
   - `top` - At the top position of pushup
   - `moving-down` - Descending phase
   - `bottom` - At the bottom position
   - `moving-up` - Ascending phase
   - `not-in-pushup` - Resting or other movements

2. **Select Posture Label**:
   - `good-form` - Proper pushup technique
   - `hips-sagging` - Hips dropping toward ground
   - `hips-high` - Pike position
   - `partial-rom` - Incomplete range of motion

3. **Click "▶ Start Recording"**
   - Perform the pushup or movement
   - The status will show "⏺ Recording" in red
   - Live graphs will display the IMU data

4. **Click "⏹ Stop Recording"**
   - Session is automatically saved
   - Summary shows sample count and duration

5. **Repeat** for different phases and postures

### 5. Export Data

1. Click **"📥 Export to JSON"**
2. Choose a save location
3. The data is saved in JSON format with all sessions

## Data Collection Protocol

### Recommended Collection Strategy

Based on `SUMMARY.MD`, collect data for each participant:

1. **Good Form Pushups**:
   - Record multiple sessions covering all phases
   - Phase: `top`, `moving-down`, `bottom`, `moving-up`
   - Posture: `good-form`
   - ~10-15 repetitions

2. **Hips Sagging**:
   - Record intentionally incorrect form
   - Phase: All phases
   - Posture: `hips-sagging`
   - ~10 repetitions

3. **Hips High (Pike)**:
   - Record pike position pushups
   - Phase: All phases
   - Posture: `hips-high`
   - ~10 repetitions

4. **Partial ROM**:
   - Record incomplete pushups
   - Phase: Mainly `moving-down` and `top`
   - Posture: `partial-rom`
   - ~10 repetitions

5. **Not In Pushup**:
   - Record standing, sitting, other movements
   - Phase: `not-in-pushup`
   - Posture: Any
   - Various movements

### Tips for Good Data

- **Consistent Placement**: Keep IMU in same position across sessions
- **Secure Attachment**: Use elastic strap, tape, or compression shirt pocket
- **Document Orientation**: Note which way the sensor is facing
- **Multiple Participants**: Collect from different people for generalization
- **Record Video**: Consider recording synchronized video for verification
- **Session Length**: 5-10 seconds per session is usually sufficient
- **Multiple Angles**: If possible, collect data with different body positions

## Output Format

The exported JSON file contains:

```json
{
  "metadata": {
    "export_timestamp": "2025-12-01T12:34:56",
    "total_sessions": 25,
    "sample_rate_hz": 40,
    "format_version": "1.0"
  },
  "sessions": [
    {
      "session_id": 0,
      "timestamp": "2025-12-01T12:30:00",
      "participant_id": "P001",
      "imu_placement": "Upper Back",
      "notes": "First session, good form",
      "phase_label": "moving-down",
      "posture_label": "good-form",
      "sample_count": 120,
      "duration_sec": 3.0,
      "sample_rate_hz": 40,
      "data": [
        {
          "timestamp": "2025-12-01T12:30:00.000",
          "elapsed_sec": 0.0,
          "ax": 0.98, "ay": 0.01, "az": 0.15,
          "gx": 2.3, "gy": -1.2, "gz": 0.5
        },
        ...
      ]
    },
    ...
  ]
}
```

### Data Fields

Each sample contains:
- **timestamp**: ISO format timestamp
- **elapsed_sec**: Time since recording started
- **ax, ay, az**: Accelerometer values in g (±2g range)
- **gx, gy, gz**: Gyroscope values in °/s (±250°/s range)

## Using the Data in TensorFlow

### Loading the Data

```python
import json
import numpy as np

# Load the exported data
with open('pushup_data_20251201_123456.json', 'r') as f:
    data = json.load(f)

# Extract all sessions
sessions = data['sessions']

# Organize by labels
for session in sessions:
    phase = session['phase_label']
    posture = session['posture_label']
    samples = session['data']

    # Convert to numpy array
    imu_data = np.array([[s['ax'], s['ay'], s['az'],
                          s['gx'], s['gy'], s['gz']]
                         for s in samples])

    # Shape: (num_samples, 6)
    print(f"Session {session['session_id']}: {imu_data.shape}")
```

### Creating Windows for Training

```python
def create_windows(imu_data, window_size=50, stride=10):
    """Create sliding windows from IMU data"""
    windows = []
    for i in range(0, len(imu_data) - window_size + 1, stride):
        window = imu_data[i:i+window_size]
        windows.append(window)
    return np.array(windows)

# Example: 50 samples (1.25s @ 40Hz), stride of 10 (0.25s)
windows = create_windows(imu_data, window_size=50, stride=10)
# Shape: (num_windows, 50, 6)
```

### Preprocessing

```python
# Normalize per channel
mean = imu_data.mean(axis=0)
std = imu_data.std(axis=0)
normalized = (imu_data - mean) / std

# Or normalize per session
for session in sessions:
    samples = np.array([[s['ax'], s['ay'], s['az'],
                        s['gx'], s['gy'], s['gz']]
                       for s in session['data']])

    # Normalize
    mean = samples.mean(axis=0)
    std = samples.std(axis=0)
    normalized = (samples - mean) / std
```

## Troubleshooting

### Cannot Connect to Device / No Devices Found

- Ensure the ESP32 is powered on and running the firmware
- Check that Bluetooth is enabled on your computer
- On Linux, you may need to run with `sudo` or configure Bluetooth permissions
- Try moving closer to the device
- Restart the ESP32 by unplugging and replugging power
- Click "Connect to Device" again to rescan for devices
- Make sure your device is advertising (check serial monitor for "Advertising" message)
- If you see your device in the list but connection fails, try selecting it again

### No Data Appearing in Graphs

- Check the BLE connection status
- Verify the firmware is sending data (check serial monitor at 115200 baud)
- Ensure the IMU is properly initialized (check serial output)

### Graphs Not Updating

- This is normal - graphs update every 250ms to reduce CPU usage
- If completely frozen, try disconnecting and reconnecting

### Recording Shows 0 Samples

- Ensure you clicked "Start Recording" before performing the movement
- Check that the device is connected
- Verify the firmware's control characteristic is working (send 0x01 to start)

### Export File Too Large

- Each session at 40Hz generates ~40 samples/second
- Consider recording shorter sessions (5-10 seconds)
- The JSON format is human-readable but not compressed
- For very large datasets, consider splitting exports

## Technical Details

### BLE UUIDs

- Service UUID: `0000ff00-0000-1000-8000-00805f9b34fb`
- IMU Characteristic: `0000ff01-0000-1000-8000-00805f9b34fb` (Notify)
- Control Characteristic: `0000ff02-0000-1000-8000-00805f9b34fb` (Write)

### Data Format

- IMU sends 9 float32 values (36 bytes): ax, ay, az, gx, gy, gz, mx, my, mz
- GUI uses first 6 values (24 bytes), ignoring magnetometer
- Sample rate: ~40 Hz (25ms period)
- Little-endian byte order

### Coordinate System

- **Accelerometer**: ±2g range, 16384 LSB/g
- **Gyroscope**: ±250°/s range, 131 LSB/°/s
- Coordinate frame depends on physical IMU orientation

## License

This project is part of the ESE 3600 GAINS pushup classification system.

## Support

For issues or questions:
1. Check the `CLAUDE.md` file for hardware setup
2. Check the `SUMMARY.md` file for data collection protocol
3. Review the ESP32 serial output for debugging
4. Verify BLE connectivity using a generic BLE scanner app

## Next Steps

After collecting data:

1. **Preprocess** the data (normalization, filtering)
2. **Create windows** using sliding window approach (1-2 second windows)
3. **Split dataset** by participant (avoid overfitting)
4. **Train model** using CNN, LSTM, or transformer architecture
5. **Evaluate** using confusion matrices and per-class metrics

See `SUMMARY.md` for detailed ML pipeline recommendations.
