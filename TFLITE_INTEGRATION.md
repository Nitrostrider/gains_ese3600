# TFLite Model Integration Guide

## Overview

Your main.cpp has been modified to use TensorFlow Lite Micro for pushup posture and phase classification. The code continuously collects IMU data in a sliding window and runs inference every 500ms.

## What's Been Done

1. **Updated platformio.ini** - Added TensorFlow Lite Micro library reference
2. **Created pushup_model_data.h/.cpp** - Placeholder for your trained model
3. **Copied IMU provider** - IMU reading code from magic_wand
4. **Modified main.cpp** - Full TFLite inference pipeline

## Next Steps

### 1. Train Your Model

Run `pushup_model_colab.ipynb` in Google Colab:
- Upload your pushup training data (multiple JSON files)
- Train the multi-task CNN-LSTM model
- Download these files:
  - `pushup_model_quantized.cc` (the model)
  - `model_metadata.json` (label classes and normalization params)

### 2. Update Model Data

Open `src/pushup_model_data.cpp` and replace the placeholder with your model:

```cpp
// Replace the entire file content with the contents of pushup_model_quantized.cc
// The file should contain:
unsigned char pushup_model_quantized_tflite[] = { 0x1c, 0x00, ... };
unsigned int pushup_model_quantized_tflite_len = XXXXX;
```

Then update the variable names at the bottom:

```cpp
#include "pushup_model_data.h"

// PASTE THE ENTIRE CONTENTS OF pushup_model_quantized.cc HERE

// Then alias to our expected names:
const unsigned char g_pushup_model_data[] = pushup_model_quantized_tflite[];
const int g_pushup_model_data_len = pushup_model_quantized_tflite_len;
```

### 3. Update Normalization Parameters

Open `model_metadata.json` from Colab and copy the normalization values.

In `src/main.cpp` (lines 47-48), update:

```cpp
// Replace these with values from model_metadata.json
float imu_mean[NUM_CHANNELS] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
float imu_std[NUM_CHANNELS] = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
```

With your actual values from `norm_params.json` generated during training, for example:

```cpp
float imu_mean[NUM_CHANNELS] = {-0.123, 0.456, 9.81, 2.34, -1.23, 0.56};
float imu_std[NUM_CHANNELS] = {1.23, 1.45, 2.34, 12.3, 15.6, 11.2};
```

### 4. Verify Label Order

Check `model_metadata.json` for the exact order of classes.

In `src/main.cpp`, update the label arrays (lines 35-42) to match:

```cpp
// Phase labels - ORDER MUST MATCH YOUR TRAINING DATA
const char* phase_labels[NUM_PHASE_CLASSES] = {
    // "bottom", 
    "moving-down", 
    "moving-up", 
    "not-in-pushup", 
    "top"
};

// Posture labels - ORDER MUST MATCH YOUR TRAINING DATA
const char* posture_labels[NUM_POSTURE_CLASSES] = {
    "good-form", 
    // "hips-high", 
    "hips-sagging", 
    // "partial-rom"
};
```

### 5. Build and Upload

```bash
pio run -t upload && pio device monitor
```

## How It Works

### Data Collection Flow

1. **IMU Reading** - Reads accelerometer + gyroscope at ~100 Hz
2. **Circular Buffer** - Stores last 100 samples (1 second window)
3. **Normalization** - Applies mean/std normalization per channel
4. **Quantization** - Converts float32 → int8 for model input
5. **Inference** - Runs every 500ms on the sliding window
6. **Output** - Prints phase/posture predictions to serial + OLED

### Serial Monitor Output

```
========== PREDICTION ==========
Phase: moving-down (87.3%)
Posture: good-form (92.1%)

All Phase Probabilities:
  moving-down: 87.3%
  top: 8.2%

All Posture Probabilities:
  good-form: 92.1%
  hips-sagging: 5.3%
================================
```

### OLED Display

```
GAINS
Ph: moving-down
    87%
Po: good-form
    92%
```

## Model Configuration

The code is currently configured for:
- **Window size**: 100 samples (~1 second at 100 Hz)
- **Channels**: 6 (ax, ay, az, gx, gy, gz)
- **Phase classes**: 5 (top, moving-down, bottom, moving-up, not-in-pushup)
- **Posture classes**: 4 (good-form, hips-sagging, hips-high, partial-rom)
- **Inference rate**: 500ms (2 Hz)
- **Tensor arena**: 120 KB

If your model has different parameters, update lines 29-32 in `main.cpp`.

## Troubleshooting

### "Model version mismatch"
- Your TFLite model was created with a different TensorFlow version
- Try retraining with the latest TensorFlow in Colab

### "Tensor allocation failed"
- Model is too large for 120 KB arena
- Increase `kTensorArenaSize` on line 58
- Or simplify your model architecture

### "Inference failed"
- Check that input shape matches model expectations
- Verify quantization parameters are correct
- Monitor serial output for detailed error messages

### Wrong predictions
- Verify normalization parameters match training data
- Check label order matches model output
- Ensure WINDOW_SIZE matches training configuration

## Memory Usage

- **Tensor Arena**: 120 KB
- **IMU Buffer**: 100 × 6 × 4 = 2.4 KB
- **Model Data**: ~50-200 KB (depends on your model)
- **Total**: ~170-320 KB

The XIAO ESP32S3 has 512 KB SRAM, so this should fit comfortably.

## Next Enhancements

1. **Add BLE streaming** - Send predictions to phone app
2. **Rep counting** - Detect complete pushup cycles
3. **Real-time feedback** - Audio/haptic alerts for bad form
4. **Data logging** - Save predictions to SD card
5. **Battery optimization** - Reduce inference rate or use wake-on-motion
