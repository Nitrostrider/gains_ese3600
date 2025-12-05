# Model Training Guide for ESP32 Deployment

## Overview

This guide explains how to train a posture classification model for push-up form analysis using the `pushup_model_colab.ipynb` notebook.

**What the model does:**
- Classifies push-up posture into 4 categories: `good-form`, `hips-sagging`, `hips-high`, `partial-rom`
- Uses IMU sensor data (accelerometer + gyroscope) from a single sensor placed on the sternum
- Designed to run on ESP32 microcontrollers using TensorFlow Lite Micro

**Important Notes:**
- ⚠️ **Posture Classification Only**: This model focuses solely on classifying push-up form/posture. Phase detection (top, bottom, moving-up, moving-down) has been removed to simplify the model.
- ⚠️ **No LSTM for ESP32**: The default notebook uses LSTM which is NOT supported on ESP32. You must use a CNN-only architecture for deployment.

## Current Status

✅ **Hardware Working:**
- OLED display: Working
- Button/LED: Working
- IMU (ICM-20600): Working perfectly!
- I2C bus: Shared successfully

❌ **Model Issue:**
- The default CNN-LSTM model uses LSTM layers
- LSTM requires `SELECT_TF_OPS` which is **NOT supported** on TFLite Micro (ESP32)
- You need to use the CNN-only model instead

## Data Processing: Sliding Windows

**What is a sliding window?**

Your training data consists of sessions with variable lengths (e.g., 50-100 samples each). However, neural networks require fixed-size inputs. The sliding window technique solves this:

1. **Window Size (50 samples):** Each training example is 50 consecutive IMU readings
   - At 40 Hz sample rate, this is 1.25 seconds of data
   - Shape: (50, 6) - 50 timesteps × 6 features (ax, ay, az, gx, gy, gz)

2. **Stride (10 samples):** How far to slide the window for the next example
   - A stride of 10 means each window overlaps by 40 samples with the previous one
   - This creates more training data from limited sessions (data augmentation)
   - At 40 Hz, stride of 10 = 0.25 second shift

**Example:**
```
Session with 80 samples:
  Window 1: samples [0-49]   → Label: "hips-sagging"
  Window 2: samples [10-59]  → Label: "hips-sagging"  (overlaps with Window 1)
  Window 3: samples [20-69]  → Label: "hips-sagging"
  Window 4: samples [30-79]  → Label: "hips-sagging"

Result: 1 session (80 samples) → 4 training windows
```

This approach:
- ✅ Provides consistent input size for the neural network
- ✅ Increases training data (more windows than sessions)
- ✅ Captures temporal patterns in the IMU data
- ✅ Helps the model learn from different parts of the movement

## The Problem

TensorFlow Lite Micro (used on embedded devices like ESP32) only supports a subset of TensorFlow operations:

**Supported:** Conv2D, Conv1D, Dense, MaxPooling, GlobalAveragePooling, ReLU, etc.
**NOT Supported:** LSTM, GRU, native TensorFlow ops

The original notebook uses LSTM which won't work on ESP32.

## The Solution

The `pushup_model_colab.ipynb` notebook is already configured with an **ESP32-compatible CNN model** that uses only TFLite-supported operations.

### Step-by-Step Training:

1. **Open `pushup_model_colab.ipynb` in Google Colab**
   - File → Open notebook → Upload the .ipynb file

2. **Enable GPU acceleration (recommended)**
   - Runtime → Change runtime type → Select "T4 GPU"

3. **Section 1: Setup and Imports**
   - Run to check TensorFlow version and GPU availability

4. **Section 2: Upload Data**
   - Upload your `pushup_data_*.json` files from the data collector
   - Files will be saved to `dataset/` folder

5. **Section 3-4: Load and Explore Data**
   - Loads all JSON files
   - Shows label distribution and sample session visualization
   - Expected: 277 sessions with 4 posture types

6. **Section 5: Create Sliding Windows**
   - **Window Size**: 50 samples (1.25 seconds @ 40Hz)
   - **Stride**: 10 samples (0.25 seconds overlap)
   - Converts variable-length sessions into fixed-size windows
   - Expected output: ~1000+ windows from 277 sessions

7. **Section 6: Prepare Training Data**
   - Normalizes IMU data (mean=0, std=1)
   - Encodes posture labels (4 classes)
   - Splits into train/validation/test (70%/15%/15%)

8. **Section 7: Build Model**
   - **Model Architecture**: CNN with GlobalAveragePooling1D
   - Uses only TFLite-compatible operations (no LSTM!)
   - Input shape: (50, 6) - 50 timesteps × 6 IMU channels
   - Output: 4 posture classes

9. **Section 8: Train the Model**
   - Trains for up to 50 epochs with early stopping
   - Uses learning rate reduction on plateau
   - Expected: 99-100% validation accuracy

10. **Section 9-10: Evaluate and Save**
    - Tests on held-out test set
    - Saves model as `pushup_model.h5`
    - Saves normalization parameters to `pushup_model_metadata.json`

11. **Section 12-13: Convert to TFLite**
    - Converts to Float32 TFLite model (169 KB)
    - Converts to INT8 quantized model (58 KB) - **Use this for ESP32!**
    - Tests accuracy of both models (should maintain 99-100%)

12. **Section 14: Convert to C Array**
    - Converts `.tflite` file to `.cc` file using `xxd`
    - Output: `pushup_model.cc` with byte array

13. **Section 15: Download Files**
    - Downloads all necessary files to your computer

### Key Files to Download:

- ✅ `pushup_model.cc` - C array for ESP32 (from Section 14)
- ✅ `pushup_model_metadata.json` - Normalization parameters
- ✅ `pushup_model_quantized.tflite` - Quantized model for testing

## Model Architecture (Current Notebook)

### ESP32-Compatible CNN Model ✅

The current notebook uses a **CNN-only architecture** that is fully compatible with TFLite Micro:

```
Input: (50, 6)
  50 timesteps × 6 IMU channels (ax, ay, az, gx, gy, gz)
  ↓
Conv1D(32, kernel=5) → BatchNorm → MaxPool(2)
  Features: 50 → 25 timesteps, 32 channels
  ↓
Conv1D(64, kernel=5) → BatchNorm → MaxPool(2)
  Features: 25 → 12 timesteps, 64 channels
  ↓
Conv1D(128, kernel=3) → BatchNorm
  Features: 12 timesteps, 128 channels
  ↓
GlobalAveragePooling1D
  Aggregates temporal features → 128 features
  ↓
Dropout(0.3)
  ↓
Dense(32, ReLU)
  ↓
Dense(4, Softmax)
  ↓
Output: 4 posture classes
  [good-form, hips-high, hips-sagging, partial-rom]
```

**Model Properties:**
- ✅ **Total Parameters**: 41,156 (161 KB unquantized)
- ✅ **Quantized Size**: 58.37 KB (89.3% compression)
- ✅ **All layers supported** by TFLite Micro
- ✅ **No LSTM/GRU layers** (these won't work on ESP32)
- ✅ **Expected Accuracy**: 99-100% on test set

**Key Features:**
- **GlobalAveragePooling1D**: Aggregates temporal features without LSTM
  - Takes 12 timesteps × 128 channels → outputs 128 features
  - Much more memory-efficient than LSTM on embedded devices
- **BatchNormalization**: Stabilizes training, improves convergence
  - Supported by TFLite Micro in inference mode
- **Posture-only classification**: Simplified from multi-task to focus on form detection

## Expected Performance

Based on the current dataset and model architecture:

**Training Results:**
- ✅ **Training Accuracy**: 99-100%
- ✅ **Validation Accuracy**: 99-100%
- ✅ **Test Accuracy**: 100% (on quantized INT8 model)
- ✅ **No accuracy loss** from quantization

**Model Size:**
- Original Keras model: 547 KB
- Float32 TFLite: 170 KB (69% reduction)
- INT8 Quantized: **58 KB** (89% reduction) ← **Use this for ESP32**

**Why it works well:**
- The current architecture is well-suited for the task
- 277 sessions provide sufficient training data
- Posture patterns are distinguishable in IMU data
- GlobalAveragePooling effectively captures motion patterns

## After Training - Deploy to ESP32

### 1. Download Model Files

From Google Colab (Section 14-15):
- ✅ `pushup_model.cc` - C byte array for ESP32
- ✅ `pushup_model_metadata.json` - Normalization parameters

### 2. Update ESP32 Firmware

**Step 2a: Replace Model Data**

Open `src/pushup_model_data.cpp` and replace the entire contents with the contents of `pushup_model.cc`:

```cpp
// File: src/pushup_model_data.cpp
// Copy entire contents from pushup_model.cc

unsigned char pushup_model_quantized_tflite[] = {
  0x1c, 0x00, 0x00, 0x00, 0x54, 0x46, 0x4c, 0x33, ...
  // (thousands of bytes)
};
unsigned int pushup_model_quantized_tflite_len = 59768;

// Then add these lines at the end:
const unsigned char* g_pushup_model_data = pushup_model_quantized_tflite;
const int g_pushup_model_data_len = pushup_model_quantized_tflite_len;
```

**Step 2b: Update Normalization Parameters**

Open `pushup_model_metadata.json` and copy the `mean` and `std` arrays.

Then update `src/main.cpp` (around lines 54-65):

```cpp
// From pushup_model_metadata.json:
// "mean": [0.21002982, -0.0125976, 0.96834566, 0.1763402, 1.13920085, 0.94968138]
// "std": [0.27886892, 0.11287905, 0.28730744, 13.64203132, 36.71071629, 6.24481333]

float imu_mean[NUM_CHANNELS] = {
    0.21002982,   // ax
    -0.0125976,   // ay
    0.96834566,   // az
    0.1763402,    // gx
    1.13920085,   // gy
    0.94968138    // gz
};

float imu_std[NUM_CHANNELS] = {
    0.27886892,   // ax
    0.11287905,   // ay
    0.28730744,   // az
    13.64203132,  // gx
    36.71071629,  // gy
    6.24481333    // gz
};
```

**Step 2c: Update Class Labels (if needed)**

Verify the posture class order matches your model:

```cpp
// In src/main.cpp
const char* posture_labels[] = {
    "good-form",    // Class 0
    "hips-high",    // Class 1
    "hips-sagging", // Class 2
    "partial-rom"   // Class 3
};
```

### 3. Build and Upload

```bash
cd /path/to/gains_ese3600
pio run -t upload
```

### 4. Test the Model

1. Open serial monitor: `pio device monitor`
2. Press the button to start inference
3. Perform a push-up with different forms
4. Observe the predictions on the OLED display

## Troubleshooting

### Training Issues

**"No JSON files found in dataset/ folder":**
- Make sure you uploaded your data files in Section 2
- Files should be named `pushup_data_*.json`

**Low accuracy or model not converging:**
- Check that you have balanced data across all 4 posture types
- Try training for more epochs (increase from 50)
- Verify your labeling is consistent

**"Out of memory" during training:**
- Reduce batch size from 32 to 16
- Use CPU instead of GPU (though slower)

### Deployment Issues

**"Failed to get registration from op code CUSTOM" on ESP32:**
- The model contains unsupported operations
- Verify you're using the CNN model (Section 7), not LSTM
- Check that conversion to INT8 succeeded in Section 13

**"Tensor allocation failed" on ESP32:**
- Model too large for tensor arena (currently 120 KB)
- Increase `kTensorArenaSize` in `src/main.cpp` (around line 58)
- Try reducing model complexity (fewer filters/layers)

**Predictions are random/incorrect:**
- ❌ **Most common issue**: Normalization parameters don't match
  - Verify `mean` and `std` in `src/main.cpp` exactly match `pushup_model_metadata.json`
  - Check for copy-paste errors or missing decimal places
- ❌ **Class order mismatch**: Check posture label order matches model output
  - Model outputs: `[good-form, hips-high, hips-sagging, partial-rom]`
- ❌ **Wrong model uploaded**: Ensure you're using `pushup_model.cc` (INT8 quantized)

**OLED shows "No IMU" or "IMU init failed":**
- Hardware issue, not model-related
- Check I2C connections to ICM-20600
- Verify I2C address (0x68 or 0x69)

## Verification Checklist

After training, verify everything before deployment:

**In Google Colab:**
- ✅ Section 13: Both Float32 and INT8 models tested
- ✅ Test accuracy is 99-100%
- ✅ No accuracy loss from quantization
- ✅ `pushup_model.cc` file generated successfully
- ✅ Model size is ~58 KB

**In ESP32 Code:**
- ✅ `pushup_model_data.cpp` contains the new model
- ✅ Normalization `mean` and `std` arrays updated in `src/main.cpp`
- ✅ Class labels match: `["good-form", "hips-high", "hips-sagging", "partial-rom"]`
- ✅ Tensor arena size is adequate (120 KB recommended)

## Quick Compatibility Test

Want to verify your model is ESP32-compatible before deploying?

Check the Section 13 output in Colab. If you see:
```
✓ Quantized INT8 TFLite model saved to pushup_model_quantized.tflite
  Size: 59,768 bytes (58.37 KB)
```

And the accuracy test shows:
```
Quantized INT8 Results:
  Posture Accuracy: 1.0000 (159/159)
  Quantized: True
```

Then your model is **fully compatible** with ESP32! ✅

## Next Steps

1. ✅ Train the model using `pushup_model_colab.ipynb`
2. ✅ Download `pushup_model.cc` and `pushup_model_metadata.json`
3. ✅ Update `src/pushup_model_data.cpp` with model data
4. ✅ Update normalization parameters in `src/main.cpp`
5. ✅ Build and upload to ESP32
6. ✅ Test with real push-ups and observe predictions!
