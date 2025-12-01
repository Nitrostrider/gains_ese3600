# Model Training Guide for ESP32 Deployment

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

## The Problem

TensorFlow Lite Micro (used on embedded devices like ESP32) only supports a subset of TensorFlow operations:

**Supported:** Conv2D, Conv1D, Dense, MaxPooling, GlobalAveragePooling, ReLU, etc.
**NOT Supported:** LSTM, GRU, native TensorFlow ops

The original notebook uses LSTM which won't work on ESP32.

## The Solution

Use the **TFLite Micro Compatible Model** (Section 8.1 in the updated notebook).

### Step-by-Step:

1. **Open `pushup_model_colab.ipynb` in Google Colab**

2. **Upload your training data** (Section 2)

3. **Run sections 1-7** normally (data loading, preprocessing, etc.)

4. **SKIP Section 8** (the LSTM model)

5. **Run Section 8.1** instead (CNN-only model)
   ```python
   model_micro = build_tflite_micro_compatible_model(...)
   ```

6. **In Section 9 (training), change line 4:**
   ```python
   # BEFORE (won't work on ESP32):
   # model.compile(...)

   # AFTER (works on ESP32):
   model_micro.compile(
       optimizer=keras.optimizers.Adam(learning_rate=0.001),
       loss={
           'phase': 'categorical_crossentropy',
           'posture': 'categorical_crossentropy'
       },
       metrics={
           'phase': ['accuracy'],
           'posture': ['accuracy']
       },
       loss_weights={'phase': 1.0, 'posture': 1.0}
   )

   # And change model.fit to model_micro.fit:
   history = model_micro.fit(...)
   ```

7. **Continue with remaining sections** using `model_micro` instead of `model`

8. **Download the .cc file** from Section 14

9. **Copy model to ESP32:**
   - Open `src/pushup_model_data.cpp`
   - Replace the placeholder with contents from `pushup_model_quantized.cc`
   - Make sure to update variable names

## Model Architecture Comparison

### LSTM Model (Section 8) - ❌ Won't Work on ESP32
```
Input (100, 6)
  ↓
Conv1D → MaxPool → Dropout
  ↓
Conv1D → MaxPool → Dropout
  ↓
LSTM (64 units) ← NOT SUPPORTED ON ESP32
  ↓
Dense → Dropout
  ↓
Phase Output (5 classes)
Posture Output (4 classes)
```

### CNN-Only Model (Section 8.1) - ✅ Works on ESP32
```
Input (100, 6)
  ↓
Conv1D(32) → MaxPool → Dropout
  ↓
Conv1D(64) → MaxPool → Dropout
  ↓
Conv1D(128) → MaxPool → Dropout
  ↓
GlobalAveragePooling1D ← Replaces LSTM
  ↓
Dense(128) → Dropout
  ↓
Dense(64) → Dropout
  ↓
Phase Output (5 classes)
Posture Output (4 classes)
```

The CNN-only model:
- Uses deeper CNN layers to compensate for no LSTM
- Uses GlobalAveragePooling to aggregate temporal features
- Is fully supported by TFLite Micro
- Should achieve similar accuracy (test it!)

## Expected Accuracy

The CNN-only model may be slightly less accurate than LSTM for time-series data, but:
- It will actually **run on your ESP32**
- With proper tuning (more filters, deeper network), it can match LSTM performance
- It's more memory efficient (important for embedded)

## After Training

1. **Download these files:**
   - `pushup_model_quantized.cc` (the model)
   - `model_metadata.json` (normalization params)

2. **Update `src/pushup_model_data.cpp`:**
   ```cpp
   // Replace entire file with contents of pushup_model_quantized.cc
   // Then add at the end:

   const unsigned char g_pushup_model_data[] = pushup_model_quantized_tflite;
   const int g_pushup_model_data_len = pushup_model_quantized_tflite_len;
   ```

3. **Update normalization in `src/main.cpp` (lines 54-65):**
   ```cpp
   // Copy mean and std from model_metadata.json
   float imu_mean[NUM_CHANNELS] = {...};
   float imu_std[NUM_CHANNELS] = {...};
   ```

4. **Upload and test!**
   ```
   pio run -t upload
   ```

## Troubleshooting

**If you still get "Failed to get registration from op code CUSTOM":**
- You used the LSTM model instead of CNN-only
- Retrain with `model_micro`

**If you get "Tensor allocation failed":**
- Model too large for 120 KB arena
- Reduce number of filters or layers
- Or increase `kTensorArenaSize` in main.cpp line 58

**If predictions are bad:**
- Check normalization params match training data
- Verify label order matches model output
- Try training for more epochs

## Quick Test

Want to test if your model works on ESP32?

After training in Colab, check the conversion output:
```python
# Section 12.1 - If this succeeds without errors:
converter = tf.lite.TFLiteConverter.from_saved_model(SAVED_MODEL_DIR)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
tflite_model = converter.convert()  # Should NOT error
```

If conversion succeeds → model is ESP32 compatible! ✅
If conversion fails → model uses unsupported ops ❌

## Next Steps

1. Update the notebook as described above
2. Train the CNN-only model
3. Download the .cc file
4. Update pushup_model_data.cpp
5. Upload to ESP32
6. You should see real predictions!
