# Project Summary: Classifying Pushup Posture Using a Single IMU

## 1. Goal

Build a machine-learning model that classifies **pushup posture** and/or **pushup phase** using data from **one IMU** (3-axis accelerometer + 3-axis gyroscope). This is for a class project with offline analysis (no real-time deployment needed).

Primary objectives:

- **Phase classification:** top, moving down, bottom, moving up, not-in-pushup  
- **Posture quality classification:** good form, hips sagging, hips too high (pike), partial ROM

---

## 2. IMU Placement

Because only one sensor is available, choose placement strategically.

### Recommended placement (for posture detection)
**Upper back / sternum** (center of chest or between shoulder blades)

- Captures torso orientation  
- Best for detecting sag/pike/neutral posture  
- Suitable for overall form assessment

### Alternative (for rep counting / phase detection)
**Forearm or wrist** (dorsal side)

- Produces strong, clear signals for up/down phase transitions  
- Less effective for posture quality  
- Better for rep counting

### Attachment guidelines

- Attach tightly using elastic straps, kinesio tape, or compression shirt pockets  
- Ensure consistent orientation (document axis directions) across sessions  
- Keep sensor stable and minimize sliding

---

## 3. Data Collection Protocol

### Sampling

- **Sampling rate:** 50–100 Hz  
- **Sensors:** Accelerometer (ax, ay, az), Gyroscope (gx, gy, gz)  
- Capture raw values with timestamps

### Session procedure (per participant)

1. **Calibration:**  
   - Stand upright and still for 3–5 seconds (optional)
2. **Record pushup sets** under controlled conditions:
   - 10 good-form pushups  
   - 10 reps with hips sagging  
   - 10 reps with hips too high / pike  
   - 10 partial ROM reps  
   - 10 slow pushups  
   - 10 normal-speed pushups  
3. **Record synchronized video** (side angle recommended for labeling)  
4. **Metadata for each session:**
   - Participant ID  
   - Height/weight (optional)  
   - IMU placement + orientation  
   - Sampling rate, date/time  
   - Notes on form or variations  

### Data variation requirements

To build a generalizable model:

- Include **multiple participants** (varied body types, fitness levels)  
- Collect **intentional incorrect form** examples  
- Capture **non-pushup movement** to help train “not-in-pushup” classification  

---

## 4. Labeling Strategy

### Understanding the Two Label Dimensions

The classification system uses **two independent labels** for each data sample:

#### Phase Labels (WHERE in the movement):
Describes the **position** in the pushup cycle:
- **`top`** - Arms fully extended, at starting position
- **`moving-down`** - Descending toward the ground
- **`bottom`** - Chest near ground, lowest position
- **`moving-up`** - Pushing back up
- **`not-in-pushup`** - Standing, resting, or other activities

Think of this as the **temporal state** of the movement.

#### Posture Labels (HOW WELL you're doing it):
Describes the **form quality** of the technique:
- **`good-form`** - Proper technique (straight body alignment, full range of motion)
- **`hips-sagging`** - Lower back arching, hips dropping (common error)
- **`hips-high`** - Pike position, butt in the air (common error)
- **`partial-rom`** - Not going through full range of motion (incomplete pushup)

Think of this as the **form quality**.

### Key Insight: Labels Are Independent!

You can have **any combination** of phase + posture:
- `moving-down` + `good-form` (properly descending)
- `bottom` + `hips-sagging` (at bottom position but with poor form)
- `top` + `partial-rom` (at top but didn't go all the way down)
- `moving-up` + `hips-high` (ascending but butt too high)

The ML model learns to classify **both dimensions simultaneously** from the IMU data.

### Using the Data Collector GUI

**Recommended approach:** Record SHORT sessions (3-10 seconds) for specific phase+posture combinations.

**For each recording session:**
1. Select BOTH a phase AND a posture label in the GUI
2. Click "Start Recording"
3. Perform that specific movement/position for 3-10 seconds
4. Click "Stop Recording"
5. Repeat for different combinations

**Example collection workflow:**

*Good Form Pushups:*
1. Select `top` + `good-form` → Record 5s holding top position
2. Select `moving-down` + `good-form` → Record 3-5 reps of slow descent
3. Select `bottom` + `good-form` → Record 5s holding bottom position
4. Select `moving-up` + `good-form` → Record 3-5 reps of ascent

*Bad Form Variations (repeat above with errors):*
5. Select `moving-down` + `hips-sagging` → Do pushups with sagging hips
6. Select `bottom` + `hips-sagging` → Hold bottom with sagging hips
7. Select `moving-down` + `hips-high` → Do pushups in pike position
8. Select `top` + `partial-rom` → Do partial pushups (only halfway down)

*Not Exercising:*
9. Select `not-in-pushup` + `good-form` → Stand/sit/walk around

This gives labeled training examples for each phase+posture combination the model needs to learn.

### Alternative: Video-based Post-Annotation

If recording longer continuous sessions, use the recorded video to assign ground-truth labels:

#### Labeling granularity

- **Rep-level labels** (simplest): label each rep with posture quality
- **Window-level labels** (more powerful): label short time intervals

#### Recommended label format

Each annotated interval should include:

- `start_timestamp`
- `end_timestamp`
- `phase_label`
- `posture_label`

---

## 5. Preprocessing Pipeline

### Steps

1. **Sync timestamps** with video annotation  
2. **Segment sensor data into windows**
   - Window length: **1–2 seconds**
   - Stride: **0.1–0.25 seconds**
   - Each window shape: **[time_steps × 6]**
3. **Normalize each channel**
   - Compute mean and std per channel on the training set  
   - Normalize accel and gyro separately  
4. **Optional filtering**
   - Low-pass at ~10–15 Hz  
   - Estimate gravity vector to compute pitch/roll of torso  
5. **Assign labels to each window**
   - Phase and posture labels inherited from annotated timeline  

---

## 6. Model Architecture Options

### Baseline (feature-based)

Extract features per window:

- Mean, std, min, max  
- Signal magnitude area  
- Energy & entropy  
- Dominant frequency  
- Estimated pitch/roll  

Train using:

- Random Forest  
- XGBoost  
- SVM  

### End-to-end deep learning (recommended)

Use raw `[T × 6]` windows.

Possible architectures:

- 1D CNN  
- LSTM / GRU  
- Temporal CNN (TCN)  
- Lightweight transformer  

Output:

- Softmax over **phase classes**  
- Softmax over **posture classes**

### Multi-task model

One encoder → two output heads:

- Head A: phase  
- Head B: posture  

Improves representation learning and overall accuracy.

---

## 7. Evaluation

Split dataset **by participant** to avoid overfitting.

Evaluation metrics:

- Overall accuracy  
- Per-class precision, recall, F1  
- Confusion matrix for posture and phase  
- Optional: rep-count accuracy  

---

## 8. What the LLM Should Help Generate

When feeding this document as context to another LLM, it should generate code for:

- IMU data collection scripts (Arduino/Python)  
- Data parsing and alignment  
- Sliding window segmentation  
- Feature extraction (if using classical ML)  
- Neural network architectures (PyTorch/Keras)  
- Training loops + evaluation scripts  
- Visualization of IMU signals and labels  
- Rep segmentation (optional)  
- Saving/loading models and preprocessing pipelines  

---
