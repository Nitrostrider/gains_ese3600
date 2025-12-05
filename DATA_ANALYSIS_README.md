# Data Analysis Notebook - Pushup IMU Data

This notebook (`data_analysis.ipynb`) provides comprehensive analysis and visualization of pushup IMU sensor data collected from the ESP32-based data collector.

## Overview

The notebook analyzes accelerometer and gyroscope data from push-up sessions to:
- Explore sensor data distributions and patterns
- Compare different posture types (good-form, hips-sagging, hips-high, partial-rom)
- Track 3D chest movement through space via double integration
- Generate insights for model training and validation

## Dataset Structure

**Input Data:**
- JSON files from `raw_dataset/pushup_data_*.json`
- Each file contains multiple sessions with labeled push-up data
- Sample rate: 40 Hz
- Sensor placement: Sternum (chest)

**Data Fields:**
- `ax, ay, az`: Accelerometer readings in g (gravity units)
- `gx, gy, gz`: Gyroscope readings in deg/s
- `phase_label`: Push-up phase (currently all "top")
- `posture_label`: Form classification (good-form, hips-sagging, hips-high, partial-rom)

## Requirements

```bash
# Install required packages
pip install pandas numpy matplotlib seaborn
```

**Python Packages:**
- `pandas`: Data manipulation
- `numpy`: Numerical operations
- `matplotlib`: Plotting and visualization
- `seaborn`: Statistical visualizations

## How to Use

1. **Place your data files** in the `raw_dataset/` directory
   - Files should match pattern: `pushup_data_*.json`
   - Or modify the glob pattern in Cell 3 to point to your data

2. **Run the notebook** in Jupyter:
   ```bash
   jupyter notebook data_analysis.ipynb
   ```

3. **Execute cells sequentially** (Runtime → Run all, or run cell by cell)

## Notebook Sections

### 1. Data Loading
- Loads all JSON files from the dataset directory
- Combines multiple files into a unified DataFrame
- Creates unique session IDs to prevent collisions across files

### 2. Data Exploration
- **Basic statistics**: Mean, std, min, max for each sensor axis
- **Label distribution**: Count of sessions per posture type
- **Sample visualization**: Time-series plots of raw IMU data

### 3. Data Quality
- **Missing value check**: Identifies incomplete data
- **Outlier detection**: Finds values beyond 3 standard deviations
- **Session statistics**: Aggregated metrics per session

### 4. Feature Engineering
- **Magnitude calculation**:
  - `accel_magnitude = √(ax² + ay² + az²)`
  - `gyro_magnitude = √(gx² + gy² + gz²)`
- **Correlation analysis**: Heatmap showing relationships between sensor axes

### 5. Posture Comparison
- **Histogram overlays**: Compare sensor distributions across posture types
- **Visualizations**: Separate plots for each axis and posture label
- Helps identify distinguishing features for each form type

### 6. 3D Position Tracking ⭐

**What it does:**
Integrates accelerometer data twice to estimate the 3D trajectory of the chest during push-ups.

**Method:**
1. **Gravity removal**: Estimates static gravity from first 10 samples
2. **First integration**: Acceleration → Velocity (m/s)
3. **Second integration**: Velocity → Position (meters)

**Visualizations:**
- **Time-series plots**: Shows acceleration → velocity → position transformation
- **3D trajectory**: Spatial path with color-coded time progression
- **Multiple views**: Top (X-Y), Side (X-Z), Front (Y-Z) projections
- **Multi-session comparison**: Overlay trajectories from different posture types

**Important Notes:**
- ⚠️ **IMU Drift**: Position estimates accumulate error over time
  - This is normal for inertial navigation systems
  - Small measurement errors compound during integration
  - Best accuracy for short movements (1-2 seconds)

- ⚠️ **Simplified Model**: Assumes minimal sensor rotation
  - For more accuracy, gyroscope-based orientation correction needed
  - Good for visualizing general movement patterns

- ⚠️ **Starting position**: All trajectories start at origin (0, 0, 0)
  - Only relative movement is tracked, not absolute position

**Expected Patterns:**
- **Good form**: Straighter vertical motion (Z-axis dominant)
- **Hips-sagging**: Curved trajectory, more Y/Z variation
- **Hips-high**: Different curved pattern
- **Partial-ROM**: Smaller displacement magnitudes

### 7. Session Statistics
- Aggregated metrics per session (mean, std, min, max)
- Includes filename and unique session ID for tracking
- Can be exported to CSV for further analysis

## Understanding the Coordinate System

**Sensor Orientation (when placed on sternum):**
- **X-axis**: Forward/backward (perpendicular to chest)
- **Y-axis**: Left/right (horizontal across chest)
- **Z-axis**: Up/down (vertical along body)

**During a push-up:**
- **Good form**: Primarily Z-axis movement (vertical)
- **Poor form**: Additional X/Y components indicate misalignment

## Key Insights from Analysis

### Data Statistics (from current dataset)
- **Total sessions**: 277
- **Total data points**: 22,880
- **Posture distribution**:
  - Good-form: 155 sessions (56%)
  - Hips-high: 41 sessions (15%)
  - Hips-sagging: 40 sessions (14%)
  - Partial-ROM: 41 sessions (15%)

### Sensor Characteristics
- **Accelerometer**: Ranges from -1g to +2g
  - Mean ~0.93g on Z-axis (gravity effect)
- **Gyroscope**: Ranges from -168°/s to +213°/s
  - Higher variability indicates dynamic movement

### Outliers
- 0.69% - 1.85% outliers detected (beyond 3σ)
- Normal for dynamic movements during push-ups
- Most outliers occur during rapid transitions

## Sliding Window Concept

The notebook uses **unique session IDs** (format: `{file_index}_{session_id}`) to handle data from multiple files:
- Prevents ID collisions when combining multiple data collection sessions
- Allows tracking of which file each session came from
- Essential for training data organization

**Note**: This is different from the sliding windows used in model training (see `MODEL_TRAINING_GUIDE.md`).

## Exporting Results

Uncomment the export cells (Section 11) to save:
- `processed_pushup_data.csv`: Full dataset with all features
- `session_statistics.csv`: Aggregated per-session metrics

```python
# Uncomment these lines in Cell 26:
df.to_csv('processed_pushup_data.csv', index=False)
session_stats.to_csv('session_statistics.csv', index=False)
```

## Troubleshooting

**Issue: "No JSON files found"**
- Ensure files are in `raw_dataset/` directory
- Check the glob pattern in Cell 3 matches your filenames

**Issue: "Module not found"**
- Install missing packages: `pip install pandas numpy matplotlib seaborn`

**Issue: "3D plot not showing"**
- Make sure `%matplotlib inline` or `%matplotlib notebook` is enabled
- Try running in Jupyter Notebook (not JupyterLab for best 3D support)

**Issue: Position estimates look wrong**
- This is normal due to IMU drift
- Longer sessions accumulate more error
- Focus on relative movement patterns, not absolute values

## Next Steps

After analyzing your data:
1. **Identify patterns**: Look for distinguishing features in each posture type
2. **Check data quality**: Ensure consistent labeling and sufficient samples
3. **Train model**: Use insights to guide feature selection in `pushup_model_colab.ipynb`
4. **Validate results**: Compare model predictions with visualization insights

## Related Files

- `DATA_COLLECTOR_README.md`: How to collect IMU data
- `MODEL_TRAINING_GUIDE.md`: How to train the classification model
- `pushup_model_colab.ipynb`: Model training notebook
- `raw_dataset/`: Directory containing collected data files

## Technical Details

### Integration Method

**Numerical Integration:**
```python
# First integration (cumulative sum)
velocity = Σ(acceleration × Δt)

# Second integration
position = Σ(velocity × Δt)
```

Where `Δt = 1/40 = 0.025 seconds` (40 Hz sample rate)

**Gravity Removal:**
```python
gravity_vector = mean(first_10_samples)
accel_nograv = accel_raw - gravity_vector
```

### Coordinate Transformations

The current implementation assumes:
- Sensor frame ≈ World frame (minimal rotation)
- For accurate tracking during large rotations, quaternion-based orientation tracking would be needed

## Questions?

For questions or issues:
1. Check the inline comments in the notebook
2. Review the `MODEL_TRAINING_GUIDE.md` for context
3. Ensure your data matches the expected JSON format

---

**Last Updated**: December 2025
**Compatible with**: Python 3.8+, Jupyter Notebook
