#!/usr/bin/env python3
"""
Preprocess Pushup Data - Remove Static Tails

This script removes static periods at the beginning and end of pushup sessions
where the sensor is not moving (person holding still after completing pushup).

Usage:
    python3 preprocess_data.py
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List


def detect_movement(imu_data: np.ndarray, accel_threshold: float = 0.08, gyro_threshold: float = 15.0) -> np.ndarray:
    """
    Detect which samples contain movement vs static periods.
    Uses both accelerometer and gyroscope data for more robust detection.

    Args:
        imu_data: Array of shape (n_samples, 6) [ax, ay, az, gx, gy, gz]
        accel_threshold: Movement threshold for acceleration variance
        gyro_threshold: Movement threshold for gyroscope magnitude (deg/s)

    Returns:
        Boolean array indicating which samples have movement
    """
    # Extract accelerometer and gyroscope data
    accel = imu_data[:, :3]  # ax, ay, az
    gyro = imu_data[:, 3:]   # gx, gy, gz

    # Calculate magnitudes
    accel_mag = np.linalg.norm(accel, axis=1)
    gyro_mag = np.linalg.norm(gyro, axis=1)

    # Calculate variance over sliding window for acceleration
    window_size = 10  # ~0.25 seconds @ 40Hz
    is_moving = np.zeros(len(imu_data), dtype=bool)

    for i in range(len(imu_data)):
        start = max(0, i - window_size // 2)
        end = min(len(imu_data), i + window_size // 2)

        # Check acceleration variance
        accel_window = accel_mag[start:end]
        accel_variance = np.var(accel_window)

        # Check gyroscope magnitude (rotation indicates movement)
        gyro_window = gyro_mag[start:end]
        gyro_max = np.max(gyro_window)

        # Movement detected if either condition is met
        is_moving[i] = (accel_variance > accel_threshold) or (gyro_max > gyro_threshold)

    return is_moving


def trim_static_tails(imu_data: np.ndarray, min_duration: float = 0.5) -> np.ndarray:
    """
    Trim static periods from beginning and end of session.

    Args:
        imu_data: Array of shape (n_samples, 6)
        min_duration: Minimum duration (seconds) to keep @ 40Hz

    Returns:
        Trimmed IMU data
    """
    is_moving = detect_movement(imu_data)

    # Find first and last movement
    moving_indices = np.where(is_moving)[0]

    if len(moving_indices) == 0:
        # No movement detected - keep all data (might be a static posture hold)
        print("  Warning: No movement detected in session")
        return imu_data

    # Add small padding before first and after last movement
    # Reduced from 20 to 8 samples (0.2 seconds instead of 0.5 seconds)
    padding = 8  # 0.2 seconds @ 40Hz
    first_movement = max(0, moving_indices[0] - padding)
    last_movement = min(len(imu_data) - 1, moving_indices[-1] + padding)

    # Ensure minimum duration
    min_samples = int(min_duration * 40)
    if last_movement - first_movement < min_samples:
        # Expand to minimum duration, centered on the movement
        mid = (first_movement + last_movement) // 2
        first_movement = max(0, mid - min_samples // 2)
        last_movement = min(len(imu_data), mid + min_samples // 2)

    return imu_data[first_movement:last_movement + 1]


def preprocess_session(session: Dict) -> Dict:
    """
    Preprocess a single session by trimming static tails.

    Args:
        session: Session dictionary with 'data' field

    Returns:
        Preprocessed session dictionary
    """
    # Extract IMU data
    raw_data = session['data']
    imu_array = np.array([
        [sample['ax'], sample['ay'], sample['az'],
         sample['gx'], sample['gy'], sample['gz']]
        for sample in raw_data
    ])

    # Trim static tails
    trimmed_imu = trim_static_tails(imu_array)

    # Check if we trimmed too much
    if len(trimmed_imu) < 40:  # Less than 1 second
        print(f"  Warning: Session {session.get('session_id', '?')} trimmed to {len(trimmed_imu)} samples (very short)")

    # Reconstruct data with timestamps
    trimmed_data = []
    for i, imu_sample in enumerate(trimmed_imu):
        original_idx = i  # Approximate - we've trimmed from start
        if original_idx < len(raw_data):
            original_sample = raw_data[original_idx]
            trimmed_data.append({
                'timestamp': original_sample.get('timestamp', ''),
                'elapsed_sec': i / 40.0,  # Recalculate elapsed time
                'ax': float(imu_sample[0]),
                'ay': float(imu_sample[1]),
                'az': float(imu_sample[2]),
                'gx': float(imu_sample[3]),
                'gy': float(imu_sample[4]),
                'gz': float(imu_sample[5])
            })

    # Update session
    preprocessed = session.copy()
    preprocessed['data'] = trimmed_data
    preprocessed['sample_count'] = len(trimmed_data)
    preprocessed['duration_sec'] = len(trimmed_data) / 40.0
    preprocessed['preprocessed'] = True
    preprocessed['original_sample_count'] = len(raw_data)

    return preprocessed


def preprocess_all_files(input_dir: str, output_dir: str):
    """
    Preprocess all JSON files in input directory.

    Args:
        input_dir: Directory containing raw JSON files
        output_dir: Directory to save preprocessed files
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Create output directory
    output_path.mkdir(exist_ok=True)

    # Find all JSON files matching the pattern
    json_files = list(input_path.glob("pushup_data_20251204*.json"))

    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return

    print(f"Found {len(json_files)} files to preprocess")

    total_sessions = 0
    total_samples_before = 0
    total_samples_after = 0

    for json_file in json_files:
        print(f"\nProcessing {json_file.name}...")

        # Load file
        with open(json_file, 'r') as f:
            data = json.load(f)

        sessions = data.get('sessions', [])
        preprocessed_sessions = []

        for session in sessions:
            original_count = len(session['data'])
            preprocessed = preprocess_session(session)
            new_count = len(preprocessed['data'])

            preprocessed_sessions.append(preprocessed)

            samples_removed = original_count - new_count
            pct_removed = (samples_removed / original_count) * 100

            print(f"  Session {session.get('session_id', '?')}: "
                  f"{original_count} → {new_count} samples "
                  f"({pct_removed:.1f}% removed)")

            total_samples_before += original_count
            total_samples_after += new_count

        total_sessions += len(sessions)

        # Save preprocessed file
        output_data = data.copy()
        output_data['sessions'] = preprocessed_sessions
        output_data['metadata']['preprocessed'] = True
        output_data['metadata']['total_sessions'] = len(preprocessed_sessions)

        output_file = output_path / json_file.name
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)

        print(f"  ✓ Saved to {output_file}")

    print(f"\n{'='*70}")
    print(f"Preprocessing Complete")
    print(f"{'='*70}")
    print(f"Total sessions: {total_sessions}")
    print(f"Samples before: {total_samples_before}")
    print(f"Samples after:  {total_samples_after}")
    print(f"Samples removed: {total_samples_before - total_samples_after} "
          f"({(total_samples_before - total_samples_after) / total_samples_before * 100:.1f}%)")


def main():
    """Main preprocessing function."""
    print("="*70)
    print("Pushup Data Preprocessing - Remove Static Tails")
    print("="*70)

    INPUT_DIR = "raw_dataset"
    OUTPUT_DIR = "preprocessed_dataset"

    try:
        preprocess_all_files(INPUT_DIR, OUTPUT_DIR)

        print(f"\n✓ Preprocessing complete!")
        print(f"\nNext steps:")
        print(f"1. Review preprocessed data in {OUTPUT_DIR}/")
        print(f"2. Update auto_label_phases.py to use {OUTPUT_DIR}/ instead of {INPUT_DIR}/")
        print(f"3. Run auto_label_phases.py to generate phase labels")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
