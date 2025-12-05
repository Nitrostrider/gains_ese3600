#!/usr/bin/env python3
"""
Auto-Label Phase Data for Pushup Detection

This script processes existing pushup training data and automatically labels
the phase (at-top, moving, at-bottom) based on IMU sensor patterns.

Usage:
    python auto_label_phases.py

The script will:
1. Read all JSON files from pushup_data/ directory
2. Analyze IMU data (accelerometer Z-axis primarily)
3. Auto-label phase for each data window
4. Generate phase-labeled dataset for multi-task model training
"""

import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


def auto_label_phase(window: np.ndarray) -> str:
    """
    Auto-label pushup phase based on IMU patterns.

    Args:
        window: numpy array of shape (n_samples, 6) containing [ax, ay, az, gx, gy, gz]

    Returns:
        Phase label: "at-top", "moving", or "at-bottom"

    Heuristic Logic:
        - High Z-axis acceleration (~0.9-1.0 g) with low variance = at-top
        - Low Z-axis acceleration (~0.4-0.6 g) with low variance = at-bottom
        - Everything else (transitioning, high variance) = moving
    """
    # Extract Z-axis acceleration (index 2)
    az = window[:, 2]

    # Calculate statistics
    az_mean = np.mean(az)
    az_std = np.std(az)
    az_max = np.max(az)
    az_min = np.min(az)

    # Calculate velocity (derivative of position ~ integral of acceleration)
    # Approximate velocity by looking at change in az over time
    az_velocity = np.diff(az)  # Change in acceleration between samples

    # Detect direction changes: look for zero-crossings in velocity
    # At top/bottom, velocity should reverse (go from +/- to -/+)
    velocity_sign_changes = np.sum(np.diff(np.sign(az_velocity)) != 0)

    # Check if there's a clear peak or valley in the window
    peak_index = np.argmax(az)  # Index of highest acceleration
    valley_index = np.argmin(az)  # Index of lowest acceleration

    # Peak/valley is clearer if it's in the middle of the window (not at edges)
    peak_in_middle = 10 < peak_index < 40  # Not in first/last 10 samples
    valley_in_middle = 10 < valley_index < 40

    # Thresholds - carefully tuned to distinguish top vs bottom
    TOP_MEAN_MIN = 0.80          # Mean az must be high for top
    BOTTOM_MEAN_MAX = 0.70       # Mean az must be low for bottom
    STABLE_THRESHOLD = 0.20      # Std threshold for stable position
    TOP_PEAK_MIN = 0.85          # Peak value indicating top position
    BOTTOM_VALLEY_MAX = 0.60     # Valley value indicating bottom position

    # Classification logic - mutually exclusive checks
    # Check for at-top: high mean AND (low variance OR peak in middle OR direction change)
    is_high_mean = az_mean > TOP_MEAN_MIN
    is_stable = az_std < STABLE_THRESHOLD
    has_top_peak = az_max > TOP_PEAK_MIN

    if is_high_mean and has_top_peak:
        # Strong indicator of top position
        if is_stable or peak_in_middle or velocity_sign_changes >= 2:
            return "at-top"

    # Check for at-bottom: low mean AND (low variance OR valley in middle OR direction change)
    is_low_mean = az_mean < BOTTOM_MEAN_MAX
    has_bottom_valley = az_min < BOTTOM_VALLEY_MAX

    if is_low_mean and has_bottom_valley:
        # Strong indicator of bottom position
        if is_stable or valley_in_middle or velocity_sign_changes >= 2:
            return "at-bottom"

    # Fallback: stable positions based on mean alone (stricter thresholds)
    if az_mean > 0.85 and az_std < 0.15:
        return "at-top"
    elif az_mean < 0.65 and az_std < 0.15:
        return "at-bottom"

    # Default: moving (transitioning between positions)
    return "moving"


def load_session_data(json_path: str) -> List[Dict]:
    """Load a single training session JSON file and return list of sessions."""
    with open(json_path, 'r') as f:
        data = json.load(f)
        # Handle new format with 'sessions' array
        if 'sessions' in data:
            return data['sessions']
        else:
            # Old format - single session
            return [data]


def create_windows_with_phase_labels(
    session: Dict,
    window_size: int = 50,
    stride: int = 10
) -> List[Tuple[np.ndarray, str, str]]:
    """
    Create sliding windows with both posture and phase labels.

    Args:
        session: Dictionary containing IMU data and labels
        window_size: Number of samples per window (default: 50 @ 40Hz = 1.25s)
        stride: Step size for sliding window (default: 10 samples)

    Returns:
        List of tuples: (window_data, posture_label, phase_label)
    """
    # Extract IMU data from list of dictionaries
    # Each sample is: {'ax': ..., 'ay': ..., 'az': ..., 'gx': ..., 'gy': ..., 'gz': ...}
    raw_data = session['data']

    # Convert to numpy array: (n_samples, 6) with columns [ax, ay, az, gx, gy, gz]
    imu_data = np.array([
        [sample['ax'], sample['ay'], sample['az'],
         sample['gx'], sample['gy'], sample['gz']]
        for sample in raw_data
    ])

    posture_label = session['posture_label']

    windows = []

    # Create sliding windows
    for i in range(0, len(imu_data) - window_size + 1, stride):
        window = imu_data[i:i+window_size]

        # Auto-label phase
        phase_label = auto_label_phase(window)

        windows.append((window, posture_label, phase_label))

    return windows


def process_all_sessions(data_dir: str) -> Dict[str, List]:
    """
    Process all JSON files in the data directory.

    Args:
        data_dir: Path to directory containing JSON session files

    Returns:
        Dictionary containing all labeled windows and statistics
    """
    data_path = Path(data_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    # Find all JSON files
    json_files = list(data_path.glob("pushup_data_20251204*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {data_dir}")

    print(f"Found {len(json_files)} session files")

    all_windows = []
    phase_counts = {"at-top": 0, "moving": 0, "at-bottom": 0}
    posture_counts = {}

    # Process each JSON file
    for json_file in json_files:
        try:
            sessions = load_session_data(str(json_file))
            file_window_count = 0

            # Process each session in the file
            for session in sessions:
                windows = create_windows_with_phase_labels(session)

                # Collect statistics
                for window, posture, phase in windows:
                    all_windows.append({
                        'window': window.tolist(),
                        'posture_label': posture,
                        'phase_label': phase
                    })

                    phase_counts[phase] += 1
                    posture_counts[posture] = posture_counts.get(posture, 0) + 1

                file_window_count += len(windows)

            print(f"  ✓ Processed {json_file.name}: {len(sessions)} sessions, {file_window_count} windows")

        except Exception as e:
            print(f"  ✗ Error processing {json_file.name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n=== Dataset Statistics ===")
    print(f"Total windows: {len(all_windows)}")
    print(f"\nPhase distribution:")
    for phase, count in phase_counts.items():
        pct = (count / len(all_windows)) * 100
        print(f"  {phase:12s}: {count:4d} ({pct:5.1f}%)")

    print(f"\nPosture distribution:")
    for posture, count in sorted(posture_counts.items()):
        pct = (count / len(all_windows)) * 100
        print(f"  {posture:15s}: {count:4d} ({pct:5.1f}%)")

    return {
        'windows': all_windows,
        'phase_counts': phase_counts,
        'posture_counts': posture_counts,
        'total_count': len(all_windows)
    }


def validate_auto_labels(
    dataset: Dict,
    sample_size: int = 10
) -> None:
    """
    Print sample windows for manual validation of auto-labeling.

    Args:
        dataset: Dataset dictionary from process_all_sessions()
        sample_size: Number of random samples to display
    """
    print(f"\n=== Validation Samples (Random {sample_size}) ===")

    windows = dataset['windows']
    indices = np.random.choice(len(windows), min(sample_size, len(windows)), replace=False)

    for idx in indices:
        w = windows[idx]
        window_data = np.array(w['window'])
        az_mean = np.mean(window_data[:, 2])
        az_std = np.std(window_data[:, 2])

        print(f"\nSample {idx}:")
        print(f"  Posture: {w['posture_label']}")
        print(f"  Phase: {w['phase_label']}")
        print(f"  az_mean: {az_mean:.3f}, az_std: {az_std:.3f}")


def save_labeled_dataset(
    dataset: Dict,
    output_path: str = "pushup_data_phase_labeled.json"
) -> None:
    """
    Save the phase-labeled dataset to a JSON file.

    Args:
        dataset: Dataset dictionary from process_all_sessions()
        output_path: Path for output JSON file
    """
    output = {
        'metadata': {
            'window_size': 50,
            'stride': 10,
            'total_windows': dataset['total_count'],
            'phase_counts': dataset['phase_counts'],
            'posture_counts': dataset['posture_counts'],
            'phase_labels': ['at-top', 'moving', 'at-bottom'],
            'posture_labels': ['good-form', 'hips-high', 'hips-sagging', 'partial-rom']
        },
        'windows': dataset['windows']
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Saved labeled dataset to {output_path}")
    print(f"  File size: {os.path.getsize(output_path) / (1024*1024):.1f} MB")


def main():
    """Main execution function."""
    print("=" * 60)
    print("Auto-Label Phase Data for Pushup Detection")
    print("=" * 60)

    # Configuration
    DATA_DIR = "preprocessed_dataset"  # Use preprocessed data
    OUTPUT_FILE = "pushup_data_phase_labeled.json"

    try:
        # Process all sessions
        dataset = process_all_sessions(DATA_DIR)

        # Validate auto-labeling
        validate_auto_labels(dataset, sample_size=10)

        # Save labeled dataset
        save_labeled_dataset(dataset, OUTPUT_FILE)

        print("\n" + "=" * 60)
        print("✓ Phase labeling complete!")
        print("=" * 60)
        print("\nNext steps:")
        print("1. Review validation samples above")
        print("2. Manually check 10% of labels if needed")
        print(f"3. Use {OUTPUT_FILE} for multi-task model training")

    except Exception as e:
        print(f"\n✗ Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
