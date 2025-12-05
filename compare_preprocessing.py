#!/usr/bin/env python3
"""
Compare Raw vs Preprocessed Data

Visualize the effect of preprocessing to ensure we're not cutting too much.

Usage:
    python3 compare_preprocessing.py
"""

import json
import numpy as np
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("matplotlib required! Install with: pip3 install matplotlib")


def load_session(filepath: str, session_idx: int = 0):
    """Load a specific session from a JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    sessions = data.get('sessions', [])
    if session_idx >= len(sessions):
        print(f"Session {session_idx} not found! File has {len(sessions)} sessions.")
        return None

    return sessions[session_idx]


def extract_imu_data(session):
    """Extract IMU data and time from session."""
    raw_data = session['data']
    imu_array = np.array([
        [sample['ax'], sample['ay'], sample['az'],
         sample['gx'], sample['gy'], sample['gz']]
        for sample in raw_data
    ])

    # Calculate time from sample index (40 Hz sample rate)
    # This is more reliable than elapsed_sec which may be calculated incorrectly
    time_array = np.arange(len(raw_data)) / 40.0

    return imu_array, time_array


def plot_comparison(raw_session, preprocessed_session):
    """Plot raw vs preprocessed data side by side."""
    if not MATPLOTLIB_AVAILABLE:
        return

    # Extract IMU data and time
    raw_imu, raw_time = extract_imu_data(raw_session)
    prep_imu, prep_time = extract_imu_data(preprocessed_session)

    # Adjust time to start from 0 for better visualization
    raw_time_from_zero = raw_time - raw_time[0]
    prep_time_from_zero = prep_time - prep_time[0] if len(prep_time) > 0 else prep_time

    # Create figure
    fig, axes = plt.subplots(4, 2, figsize=(16, 12))
    fig.suptitle('Raw vs Preprocessed Data Comparison', fontsize=16)

    # Column 0: Raw data
    # Column 1: Preprocessed data

    # Row 0: Accelerometer X, Y, Z
    ax = axes[0, 0]
    ax.plot(raw_time_from_zero, raw_imu[:, 0], label='ax', alpha=0.7)
    ax.plot(raw_time_from_zero, raw_imu[:, 1], label='ay', alpha=0.7)
    ax.plot(raw_time_from_zero, raw_imu[:, 2], label='az', linewidth=2)
    ax.set_ylabel('Acceleration (g)')
    ax.set_title(f'RAW - Accelerometer ({len(raw_imu)} samples, {raw_time_from_zero[-1]:.2f}s)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(prep_time_from_zero, prep_imu[:, 0], label='ax', alpha=0.7)
    ax.plot(prep_time_from_zero, prep_imu[:, 1], label='ay', alpha=0.7)
    ax.plot(prep_time_from_zero, prep_imu[:, 2], label='az', linewidth=2)
    ax.set_ylabel('Acceleration (g)')
    ax.set_title(f'PREPROCESSED - Accelerometer ({len(prep_imu)} samples, {prep_time_from_zero[-1]:.2f}s)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Row 1: Gyroscope X, Y, Z
    ax = axes[1, 0]
    ax.plot(raw_time_from_zero, raw_imu[:, 3], label='gx', alpha=0.7)
    ax.plot(raw_time_from_zero, raw_imu[:, 4], label='gy', alpha=0.7)
    ax.plot(raw_time_from_zero, raw_imu[:, 5], label='gz', alpha=0.7)
    ax.set_ylabel('Angular Velocity (deg/s)')
    ax.set_title('RAW - Gyroscope')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(prep_time_from_zero, prep_imu[:, 3], label='gx', alpha=0.7)
    ax.plot(prep_time_from_zero, prep_imu[:, 4], label='gy', alpha=0.7)
    ax.plot(prep_time_from_zero, prep_imu[:, 5], label='gz', alpha=0.7)
    ax.set_ylabel('Angular Velocity (deg/s)')
    ax.set_title('PREPROCESSED - Gyroscope')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Row 2: Z-axis acceleration (most important for phase detection)
    ax = axes[2, 0]
    ax.plot(raw_time_from_zero, raw_imu[:, 2], linewidth=2, color='green')
    ax.axhline(y=0.85, color='b', linestyle='--', alpha=0.5, label='top threshold')
    ax.axhline(y=0.60, color='r', linestyle='--', alpha=0.5, label='bottom threshold')
    ax.set_ylabel('az (g)')
    ax.set_title('RAW - Z-Axis Acceleration')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    ax = axes[2, 1]
    ax.plot(prep_time_from_zero, prep_imu[:, 2], linewidth=2, color='green')
    ax.axhline(y=0.85, color='b', linestyle='--', alpha=0.5, label='top threshold')
    ax.axhline(y=0.60, color='r', linestyle='--', alpha=0.5, label='bottom threshold')
    ax.set_ylabel('az (g)')
    ax.set_title('PREPROCESSED - Z-Axis Acceleration')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Row 3: Acceleration magnitude (to see movement)
    raw_accel = raw_imu[:, :3]
    raw_mag = np.linalg.norm(raw_accel, axis=1)
    prep_accel = prep_imu[:, :3]
    prep_mag = np.linalg.norm(prep_accel, axis=1)

    ax = axes[3, 0]
    ax.plot(raw_time_from_zero, raw_mag, linewidth=2, color='purple')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='gravity (1g)')
    ax.set_xlabel('Time (seconds, from start)')
    ax.set_ylabel('Magnitude (g)')
    ax.set_title('RAW - Acceleration Magnitude')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    ax = axes[3, 1]
    ax.plot(prep_time_from_zero, prep_mag, linewidth=2, color='purple')
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='gravity (1g)')
    ax.set_xlabel('Time (seconds, from start)')
    ax.set_ylabel('Magnitude (g)')
    ax.set_title('PREPROCESSED - Acceleration Magnitude')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_session_comparison(raw_session, preprocessed_session):
    """Print text comparison of sessions."""
    print("\n" + "="*70)
    print("SESSION COMPARISON")
    print("="*70)

    raw_imu, raw_time = extract_imu_data(raw_session)
    prep_imu, prep_time = extract_imu_data(preprocessed_session)

    print(f"\nRAW DATA:")
    print(f"  Samples: {len(raw_imu)}")
    print(f"  Duration: {len(raw_imu) / 40.0:.2f} seconds")
    print(f"  Posture: {raw_session.get('posture_label', 'N/A')}")
    print(f"  Phase: {raw_session.get('phase_label', 'N/A')}")

    print(f"\nPREPROCESSED DATA:")
    print(f"  Samples: {len(prep_imu)}")
    print(f"  Duration: {len(prep_imu) / 40.0:.2f} seconds")
    print(f"  Posture: {preprocessed_session.get('posture_label', 'N/A')}")
    print(f"  Phase: {preprocessed_session.get('phase_label', 'N/A')}")

    samples_removed = len(raw_imu) - len(prep_imu)
    pct_removed = (samples_removed / len(raw_imu)) * 100

    print(f"\nCHANGES:")
    print(f"  Samples removed: {samples_removed} ({pct_removed:.1f}%)")
    print(f"  Time removed: {samples_removed / 40.0:.2f} seconds")

    # Analyze what was cut
    raw_az = raw_imu[:, 2]
    prep_az = prep_imu[:, 2]

    print(f"\nZ-AXIS STATISTICS:")
    print(f"  Raw:  mean={np.mean(raw_az):.3f}, std={np.std(raw_az):.3f}, "
          f"min={np.min(raw_az):.3f}, max={np.max(raw_az):.3f}")
    print(f"  Prep: mean={np.mean(prep_az):.3f}, std={np.std(prep_az):.3f}, "
          f"min={np.min(prep_az):.3f}, max={np.max(prep_az):.3f}")

    # Check if we cut too much
    if len(prep_imu) < 40:  # Less than 1 second
        print(f"\n⚠️  WARNING: Preprocessed session is very short (<1 second)")
    elif pct_removed > 80:
        print(f"\n⚠️  WARNING: Removed {pct_removed:.1f}% of data (might be too much)")
    elif pct_removed < 10:
        print(f"\n⚠️  WARNING: Removed only {pct_removed:.1f}% of data (tail might still be present)")
    else:
        print(f"\n✓ Preprocessing looks reasonable ({pct_removed:.1f}% removed)")


def main():
    """Main comparison function."""
    if not MATPLOTLIB_AVAILABLE:
        return 1

    print("="*70)
    print("Compare Raw vs Preprocessed Data")
    print("="*70)

    # Configuration
    RAW_DIR = "raw_dataset"
    PREP_DIR = "preprocessed_dataset"

    # Find files
    raw_files = list(Path(RAW_DIR).glob("pushup_data_20251204*.json"))
    prep_files = list(Path(PREP_DIR).glob("pushup_data_20251204*.json"))

    if not raw_files:
        print(f"\n✗ No raw files found in {RAW_DIR}/")
        return 1

    if not prep_files:
        print(f"\n✗ No preprocessed files found in {PREP_DIR}/")
        print(f"Run preprocess_data.py first!")
        return 1

    # Use first matching file
    raw_file = raw_files[0]
    prep_file = Path(PREP_DIR) / raw_file.name

    if not prep_file.exists():
        print(f"\n✗ Preprocessed version of {raw_file.name} not found!")
        return 1

    print(f"\nComparing: {raw_file.name}")

    # Ask which session to compare
    with open(raw_file, 'r') as f:
        data = json.load(f)
    total_sessions = len(data.get('sessions', []))

    print(f"File contains {total_sessions} sessions")
    session_idx = int(input(f"Enter session index to compare (0-{total_sessions-1}): "))

    # Load sessions
    raw_session = load_session(str(raw_file), session_idx)
    prep_session = load_session(str(prep_file), session_idx)

    if raw_session is None or prep_session is None:
        return 1

    # Print comparison
    print_session_comparison(raw_session, prep_session)

    # Plot
    print("\nGenerating plots...")
    plot_comparison(raw_session, prep_session)

    return 0


if __name__ == "__main__":
    exit(main())
