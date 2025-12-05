#!/usr/bin/env python3
"""
Validate Phase Labels - Manual Inspection Tool

This script loads the auto-labeled phase data and displays samples
for manual validation. You can verify that the phase labels match
the IMU patterns.

Usage:
    python3 validate_phase_labels.py
"""

import json
import numpy as np
import random

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Note: matplotlib not available. Install with: pip3 install matplotlib for plotting")


def load_labeled_data(filepath: str = "pushup_data_phase_labeled.json"):
    """Load the phase-labeled dataset."""
    with open(filepath, 'r') as f:
        return json.load(f)


def display_window_summary(window_data: dict, index: int):
    """Display summary statistics for a window."""
    window = np.array(window_data['window'])  # Shape: (50, 6)

    # Extract channels
    ax = window[:, 0]
    ay = window[:, 1]
    az = window[:, 2]
    gx = window[:, 3]
    gy = window[:, 4]
    gz = window[:, 5]

    # Calculate statistics
    az_mean = np.mean(az)
    az_std = np.std(az)
    az_min = np.min(az)
    az_max = np.max(az)

    # Acceleration magnitude
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    accel_mag_mean = np.mean(accel_mag)

    # Gyro magnitude (indicates rotation)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    gyro_mag_mean = np.mean(gyro_mag)

    print(f"\n{'='*70}")
    print(f"Window #{index}")
    print(f"{'='*70}")
    print(f"Posture Label: {window_data['posture_label']}")
    print(f"Phase Label:   {window_data['phase_label']}")
    print(f"\nZ-Axis Acceleration (az) Statistics:")
    print(f"  Mean:   {az_mean:7.4f} g")
    print(f"  Std:    {az_std:7.4f} g")
    print(f"  Min:    {az_min:7.4f} g")
    print(f"  Max:    {az_max:7.4f} g")
    print(f"\nOther Indicators:")
    print(f"  Accel Magnitude: {accel_mag_mean:7.4f} g (should be ~1.0 g at rest)")
    print(f"  Gyro Magnitude:  {gyro_mag_mean:7.2f} deg/s (high = rotating/moving)")

    # Interpretation hints
    print(f"\nInterpretation:")
    if az_mean > 0.85 and az_std < 0.15:
        print("  ✓ High stable az → Should be 'at-top'")
    elif az_mean < 0.65 and az_std < 0.15:
        print("  ✓ Low stable az → Should be 'at-bottom'")
    else:
        print("  ✓ Mid-range or high variance → Should be 'moving'")

    # Show first few samples
    print(f"\nFirst 5 samples (showing az values):")
    for i in range(min(5, len(az))):
        print(f"  Sample {i}: az = {az[i]:.4f} g")

    print(f"{'='*70}")


def interactive_validation(data: dict):
    """Interactive validation mode."""
    windows = data['windows']
    total = len(windows)

    print(f"\n{'='*70}")
    print(f"INTERACTIVE VALIDATION MODE")
    print(f"{'='*70}")
    print(f"Total windows: {total}")
    print(f"\nPhase distribution:")
    for phase, count in data['metadata']['phase_counts'].items():
        pct = (count / total) * 100
        print(f"  {phase:12s}: {count:4d} ({pct:5.1f}%)")

    print(f"\nCommands:")
    print(f"  [number] - View specific window by index")
    print(f"  'r'      - View random window")
    print(f"  'top'    - View random 'at-top' window")
    print(f"  'bottom' - View random 'at-bottom' window")
    print(f"  'moving' - View random 'moving' window")
    if MATPLOTLIB_AVAILABLE:
        print(f"  'plot'   - Plot current/last window")
        print(f"  'compare'- Plot phase comparison")
    print(f"  'q'      - Quit")

    last_window_idx = None

    while True:
        print(f"\n" + "-"*70)
        cmd = input("Enter command: ").strip().lower()

        if cmd == 'q':
            print("Exiting validation.")
            break
        elif cmd == 'r':
            idx = random.randint(0, total - 1)
            display_window_summary(windows[idx], idx)
            last_window_idx = idx
        elif cmd in ['top', 'at-top']:
            # Find random at-top window
            at_top = [i for i, w in enumerate(windows) if w['phase_label'] == 'at-top']
            if at_top:
                idx = random.choice(at_top)
                display_window_summary(windows[idx], idx)
                last_window_idx = idx
            else:
                print("No 'at-top' windows found!")
        elif cmd in ['bottom', 'at-bottom']:
            # Find random at-bottom window
            at_bottom = [i for i, w in enumerate(windows) if w['phase_label'] == 'at-bottom']
            if at_bottom:
                idx = random.choice(at_bottom)
                display_window_summary(windows[idx], idx)
                last_window_idx = idx
            else:
                print("No 'at-bottom' windows found!")
        elif cmd == 'moving':
            # Find random moving window
            moving = [i for i, w in enumerate(windows) if w['phase_label'] == 'moving']
            if moving:
                idx = random.choice(moving)
                display_window_summary(windows[idx], idx)
                last_window_idx = idx
            else:
                print("No 'moving' windows found!")
        elif cmd == 'plot':
            if last_window_idx is not None:
                plot_window(windows[last_window_idx], last_window_idx)
            else:
                print("No window to plot! View a window first.")
        elif cmd == 'compare':
            plot_phase_comparison(data, num_samples=3)
        elif cmd.isdigit():
            idx = int(cmd)
            if 0 <= idx < total:
                display_window_summary(windows[idx], idx)
                last_window_idx = idx
            else:
                print(f"Invalid index! Must be 0-{total-1}")
        else:
            print("Unknown command. Try 'r', 'top', 'bottom', 'moving', 'plot', 'compare', or a number.")


def quick_validation(data: dict, num_samples: int = 10):
    """Quick validation - show random samples from each phase."""
    windows = data['windows']

    print(f"\n{'='*70}")
    print(f"QUICK VALIDATION - {num_samples} Random Samples per Phase")
    print(f"{'='*70}")

    for phase in ['at-top', 'moving', 'at-bottom']:
        print(f"\n\n{'#'*70}")
        print(f"# PHASE: {phase.upper()}")
        print(f"{'#'*70}")

        # Find all windows with this phase
        phase_windows = [(i, w) for i, w in enumerate(windows) if w['phase_label'] == phase]

        if not phase_windows:
            print(f"No windows found for phase '{phase}'")
            continue

        # Sample randomly
        sample_count = min(num_samples, len(phase_windows))
        samples = random.sample(phase_windows, sample_count)

        for idx, window in samples:
            display_window_summary(window, idx)


def plot_window(window_data: dict, index: int):
    """Plot a single window with its phase label."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib required for plotting! Install with: pip3 install matplotlib")
        return

    window = np.array(window_data['window'])  # Shape: (50, 6)

    # Extract channels
    ax = window[:, 0]
    ay = window[:, 1]
    az = window[:, 2]
    gx = window[:, 3]
    gy = window[:, 4]
    gz = window[:, 5]

    # Time axis (50 samples @ 40Hz = 1.25 seconds)
    time = np.linspace(0, 1.25, 50)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

    # Plot accelerometer
    ax1.plot(time, ax, label='ax', alpha=0.7)
    ax1.plot(time, ay, label='ay', alpha=0.7)
    ax1.plot(time, az, label='az', linewidth=2)
    ax1.axhline(y=0.85, color='g', linestyle='--', alpha=0.5, label='at-top threshold')
    ax1.axhline(y=0.65, color='r', linestyle='--', alpha=0.5, label='at-bottom threshold')
    ax1.set_ylabel('Acceleration (g)')
    ax1.set_title(f'Window #{index} - Phase: {window_data["phase_label"]} | Posture: {window_data["posture_label"]}')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot gyroscope
    ax2.plot(time, gx, label='gx', alpha=0.7)
    ax2.plot(time, gy, label='gy', alpha=0.7)
    ax2.plot(time, gz, label='gz', alpha=0.7)
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Angular Velocity (deg/s)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_phase_comparison(data: dict, num_samples: int = 3):
    """Plot comparison of all three phases side by side."""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib required for plotting! Install with: pip3 install matplotlib")
        return

    windows = data['windows']

    # Get samples from each phase
    samples = {}
    for phase in ['at-top', 'moving', 'at-bottom']:
        phase_windows = [w for w in windows if w['phase_label'] == phase]
        if phase_windows:
            samples[phase] = random.sample(phase_windows, min(num_samples, len(phase_windows)))

    # Create figure
    fig, axes = plt.subplots(3, num_samples, figsize=(16, 10))
    fig.suptitle('Phase Label Comparison - Z-Axis Acceleration', fontsize=16)

    time = np.linspace(0, 1.25, 50)

    for row, phase in enumerate(['at-top', 'moving', 'at-bottom']):
        if phase not in samples:
            continue

        for col, window_data in enumerate(samples[phase]):
            window = np.array(window_data['window'])
            az = window[:, 2]

            ax = axes[row, col] if num_samples > 1 else axes[row]

            ax.plot(time, az, linewidth=2, color='blue')
            ax.axhline(y=0.85, color='g', linestyle='--', alpha=0.3)
            ax.axhline(y=0.65, color='r', linestyle='--', alpha=0.3)
            ax.set_ylim(0, 1.2)
            ax.grid(True, alpha=0.3)

            if col == 0:
                ax.set_ylabel(f'{phase}\naz (g)')
            if row == 2:
                ax.set_xlabel('Time (s)')

            # Add statistics
            az_mean = np.mean(az)
            az_std = np.std(az)
            ax.set_title(f'μ={az_mean:.2f}, σ={az_std:.2f}', fontsize=10)

    plt.tight_layout()
    plt.show()


def check_label_consistency(data: dict):
    """Check for potential labeling issues."""
    windows = data['windows']

    print(f"\n{'='*70}")
    print(f"CONSISTENCY CHECK")
    print(f"{'='*70}")

    issues = []

    for i, w in enumerate(windows):
        window = np.array(w['window'])
        az = window[:, 2]
        az_mean = np.mean(az)
        az_std = np.std(az)
        phase = w['phase_label']

        # Check for inconsistencies
        if phase == 'at-top' and az_mean < 0.85:
            issues.append((i, f"at-top but az_mean={az_mean:.3f} < 0.85"))
        elif phase == 'at-top' and az_std > 0.15:
            issues.append((i, f"at-top but az_std={az_std:.3f} > 0.15 (unstable)"))
        elif phase == 'at-bottom' and az_mean > 0.65:
            issues.append((i, f"at-bottom but az_mean={az_mean:.3f} > 0.65"))
        elif phase == 'at-bottom' and az_std > 0.15:
            issues.append((i, f"at-bottom but az_std={az_std:.3f} > 0.15 (unstable)"))

    if issues:
        print(f"\nFound {len(issues)} potential labeling issues:")
        print(f"(These may be edge cases, not necessarily errors)\n")
        for idx, issue in issues[:20]:  # Show first 20
            print(f"  Window {idx:4d}: {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more")
    else:
        print("\n✓ No obvious labeling inconsistencies found!")

    # Calculate percentage
    pct_issues = (len(issues) / len(windows)) * 100
    print(f"\nIssue rate: {pct_issues:.1f}% ({len(issues)}/{len(windows)} windows)")


def main():
    """Main validation function."""
    print("="*70)
    print("Phase Label Validation Tool")
    print("="*70)

    # Load data
    try:
        data = load_labeled_data()
        print(f"\n✓ Loaded dataset: {data['metadata']['total_windows']} windows")
    except FileNotFoundError:
        print("\n✗ Error: pushup_data_phase_labeled.json not found!")
        print("Run auto_label_phases.py first to generate the labeled data.")
        return 1

    print("\nValidation modes:")
    print("  1. Quick validation (10 random samples per phase)")
    print("  2. Consistency check (find potential issues)")
    print("  3. Interactive mode (explore specific windows)")
    print("  4. All of the above")
    if MATPLOTLIB_AVAILABLE:
        print("  5. Plot phase comparison")

    choice = input("\nEnter choice (1-5): ").strip()

    if choice == '1':
        quick_validation(data, num_samples=10)
    elif choice == '2':
        check_label_consistency(data)
    elif choice == '3':
        interactive_validation(data)
    elif choice == '4':
        quick_validation(data, num_samples=5)
        check_label_consistency(data)
        print("\n\nStarting interactive mode...")
        interactive_validation(data)
    elif choice == '5' and MATPLOTLIB_AVAILABLE:
        plot_phase_comparison(data, num_samples=3)
    else:
        print("Invalid choice!")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
