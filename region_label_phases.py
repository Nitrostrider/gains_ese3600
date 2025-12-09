import json
import numpy as np
import os
import glob
import matplotlib.pyplot as plt
from auto_label_phases import gaussian_smooth, load_all_json_files


###############################################################################
# Region-Based Phase Labeling
###############################################################################

def label_regions_simple(imu, min_region_size=20):
    """
    Label contiguous regions of a session based on smoothed acceleration.

    Uses a simple state machine approach:
    - Detect peaks (at-bottom) when az is high
    - Detect plateaus (at-top) when az is stable and moderate
    - Everything else is moving

    Args:
        imu: (N, 6) numpy array containing [ax, ay, az, gx, gy, gz]
        min_region_size: Minimum samples for a region (prevents flickering)

    Returns:
        regions: List of (start_idx, end_idx, phase_label) tuples
        labels: Per-sample phase labels (same length as imu)
    """
    az = gaussian_smooth(imu[:, 2], sigma=5)
    gy = gaussian_smooth(imu[:, 4], sigma=3)

    # Thresholds
    VALLEY_MAX = 0.70       # Below this = moving (descent/ascent)
    PLATEAU_MIN = 0.72      # Stable region at top
    PLATEAU_MAX = 1.10
    PEAK_MIN = 1.12         # High acceleration at bottom
    STABLE_STD = 0.10       # Low variance = stable

    # Use a sliding window to classify each sample
    window_size = 40
    half_window = window_size // 2

    sample_labels = []

    for i in range(len(imu)):
        # Get local window around this sample
        start = max(0, i - half_window)
        end = min(len(imu), i + half_window + 1)

        local_az = az[start:end]
        local_gy = gy[start:end]

        az_mean = np.mean(local_az)
        az_std = np.std(local_az)
        gy_max = np.max(np.abs(local_gy))

        # Classify this sample
        if az_mean < VALLEY_MAX:
            label = "moving"
        elif az_mean >= PEAK_MIN:
            label = "at-bottom"
        elif (PLATEAU_MIN <= az_mean <= PLATEAU_MAX and
              az_std < STABLE_STD and
              gy_max < 15.0):
            label = "at-top"
        else:
            label = "moving"

        sample_labels.append(label)

    # Merge into contiguous regions
    regions = []
    current_label = sample_labels[0]
    start_idx = 0

    for i in range(1, len(sample_labels)):
        if sample_labels[i] != current_label:
            # Region boundary detected
            if i - start_idx >= min_region_size:
                regions.append((start_idx, i - 1, current_label))
            else:
                # Too small, mark as previous region or moving
                sample_labels[start_idx:i] = [current_label] * (i - start_idx)

            current_label = sample_labels[i]
            start_idx = i

    # Add final region
    if len(sample_labels) - start_idx >= min_region_size:
        regions.append((start_idx, len(sample_labels) - 1, current_label))

    return regions, sample_labels


def label_regions_with_state_machine(imu, min_stable_duration=15):
    """
    Label regions using a state machine with transition detection.

    This approach looks for clear transitions:
    1. Start at-top (plank position)
    2. Detect descent (moving down)
    3. Detect at-bottom (chest near ground)
    4. Detect ascent (moving up)
    5. Return to at-top

    Args:
        imu: (N, 6) numpy array containing [ax, ay, az, gx, gy, gz]
        min_stable_duration: Minimum samples to confirm stable state

    Returns:
        regions: List of (start_idx, end_idx, phase_label) tuples
        labels: Per-sample phase labels
    """
    az = gaussian_smooth(imu[:, 2], sigma=5)
    gy = gaussian_smooth(imu[:, 4], sigma=3)

    # Detection thresholds
    TOP_AZ_MIN = 0.72
    TOP_AZ_MAX = 1.10
    BOTTOM_AZ_MIN = 1.12
    MOVING_AZ_MAX = 0.70
    STABLE_STD = 0.10
    GYRO_MOVING = 20.0

    labels = []
    state = "at-top"  # Start in plank position
    state_duration = 0

    window_size = 30

    for i in range(len(imu)):
        # Get local statistics
        start = max(0, i - window_size // 2)
        end = min(len(imu), i + window_size // 2 + 1)

        local_az = az[start:end]
        local_gy = gy[start:end]

        az_mean = np.mean(local_az)
        az_std = np.std(local_az)
        gy_max = np.max(np.abs(local_gy))

        # State machine transitions
        if state == "at-top":
            # Check if starting to move down
            if az_mean < MOVING_AZ_MAX or gy_max > GYRO_MOVING:
                if state_duration >= min_stable_duration:
                    state = "moving"
                    state_duration = 0
            else:
                state_duration += 1

        elif state == "moving":
            # Check if reached bottom
            if az_mean >= BOTTOM_AZ_MIN and az_std < STABLE_STD:
                state = "at-bottom"
                state_duration = 0
            # Check if returned to top (without hitting bottom)
            elif (TOP_AZ_MIN <= az_mean <= TOP_AZ_MAX and
                  az_std < STABLE_STD and
                  gy_max < 15.0 and
                  state_duration >= min_stable_duration):
                state = "at-top"
                state_duration = 0
            else:
                state_duration += 1

        elif state == "at-bottom":
            # Check if starting to move up
            if az_mean < BOTTOM_AZ_MIN or gy_max > GYRO_MOVING:
                if state_duration >= min_stable_duration:
                    state = "moving"
                    state_duration = 0
            else:
                state_duration += 1

        labels.append(state)

    # Convert to regions
    regions = []
    current_label = labels[0]
    start_idx = 0

    for i in range(1, len(labels)):
        if labels[i] != current_label:
            regions.append((start_idx, i - 1, current_label))
            current_label = labels[i]
            start_idx = i

    # Add final region
    regions.append((start_idx, len(labels) - 1, current_label))

    return regions, labels


def count_pushups_from_regions(regions):
    """
    Count complete pushups from phase regions.

    A complete pushup cycle is:
    at-top → moving → at-bottom → moving → at-top

    Args:
        regions: List of (start_idx, end_idx, phase_label) tuples

    Returns:
        count: Number of complete pushups
        cycles: List of pushup cycle info [(start, end, phases), ...]
    """
    count = 0
    cycles = []

    # Extract just the phase labels
    phase_sequence = [r[2] for r in regions]

    i = 0
    while i < len(phase_sequence) - 4:
        # Look for the pattern: at-top, moving, at-bottom, moving, at-top
        if (phase_sequence[i] == "at-top" and
            phase_sequence[i+1] == "moving" and
            phase_sequence[i+2] == "at-bottom" and
            phase_sequence[i+3] == "moving" and
            phase_sequence[i+4] == "at-top"):

            # Found a complete cycle
            cycle_start = regions[i][0]
            cycle_end = regions[i+4][1]

            cycles.append({
                "start_idx": cycle_start,
                "end_idx": cycle_end,
                "duration": cycle_end - cycle_start + 1,
                "regions": regions[i:i+5]
            })

            count += 1
            i += 4  # Move to the start of next potential cycle
        else:
            i += 1

    return count, cycles


###############################################################################
# Visualization
###############################################################################

def plot_regions(imu, regions, labels=None, title="Phase Regions"):
    """
    Plot IMU data with phase regions highlighted.

    Args:
        imu: (N, 6) numpy array
        regions: List of (start, end, label) tuples
        labels: Optional per-sample labels
        title: Plot title
    """
    az = imu[:, 2]
    t = np.arange(len(az))

    colors = {
        "at-top": "blue",
        "moving": "orange",
        "at-bottom": "red"
    }

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10))

    # Top plot: Raw signal with region backgrounds
    ax1.plot(t, az, color='black', linewidth=1.5, label='az (raw)')

    # Shade regions
    for start, end, label in regions:
        ax1.axvspan(start, end, alpha=0.3, color=colors[label])

    # Add threshold lines
    ax1.axhline(y=0.70, color='orange', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(y=0.72, color='blue', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(y=1.10, color='blue', linestyle='--', linewidth=1, alpha=0.5)
    ax1.axhline(y=1.12, color='red', linestyle='--', linewidth=1, alpha=0.5)

    ax1.set_ylabel("Accelerometer Z (g)", fontsize=12)
    ax1.set_title(title, fontsize=14)
    ax1.grid(alpha=0.3)
    ax1.legend()

    # Bottom plot: State timeline
    if labels:
        state_numeric = []
        for label in labels:
            if label == "at-top":
                state_numeric.append(2)
            elif label == "at-bottom":
                state_numeric.append(0)
            else:  # moving
                state_numeric.append(1)

        ax2.fill_between(t, 0, state_numeric, step='post', alpha=0.6)
        ax2.set_yticks([0, 1, 2])
        ax2.set_yticklabels(['at-bottom', 'moving', 'at-top'])
        ax2.set_xlabel("Sample Index", fontsize=12)
        ax2.set_ylabel("Phase", fontsize=12)
        ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.show()


def print_region_stats(regions, sample_rate=40):
    """Print statistics about labeled regions."""
    print("\nRegion Statistics:")
    print("=" * 70)

    phase_durations = {"at-top": [], "moving": [], "at-bottom": []}

    for start, end, label in regions:
        duration = end - start + 1
        phase_durations[label].append(duration)

    for phase in ["at-top", "moving", "at-bottom"]:
        durations = phase_durations[phase]
        if durations:
            count = len(durations)
            total_samples = sum(durations)
            avg_duration = np.mean(durations)
            avg_time = avg_duration / sample_rate

            print(f"\n{phase}:")
            print(f"  Count: {count} regions")
            print(f"  Total samples: {total_samples}")
            print(f"  Avg duration: {avg_duration:.1f} samples ({avg_time:.2f}s)")
            print(f"  Min/Max: {min(durations)}/{max(durations)} samples")
        else:
            print(f"\n{phase}: No regions detected")


def process_all_to_regions(raw_dataset_dir: str, method="simple"):
    """
    Process all JSON files and create region-based labels.

    Args:
        raw_dataset_dir: Path to directory containing raw JSON files
        method: "simple" or "state_machine"

    Returns:
        labeled_data: Dictionary with metadata and labeled sessions
    """
    all_sessions_data = load_all_json_files(raw_dataset_dir)
    all_labeled_sessions = []

    total_pushups = 0
    phase_sample_counts = {"at-top": 0, "moving": 0, "at-bottom": 0}

    print(f"Processing {len(all_sessions_data)} files with {method} method...")
    print()

    for filename, session_data in all_sessions_data:
        file_pushups = 0

        for session in session_data["sessions"]:
            data = session["data"]

            # Extract IMU data
            ax = [d["ax"] for d in data]
            ay = [d["ay"] for d in data]
            az = [d["az"] for d in data]
            gx = [d["gx"] for d in data]
            gy = [d["gy"] for d in data]
            gz = [d["gz"] for d in data]

            imu = np.stack([ax, ay, az, gx, gy, gz], axis=1)

            # Label regions
            if method == "state_machine":
                regions, labels = label_regions_with_state_machine(imu)
            else:
                regions, labels = label_regions_simple(imu)

            # Count pushups
            pushup_count, cycles = count_pushups_from_regions(regions)
            total_pushups += pushup_count
            file_pushups += pushup_count

            # Count samples per phase
            for label in labels:
                phase_sample_counts[label] += 1

            # Store labeled session
            labeled_session = {
                "session_id": session["session_id"],
                "posture_label": session.get("posture_label", "unknown"),
                "regions": [(int(s), int(e), l) for s, e, l in regions],
                "pushup_count": pushup_count,
                "cycles": cycles,
                "total_samples": len(labels)
            }

            all_labeled_sessions.append(labeled_session)

        print(f"  {filename}: {file_pushups} pushups detected")

    # Create output structure
    labeled_data = {
        "metadata": {
            "labeling_method": method,
            "total_sessions": len(all_labeled_sessions),
            "total_pushups": total_pushups,
            "phase_sample_counts": phase_sample_counts,
            "phase_labels": ["at-top", "moving", "at-bottom"]
        },
        "sessions": all_labeled_sessions
    }

    return labeled_data


###############################################################################
# Main
###############################################################################

if __name__ == "__main__":
    import sys

    RAW_DATASET_DIR = "raw_dataset"
    OUTPUT_FILE = "pushup_data_regions.json"

    # Usage:
    # python3 region_label_phases.py --visualize [session_id] [method]
    # python3 region_label_phases.py [method]
    # method = "simple" or "state_machine" (default: simple)

    method = "simple"

    if len(sys.argv) > 1 and sys.argv[1] == "--visualize":
        # Visualization mode
        json_path = "raw_dataset/pushup_data_20251204_181709.json"
        SESSION_TO_ANALYZE = 0

        if len(sys.argv) > 2:
            try:
                SESSION_TO_ANALYZE = int(sys.argv[2])
            except ValueError:
                method = sys.argv[2]

        if len(sys.argv) > 3:
            method = sys.argv[3]

        print(f"Loading session {SESSION_TO_ANALYZE}...")
        print(f"Using {method} method\n")

        # Load session
        with open(json_path, "r") as f:
            raw = json.load(f)

        session = raw["sessions"][SESSION_TO_ANALYZE]
        data = session["data"]

        ax = [d["ax"] for d in data]
        ay = [d["ay"] for d in data]
        az = [d["az"] for d in data]
        gx = [d["gx"] for d in data]
        gy = [d["gy"] for d in data]
        gz = [d["gz"] for d in data]

        imu = np.stack([ax, ay, az, gx, gy, gz], axis=1)

        # Label regions
        if method == "state_machine":
            regions, labels = label_regions_with_state_machine(imu)
        else:
            regions, labels = label_regions_simple(imu)

        # Count pushups
        pushup_count, cycles = count_pushups_from_regions(regions)

        print(f"Detected {pushup_count} complete pushups")
        print(f"Found {len(regions)} regions")

        print_region_stats(regions)

        if cycles:
            print(f"\nPushup Cycles:")
            for i, cycle in enumerate(cycles, 1):
                print(f"  Cycle {i}: samples {cycle['start_idx']}-{cycle['end_idx']} ({cycle['duration']} samples)")

        plot_regions(imu, regions, labels,
                    title=f"Session {SESSION_TO_ANALYZE} - Region Labels ({method})")

    else:
        # Full processing mode
        if len(sys.argv) > 1:
            method = sys.argv[1]

        print("=" * 70)
        print(f"Region-based phase labeling ({method} method)")
        print("=" * 70)

        labeled_data = process_all_to_regions(RAW_DATASET_DIR, method=method)

        # Save to JSON
        print(f"\nSaving to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, "w") as f:
            json.dump(labeled_data, f, indent=2)

        print("\n" + "=" * 70)
        print("Processing Complete!")
        print("=" * 70)
        print(f"\nTotal sessions: {labeled_data['metadata']['total_sessions']}")
        print(f"Total pushups detected: {labeled_data['metadata']['total_pushups']}")

        print(f"\nPhase Distribution (by samples):")
        total_samples = sum(labeled_data['metadata']['phase_sample_counts'].values())
        for phase in ["at-top", "moving", "at-bottom"]:
            count = labeled_data['metadata']['phase_sample_counts'][phase]
            pct = 100 * count / total_samples if total_samples > 0 else 0
            print(f"  {phase:12s}: {count:6d} ({pct:5.1f}%)")

        print(f"\nOutput saved to: {OUTPUT_FILE}")
