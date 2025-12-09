import json
import numpy as np
import os
import glob
import matplotlib.pyplot as plt


###############################################################################
# Utility Functions
###############################################################################
def smooth(x, k=5):
    """
    Smooth a signal using a simple moving average.

    Args:
        x: Input signal array
        k: Kernel size for smoothing (default: 5)

    Returns:
        Smoothed signal array
    """
    if len(x) < k:
        return x
    kernel = np.ones(k) / k
    return np.convolve(x, kernel, mode="same")


def gaussian_smooth(x, sigma=2):
    """
    Smooth a signal using Gaussian filter.

    Args:
        x: Input signal array
        sigma: Standard deviation of Gaussian kernel

    Returns:
        Smoothed signal array
    """
    if len(x) < 3:
        return x

    # Create Gaussian kernel
    kernel_size = int(4 * sigma + 1)
    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel_half = kernel_size // 2
    kernel = np.exp(-np.arange(-kernel_half, kernel_half + 1)**2 / (2 * sigma**2))
    kernel = kernel / np.sum(kernel)

    return np.convolve(x, kernel, mode="same")


###############################################################################
# Phase Labeling
###############################################################################
def auto_label_phase(window: np.ndarray, debug=False) -> str:
    """
    Automatically label pushup phase based on IMU data.

    Three phase classification: at-top, moving, at-bottom

    IMPORTANT: The positive Z-axis of the accelerometer points DOWN toward
    the ground (placed on sternum pointing toward floor).

    Based on Z-position integration analysis:
    - at-top: Z-position is at zero (sternum highest, plank position)
              → Z-acceleration is at plateau (around 0.8-1.0g)
              → Low variance in az, low gyro activity
    - at-bottom: Z-position is at minimum (sternum lowest, chest near ground)
                 → Z-acceleration is at peak (around 1.3-1.5g)
                 → Momentary high acceleration, low gyro activity
    - moving: Z-position is changing (descent or ascent)
              → Z-acceleration is at valley (around 0.4-0.7g)
              → High variance, high gyro activity, or low az values

    Args:
        window: (N, 6) numpy array containing [ax, ay, az, gx, gy, gz]

    Returns:
        Phase label: "at-top", "moving", or "at-bottom"
    """
    window = np.asarray(window)

    # Apply stronger smoothing for better plateau/peak/valley detection
    az = gaussian_smooth(window[:, 2], sigma=5)  # Z-acceleration with Gaussian smoothing (increased from 3)
    gy = gaussian_smooth(window[:, 4], sigma=3)  # Y-gyroscope (increased from 2)

    # Calculate statistical features
    az_mean = float(np.mean(az))
    az_median = float(np.median(az))
    az_std = float(np.std(az))
    az_min = float(np.min(az))
    az_max = float(np.max(az))

    gy_std = float(np.std(gy))
    gy_max_abs = float(np.max(np.abs(gy)))

    # Thresholds based on empirical analysis of Z-position graphs
    # These are tuned to match the plateau/valley/peak patterns

    # Acceleration ranges (more refined)
    VALLEY_MAX = 0.70           # Moving phase shows valleys below this
    PLATEAU_MIN = 0.72          # Top plateau starts here
    PLATEAU_MAX = 1.10          # Top plateau ends here
    PEAK_MIN = 1.12             # Bottom peak starts here

    # Stability thresholds
    STABLE_STD_MAX = 0.10       # Low variance indicates stable position
    HIGH_VARIANCE_MIN = 0.15    # High variance indicates movement
    STATIC_GY_MAX = 15.0        # Low gyro = static position
    MOVING_GY_MIN = 20.0        # High gyro = definitely moving

    # Decision logic based on signal characteristics

    # RULE 1: Clear valley in acceleration = moving
    if az_mean < VALLEY_MAX or az_median < VALLEY_MAX:
        if debug:
            print(f"  RULE 1 (Valley): az_mean={az_mean:.3f}, az_median={az_median:.3f} < {VALLEY_MAX} → moving")
        return "moving"

    # RULE 2: Clear peak in acceleration = at-bottom
    if az_mean >= PEAK_MIN and az_median >= PEAK_MIN:
        if debug:
            print(f"  RULE 2 (Peak): az_mean={az_mean:.3f}, az_median={az_median:.3f} >= {PEAK_MIN} → at-bottom")
        return "at-bottom"

    # RULE 3: High gyro activity = moving (regardless of acceleration)
    if gy_max_abs > MOVING_GY_MIN or gy_std > 10.0:
        if debug:
            print(f"  RULE 3 (High Gyro): gy_max={gy_max_abs:.1f}, gy_std={gy_std:.1f} → moving")
        return "moving"

    # RULE 4: Stable plateau with low variance = at-top
    if (PLATEAU_MIN <= az_mean <= PLATEAU_MAX and
        PLATEAU_MIN <= az_median <= PLATEAU_MAX and
        az_std < STABLE_STD_MAX and
        gy_max_abs < STATIC_GY_MAX):
        if debug:
            print(f"  RULE 4 (Plateau): az_mean={az_mean:.3f}, az_std={az_std:.3f}, gy_max={gy_max_abs:.1f} → at-top")
        return "at-top"

    # RULE 5: High variance indicates movement
    if az_std > HIGH_VARIANCE_MIN:
        if debug:
            print(f"  RULE 5 (High Variance): az_std={az_std:.3f} > {HIGH_VARIANCE_MIN} → moving")
        return "moving"

    # RULE 6: Use mean to decide between remaining cases
    if az_mean >= PEAK_MIN:
        if debug:
            print(f"  RULE 6 (Fallback Peak): az_mean={az_mean:.3f} >= {PEAK_MIN} → at-bottom")
        return "at-bottom"
    elif PLATEAU_MIN <= az_mean <= PLATEAU_MAX:
        if debug:
            print(f"  RULE 6 (Fallback Plateau): az_mean={az_mean:.3f} in [{PLATEAU_MIN}, {PLATEAU_MAX}] → at-top")
        return "at-top"
    else:
        # Default to moving for ambiguous cases
        if debug:
            print(f"  RULE 6 (Fallback Default): az_mean={az_mean:.3f} → moving")
        return "moving"


###############################################################################
# Data Loading Functions
###############################################################################
def load_true_json_format(json_path: str):
    """
    Load and flatten all sessions from JSON file.

    Expected JSON format:
    {
      "metadata": {...},
      "sessions": [
         {
           "session_id": ...,
           "data": [
             {"ax":..., "ay":..., "az":..., "gx":..., "gy":..., "gz":...},
             ...
           ]
         },
         ...
      ]
    }

    Args:
        json_path: Path to JSON file

    Returns:
        all_imu: Flattened (N, 6) numpy array of IMU data
        session_ranges: List of (start, end) tuples for each session
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    all_imu = []
    session_ranges = []

    for session in raw["sessions"]:
        data = session["data"]

        # Extract each axis
        ax = [d["ax"] for d in data]
        ay = [d["ay"] for d in data]
        az = [d["az"] for d in data]
        gx = [d["gx"] for d in data]
        gy = [d["gy"] for d in data]
        gz = [d["gz"] for d in data]

        imu = np.stack([ax, ay, az, gx, gy, gz], axis=1)

        start = len(all_imu)
        end = start + len(imu)
        session_ranges.append((start, end))

        all_imu.extend(imu)

    return np.array(all_imu), session_ranges


def load_single_session(json_path: str, session_id: int):
    """
    Load a specific session from JSON file.

    Args:
        json_path: Path to JSON file
        session_id: ID of the session to load

    Returns:
        imu: (N, 6) numpy array containing [ax, ay, az, gx, gy, gz]

    Raises:
        ValueError: If session_id is not found
    """
    with open(json_path, "r") as f:
        raw = json.load(f)

    # Find the correct session
    for session in raw["sessions"]:
        if session["session_id"] == session_id:
            data = session["data"]

            ax = [d["ax"] for d in data]
            ay = [d["ay"] for d in data]
            az = [d["az"] for d in data]
            gx = [d["gx"] for d in data]
            gy = [d["gy"] for d in data]
            gz = [d["gz"] for d in data]

            imu = np.stack([ax, ay, az, gx, gy, gz], axis=1)
            return imu

    raise ValueError(f"Session ID {session_id} not found.")


def load_all_json_files(raw_dataset_dir: str):
    """
    Load all JSON files from the raw dataset directory.

    Args:
        raw_dataset_dir: Path to directory containing JSON files

    Returns:
        all_sessions: List of tuples (filename, session_data_dict)
    """
    json_files = sorted(glob.glob(os.path.join(raw_dataset_dir, "*.json")))
    all_sessions = []

    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
            filename = os.path.basename(json_file)
            all_sessions.append((filename, data))

    return all_sessions


###############################################################################
# Processing Functions
###############################################################################
def process_session(json_path: str, window_size=40, step_size=20):
    """
    Process all sessions using sliding window approach.

    Args:
        json_path: Path to JSON file
        window_size: Size of sliding window (default: 40)
        step_size: Step size for sliding window (default: 20)

    Returns:
        imu: Full IMU data array
        centers: List of window center indices
        labels: List of phase labels
    """
    imu, _ = load_true_json_format(json_path)

    labels = []
    centers = []

    i = 0
    while i + window_size <= len(imu):
        window = imu[i:i + window_size]
        phase = auto_label_phase(window)
        labels.append(phase)
        centers.append(i + window_size // 2)
        i += step_size

    return imu, centers, labels


def process_one_session(imu, window_size=40, step_size=20):
    """
    Process a single session using sliding window approach.

    Args:
        imu: (N, 6) numpy array of IMU data
        window_size: Size of sliding window (default: 40)
        step_size: Step size for sliding window (default: 20)

    Returns:
        centers: List of window center indices
        labels: List of phase labels
    """
    labels = []
    centers = []

    i = 0
    while i + window_size <= len(imu):
        window = imu[i:i + window_size]
        phase = auto_label_phase(window)
        labels.append(phase)
        centers.append(i + window_size // 2)
        i += step_size

    return centers, labels


def process_all_files_to_windows(raw_dataset_dir: str, window_size=40, step_size=20):
    """
    Process all JSON files and create labeled windows.

    Args:
        raw_dataset_dir: Path to directory containing raw JSON files
        window_size: Size of sliding window (default: 40)
        step_size: Step size for sliding window (default: 20)

    Returns:
        labeled_data: Dictionary containing metadata and labeled windows
    """
    all_sessions = load_all_json_files(raw_dataset_dir)
    all_windows = []
    phase_counts = {"at-top": 0, "moving": 0, "at-bottom": 0}
    posture_counts = {}
    file_stats = []

    print(f"Processing {len(all_sessions)} JSON files...")
    print()

    total_sessions = 0
    for filename, session_data in all_sessions:
        file_phase_counts = {"at-top": 0, "moving": 0, "at-bottom": 0}
        file_windows = 0

        for session in session_data["sessions"]:
            total_sessions += 1
            data = session["data"]

            # Extract IMU data
            ax = [d["ax"] for d in data]
            ay = [d["ay"] for d in data]
            az = [d["az"] for d in data]
            gx = [d["gx"] for d in data]
            gy = [d["gy"] for d in data]
            gz = [d["gz"] for d in data]

            imu = np.stack([ax, ay, az, gx, gy, gz], axis=1)

            # Get original posture label from session
            posture_label = session.get("posture_label", "unknown")

            # Count postures
            if posture_label not in posture_counts:
                posture_counts[posture_label] = 0

            # Create sliding windows
            i = 0
            while i + window_size <= len(imu):
                window = imu[i:i + window_size]
                phase_label = auto_label_phase(window)

                window_dict = {
                    "window": window.tolist(),
                    "posture_label": posture_label,
                    "phase_label": phase_label
                }

                all_windows.append(window_dict)
                phase_counts[phase_label] += 1
                posture_counts[posture_label] += 1
                file_phase_counts[phase_label] += 1
                file_windows += 1

                i += step_size

        # Print file statistics
        print(f"  {filename}:")
        print(f"    Windows: {file_windows}")
        print(f"    at-top: {file_phase_counts['at-top']} ({100*file_phase_counts['at-top']/max(file_windows,1):.1f}%)")
        print(f"    moving: {file_phase_counts['moving']} ({100*file_phase_counts['moving']/max(file_windows,1):.1f}%)")
        print(f"    at-bottom: {file_phase_counts['at-bottom']} ({100*file_phase_counts['at-bottom']/max(file_windows,1):.1f}%)")
        print()

        file_stats.append({
            "filename": filename,
            "windows": file_windows,
            "phase_distribution": file_phase_counts
        })

    # Create output structure
    labeled_data = {
        "metadata": {
            "window_size": window_size,
            "stride": step_size,
            "total_windows": len(all_windows),
            "total_sessions": total_sessions,
            "total_files": len(all_sessions),
            "phase_counts": phase_counts,
            "posture_counts": posture_counts,
            "phase_labels": ["at-top", "moving", "at-bottom"],
            "posture_labels": list(posture_counts.keys()),
            "file_statistics": file_stats
        },
        "windows": all_windows
    }

    return labeled_data


###############################################################################
# Visualization Functions
###############################################################################
def plot_session_with_labels(imu, centers, labels):
    """
    Plot IMU data with phase labels overlay (all sessions).

    Args:
        imu: (N, 6) numpy array of IMU data
        centers: List of window center indices
        labels: List of phase labels
    """
    az = imu[:, 2]
    t = np.arange(len(az))

    colors = {
        "at-top": "blue",
        "moving": "orange",
        "at-bottom": "red"
    }

    plt.figure(figsize=(14, 6))
    plt.plot(t, az, label="az", color="black")

    px = []
    py = []
    pc = []
    for c, lab in zip(centers, labels):
        if c < len(az):
            px.append(c)
            py.append(az[c])
            pc.append(colors[lab])

    plt.scatter(px, py, c=pc, s=40)

    for lab, col in colors.items():
        plt.scatter([], [], c=col, s=40, label=lab)

    plt.xlabel("Sample index")
    plt.ylabel("az (accel Z)")
    plt.title("Pushup Phase Labels Overlay")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()


def plot_single_session(imu, centers, labels, show_smoothed=True, show_thresholds=True):
    """
    Plot IMU data with phase labels overlay (single session).

    Args:
        imu: (N, 6) numpy array of IMU data
        centers: List of window center indices
        labels: List of phase labels
        show_smoothed: Whether to show smoothed signal overlay
        show_thresholds: Whether to show classification threshold lines
    """
    az_raw = imu[:, 2]
    t = np.arange(len(az_raw))

    # Apply the same smoothing used in classification
    az_smoothed = gaussian_smooth(az_raw, sigma=5)

    colors = {
        "at-top": "blue",
        "moving": "orange",
        "at-bottom": "red"
    }

    # Thresholds from auto_label_phase function
    VALLEY_MAX = 0.70
    PLATEAU_MIN = 0.72
    PLATEAU_MAX = 1.10
    PEAK_MIN = 1.12

    plt.figure(figsize=(16, 8))

    # Plot raw signal
    plt.plot(t, az_raw, label="az (raw)", color="lightgray", linewidth=1, alpha=0.7)

    # Plot smoothed signal
    if show_smoothed:
        plt.plot(t, az_smoothed, label="az (smoothed)", color="black", linewidth=2)

    # Plot threshold lines
    if show_thresholds:
        plt.axhline(y=VALLEY_MAX, color='orange', linestyle='--', linewidth=1.5,
                   alpha=0.6, label=f'Valley Max ({VALLEY_MAX}g) - Moving')
        plt.axhline(y=PLATEAU_MIN, color='blue', linestyle='--', linewidth=1.5,
                   alpha=0.6, label=f'Plateau Range ({PLATEAU_MIN}-{PLATEAU_MAX}g) - At Top')
        plt.axhline(y=PLATEAU_MAX, color='blue', linestyle='--', linewidth=1.5, alpha=0.6)
        plt.axhline(y=PEAK_MIN, color='red', linestyle='--', linewidth=1.5,
                   alpha=0.6, label=f'Peak Min ({PEAK_MIN}g) - At Bottom')

    # Plot classification labels
    px, py, col = [], [], []
    for c, lab in zip(centers, labels):
        if c < len(az_smoothed):
            px.append(c)
            py.append(az_smoothed[c])  # Use smoothed value for label placement
            col.append(colors[lab])

    plt.scatter(px, py, c=col, s=60, edgecolors='black', linewidth=1.5, zorder=5)

    # Add legend for labels
    for lab, c in colors.items():
        plt.scatter([], [], c=c, s=60, edgecolors='black', linewidth=1.5, label=lab)

    plt.title("Phase Labels for Selected Session (with Smoothing & Thresholds)", fontsize=14)
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("Accelerometer Z (g)", fontsize=12)
    plt.grid(alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"\nSignal Statistics:")
    print(f"  Raw az:      min={az_raw.min():.3f}, max={az_raw.max():.3f}, mean={az_raw.mean():.3f}, std={az_raw.std():.3f}")
    print(f"  Smoothed az: min={az_smoothed.min():.3f}, max={az_smoothed.max():.3f}, mean={az_smoothed.mean():.3f}, std={az_smoothed.std():.3f}")
    print(f"\nLabel Distribution:")
    label_counts = {lab: labels.count(lab) for lab in set(labels)}
    total = len(labels)
    for lab in ["at-top", "moving", "at-bottom"]:
        count = label_counts.get(lab, 0)
        print(f"  {lab:12s}: {count:3d} ({100*count/total:5.1f}%)")


###############################################################################
# Main
###############################################################################
if __name__ == "__main__":
    import sys

    # Configuration
    RAW_DATASET_DIR = "raw_dataset"
    OUTPUT_FILE = "pushup_data_phase_labeled.json"
    WINDOW_SIZE = 60  # Increased from 40 (1.5 seconds @ 40Hz)
    STEP_SIZE = 20    # Keep overlap for better coverage

    # Usage:
    # 1. Visualize a single session: python3 auto_label_phases.py --visualize [session_id]
    # 2. Visualize with debug info: python3 auto_label_phases.py --visualize [session_id] --debug
    # 3. Process all files: python3 auto_label_phases.py

    # Check if user wants to visualize a single session
    if len(sys.argv) > 1 and sys.argv[1] == "--visualize":
        # Visualization mode
        json_path = "raw_dataset/pushup_data_20251204_181709.json"
        SESSION_TO_ANALYZE = 0
        DEBUG_MODE = False

        # Parse arguments
        for i, arg in enumerate(sys.argv[2:], start=2):
            if arg == "--debug":
                DEBUG_MODE = True
            else:
                try:
                    SESSION_TO_ANALYZE = int(arg)
                except ValueError:
                    pass

        print(f"Loading session {SESSION_TO_ANALYZE} for visualization...")
        print(f"Window size: {WINDOW_SIZE}, Step size: {STEP_SIZE}")
        print(f"Debug mode: {DEBUG_MODE}\n")

        imu = load_single_session(json_path, SESSION_TO_ANALYZE)

        # Process with optional debug output
        if DEBUG_MODE:
            print("Classification debug output:")
            labels = []
            centers = []
            i = 0
            window_num = 0
            while i + WINDOW_SIZE <= len(imu):
                window = imu[i:i + WINDOW_SIZE]
                center = i + WINDOW_SIZE // 2
                print(f"\nWindow {window_num} (center={center}):")
                phase = auto_label_phase(window, debug=True)
                labels.append(phase)
                centers.append(center)
                i += STEP_SIZE
                window_num += 1
        else:
            centers, labels = process_one_session(imu, WINDOW_SIZE, STEP_SIZE)

        plot_single_session(imu, centers, labels)
        print(f"\nFinished labeling session {SESSION_TO_ANALYZE}.")
        print(f"To see classification details, run with: --visualize {SESSION_TO_ANALYZE} --debug")

    else:
        # Full processing mode
        print("=" * 70)
        print("Auto-labeling all pushup data with phase labels")
        print("=" * 70)
        print(f"Raw dataset directory: {RAW_DATASET_DIR}")
        print(f"Output file: {OUTPUT_FILE}")
        print(f"Window size: {WINDOW_SIZE}")
        print(f"Step size: {STEP_SIZE}")
        print()

        # Process all files
        labeled_data = process_all_files_to_windows(
            RAW_DATASET_DIR,
            window_size=WINDOW_SIZE,
            step_size=STEP_SIZE
        )

        # Save to JSON
        print(f"\nSaving labeled data to {OUTPUT_FILE}...")
        with open(OUTPUT_FILE, "w") as f:
            json.dump(labeled_data, f, indent=2)

        print("=" * 70)
        print("Processing complete!")
        print("=" * 70)
        print(f"\nDataset Statistics:")
        print(f"  Total files processed: {labeled_data['metadata']['total_files']}")
        print(f"  Total sessions: {labeled_data['metadata']['total_sessions']}")
        print(f"  Total windows created: {labeled_data['metadata']['total_windows']}")
        print(f"  Window size: {labeled_data['metadata']['window_size']}")
        print(f"  Stride: {labeled_data['metadata']['stride']}")

        print(f"\nPhase Label Distribution:")
        total_windows = labeled_data['metadata']['total_windows']
        for phase in ["at-top", "moving", "at-bottom"]:
            count = labeled_data['metadata']['phase_counts'][phase]
            percentage = (count / total_windows) * 100
            bar_length = int(percentage / 2)
            bar = "█" * bar_length
            print(f"  {phase:12s}: {count:5d} ({percentage:5.1f}%) {bar}")

        print(f"\nPosture Label Distribution:")
        for posture, count in sorted(labeled_data['metadata']['posture_counts'].items()):
            percentage = (count / total_windows) * 100
            bar_length = int(percentage / 2)
            bar = "█" * bar_length
            print(f"  {posture:12s}: {count:5d} ({percentage:5.1f}%) {bar}")

        print(f"\nOutput saved to: {OUTPUT_FILE}")
