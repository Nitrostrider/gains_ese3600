"""
Pushup IMU Data Collection GUI
Connects to XIAO ESP32S3 via BLE to collect accelerometer and gyroscope data
for pushup phase and posture classification.

Based on MagicWand data collector template.
"""

import asyncio
import json
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from datetime import datetime
import struct
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np

try:
    from bleak import BleakClient, BleakScanner
except ImportError:
    print("ERROR: bleak library not found. Install with: pip install bleak")
    exit(1)

# BLE Configuration (matching MW_DataCollection firmware)
DEVICE_NAME_PREFIX = "GAINS-Pushup"  # Device advertises as this name
SERVICE_UUID = "0000ff00-0000-1000-8000-00805f9b34fb"
IMU_CHAR_UUID = "0000ff01-0000-1000-8000-00805f9b34fb"
CTRL_CHAR_UUID = "0000ff02-0000-1000-8000-00805f9b34fb"

# Data collection settings
SAMPLE_RATE = 40  # Hz (25ms period from firmware)
BUFFER_SIZE = 200  # Keep last 5 seconds of data for visualization




class PushupDataCollector:
    def __init__(self, root):
        self.root = root
        self.root.title("Pushup IMU Data Collector")
        self.root.geometry("1000x800")

        # BLE state
        self.client = None
        self.device = None
        self.is_connected = False

        # Recording state
        self.is_recording = False
        self.current_session = []
        self.all_sessions = []
        self.session_metadata = {}
        self.recording_start_time = None

        # Real-time data buffers for visualization
        self.accel_buffer = {'x': deque(maxlen=BUFFER_SIZE),
                            'y': deque(maxlen=BUFFER_SIZE),
                            'z': deque(maxlen=BUFFER_SIZE)}
        self.gyro_buffer = {'x': deque(maxlen=BUFFER_SIZE),
                           'y': deque(maxlen=BUFFER_SIZE),
                           'z': deque(maxlen=BUFFER_SIZE)}
        self.time_buffer = deque(maxlen=BUFFER_SIZE)
        self.start_time = None

        # Create UI
        self.create_ui()

        # Asyncio loop for BLE
        self.loop = asyncio.new_event_loop()

    def create_ui(self):
        """Create the main UI layout"""

        # ============ Connection Frame ============
        conn_frame = ttk.LabelFrame(self.root, text="1. Device Connection", padding=10)
        conn_frame.grid(row=0, column=0, columnspan=2, sticky='ew', padx=10, pady=5)

        ttk.Button(conn_frame, text="Connect to Device",
                  command=self.connect_device).grid(row=0, column=0, padx=5)
        ttk.Button(conn_frame, text="Disconnect",
                  command=self.disconnect_device).grid(row=0, column=1, padx=5)

        self.status_label = ttk.Label(conn_frame, text="⚫ Not Connected",
                                     foreground="red", font=('Arial', 10, 'bold'))
        self.status_label.grid(row=0, column=2, padx=20)

        # Debug/log area
        self.log_text = tk.Text(conn_frame, height=4, width=80, font=('Courier', 9))
        self.log_text.grid(row=1, column=0, columnspan=4, pady=5, sticky='ew')
        scrollbar = ttk.Scrollbar(conn_frame, command=self.log_text.yview)
        scrollbar.grid(row=1, column=4, sticky='ns')
        self.log_text.config(yscrollcommand=scrollbar.set)

        # ============ Metadata Frame ============
        meta_frame = ttk.LabelFrame(self.root, text="Session Metadata", padding=10)
        meta_frame.grid(row=1, column=0, sticky='nsew', padx=10, pady=5)

        ttk.Label(meta_frame, text="Participant ID:").grid(row=0, column=0, sticky='w', pady=2)
        self.participant_entry = ttk.Entry(meta_frame, width=20)
        self.participant_entry.grid(row=0, column=1, sticky='w', pady=2)

        ttk.Label(meta_frame, text="IMU Placement:").grid(row=1, column=0, sticky='w', pady=2)
        self.placement_combo = ttk.Combobox(meta_frame, width=18,
            values=["Upper Back", "Sternum", "Forearm", "Wrist", "Lower Back", "Hip"])
        self.placement_combo.set("Upper Back")
        self.placement_combo.grid(row=1, column=1, sticky='w', pady=2)

        ttk.Label(meta_frame, text="Notes:").grid(row=2, column=0, sticky='nw', pady=2)
        self.notes_text = tk.Text(meta_frame, height=3, width=20, font=('Arial', 9))
        self.notes_text.grid(row=2, column=1, sticky='w', pady=2)

        # ============ Recording Frame ============
        rec_frame = ttk.LabelFrame(self.root, text="2. Record Training Data", padding=10)
        rec_frame.grid(row=1, column=1, sticky='nsew', padx=10, pady=5)

        # Phase selection
        ttk.Label(rec_frame, text="Phase Label:", font=('Arial', 9, 'bold')).grid(
            row=0, column=0, sticky='w', pady=2)
        self.phase_var = tk.StringVar(value="not-in-pushup")
        phases = ["top", "moving-down", "bottom", "moving-up", "not-in-pushup"]
        for i, phase in enumerate(phases):
            ttk.Radiobutton(rec_frame, text=phase, variable=self.phase_var,
                          value=phase).grid(row=i+1, column=0, sticky='w', padx=10)

        # Posture selection
        ttk.Label(rec_frame, text="Posture Label:", font=('Arial', 9, 'bold')).grid(
            row=0, column=1, sticky='w', pady=2, padx=20)
        self.posture_var = tk.StringVar(value="good-form")
        postures = ["good-form", "hips-sagging", "hips-high", "partial-rom"]
        for i, posture in enumerate(postures):
            ttk.Radiobutton(rec_frame, text=posture, variable=self.posture_var,
                          value=posture).grid(row=i+1, column=1, sticky='w', padx=30)

        # Recording controls
        control_frame = ttk.Frame(rec_frame)
        control_frame.grid(row=6, column=0, columnspan=2, pady=10)

        self.record_btn = ttk.Button(control_frame, text="▶ Start Recording",
                                     command=self.start_recording, style='Green.TButton')
        self.record_btn.grid(row=0, column=0, padx=5)

        self.stop_btn = ttk.Button(control_frame, text="⏹ Stop Recording",
                                   command=self.stop_recording, state='disabled')
        self.stop_btn.grid(row=0, column=1, padx=5)

        self.record_status = ttk.Label(control_frame, text="Idle",
                                      font=('Arial', 10, 'bold'))
        self.record_status.grid(row=1, column=0, columnspan=2, pady=5)

        # Timer display
        self.timer_label = ttk.Label(control_frame, text="Duration: 0.0s",
                                     font=('Courier', 10))
        self.timer_label.grid(row=2, column=0, columnspan=2, pady=2)

        # ============ Visualization Frame ============
        viz_frame = ttk.LabelFrame(self.root, text="Live IMU Data", padding=10)
        viz_frame.grid(row=2, column=0, columnspan=2, sticky='nsew', padx=10, pady=5)

        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 4), dpi=80)
        self.ax_accel = self.fig.add_subplot(121)
        self.ax_gyro = self.fig.add_subplot(122)

        self.ax_accel.set_title('Accelerometer (g)')
        self.ax_accel.set_xlabel('Time (s)')
        self.ax_accel.set_ylim(-4, 4)
        self.ax_accel.grid(True, alpha=0.3)

        self.ax_gyro.set_title('Gyroscope (°/s)')
        self.ax_gyro.set_xlabel('Time (s)')
        self.ax_gyro.set_ylim(-500, 500)
        self.ax_gyro.grid(True, alpha=0.3)

        self.canvas = FigureCanvasTkAgg(self.fig, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ============ Summary Frame ============
        summary_frame = ttk.LabelFrame(self.root, text="3. Summary & Export", padding=10)
        summary_frame.grid(row=3, column=0, columnspan=2, sticky='ew', padx=10, pady=5)

        self.summary_label = ttk.Label(summary_frame, text="0 sessions recorded",
                                      font=('Arial', 10, 'bold'))
        self.summary_label.grid(row=0, column=0, padx=10, pady=5)

        ttk.Button(summary_frame, text="📥 Export to JSON",
                  command=self.export_data).grid(row=0, column=1, padx=5)
        ttk.Button(summary_frame, text="🗑 Clear All",
                  command=self.clear_all).grid(row=0, column=2, padx=5)

        # Session list
        list_frame = ttk.Frame(summary_frame)
        list_frame.grid(row=1, column=0, columnspan=3, sticky='ew', pady=5)

        self.session_listbox = tk.Listbox(list_frame, height=6, font=('Courier', 9))
        self.session_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        list_scroll = ttk.Scrollbar(list_frame, command=self.session_listbox.yview)
        list_scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.session_listbox.config(yscrollcommand=list_scroll.set)

        # Configure grid weights for resizing
        self.root.grid_rowconfigure(2, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        # Style configuration
        style = ttk.Style()
        style.configure('Green.TButton', foreground='green')

    def log(self, message):
        """Add message to log window"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()

    def connect_device(self):
        """Scan for and connect to BLE device"""
        self.log("Scanning for BLE devices...")
        asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self._connect_async())

    async def _connect_async(self):
        """Async BLE connection - automatically connects to GAINS-Pushup device"""
        try:
            self.log(f"Scanning for '{DEVICE_NAME_PREFIX}' device...")
            devices = await BleakScanner.discover(timeout=5.0)

            # Filter for our device name
            target = None
            for device in devices:
                if device.name and device.name.startswith(DEVICE_NAME_PREFIX):
                    target = device
                    break

            if not target:
                self.log(f"ERROR: No device found with name '{DEVICE_NAME_PREFIX}'")
                messagebox.showerror("Device Not Found",
                    f"Could not find '{DEVICE_NAME_PREFIX}' device.\n\n" +
                    "Make sure:\n" +
                    "- Your device is powered on\n" +
                    "- Bluetooth is enabled on this computer\n" +
                    "- The firmware is running (device should advertise as GAINS-Pushup)\n" +
                    "- You are within range")
                return

            device_name = target.name
            self.log(f"Found {device_name} ({target.address})")
            self.log("Connecting...")

            # Connect to device
            self.client = BleakClient(target.address)
            await self.client.connect()

            if not self.client.is_connected:
                raise Exception("Failed to establish connection")

            self.log("Connected - discovering services...")

            # Give BLE stack time to complete service discovery
            await asyncio.sleep(0.5)

            # Verify control characteristic exists and is writable
            try:
                # Try to access the characteristic to verify it exists
                services = self.client.services
                found_ctrl = False
                for service in services:
                    for char in service.characteristics:
                        if char.uuid.lower() == CTRL_CHAR_UUID.lower():
                            found_ctrl = True
                            self.log(f"✓ Found control characteristic (writable: {'write' in char.properties})")
                            break
                    if found_ctrl:
                        break

                if not found_ctrl:
                    raise Exception(f"Control characteristic {CTRL_CHAR_UUID} not found")
            except Exception as e:
                self.log(f"Warning: Could not verify control characteristic: {e}")

            self.log("Starting notifications...")

            # Start notifications on IMU characteristic
            await self.client.start_notify(IMU_CHAR_UUID, self._imu_notification_handler)

            self.is_connected = True
            self.status_label.config(text=f"🟢 Connected to {device_name}", foreground="green")
            self.log(f"✓ Connected successfully to {device_name}!")

            # Start visualization update
            self.update_plots()

        except Exception as e:
            self.log(f"ERROR: Connection failed - {str(e)}")
            messagebox.showerror("Connection Error", f"Failed to connect:\n{str(e)}")

    def disconnect_device(self):
        """Disconnect from BLE device"""
        if self.client and self.is_connected:
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self._disconnect_async())

    async def _disconnect_async(self):
        """Async BLE disconnection"""
        try:
            if self.is_recording:
                await self.client.write_gatt_char(CTRL_CHAR_UUID, bytearray([0x00]))
                self.is_recording = False

            await self.client.disconnect()
            self.is_connected = False
            self.status_label.config(text="⚫ Not Connected", foreground="red")
            self.log("Disconnected")
        except Exception as e:
            self.log(f"ERROR during disconnect: {str(e)}")

    def _imu_notification_handler(self, sender, data):
        """Handle incoming IMU data from BLE"""
        if len(data) < 24:
            return

        # Parse 6 floats (ax, ay, az, gx, gy, gz) - little endian
        values = struct.unpack('<ffffff', data[:24])
        ax, ay, az, gx, gy, gz = values

        # Update buffers for visualization
        if self.start_time is None:
            self.start_time = datetime.now()

        elapsed = (datetime.now() - self.start_time).total_seconds()
        self.time_buffer.append(elapsed)

        self.accel_buffer['x'].append(ax)
        self.accel_buffer['y'].append(ay)
        self.accel_buffer['z'].append(az)

        self.gyro_buffer['x'].append(gx)
        self.gyro_buffer['y'].append(gy)
        self.gyro_buffer['z'].append(gz)

        # Store data if recording
        if self.is_recording:
            sample = {
                'timestamp': datetime.now().isoformat(),
                'elapsed_sec': elapsed,
                'ax': ax, 'ay': ay, 'az': az,
                'gx': gx, 'gy': gy, 'gz': gz
            }
            self.current_session.append(sample)

    def update_plots(self):
        """Update real-time plots"""
        if not self.is_connected:
            return

        try:
            if len(self.time_buffer) > 0:
                times = list(self.time_buffer)

                # Update accelerometer plot
                self.ax_accel.clear()
                self.ax_accel.plot(times, list(self.accel_buffer['x']), 'r-', label='X', linewidth=1.5)
                self.ax_accel.plot(times, list(self.accel_buffer['y']), 'g-', label='Y', linewidth=1.5)
                self.ax_accel.plot(times, list(self.accel_buffer['z']), 'b-', label='Z', linewidth=1.5)
                self.ax_accel.set_title('Accelerometer (g)')
                self.ax_accel.set_xlabel('Time (s)')
                self.ax_accel.set_ylim(-4, 4)
                self.ax_accel.legend(loc='upper right')
                self.ax_accel.grid(True, alpha=0.3)

                # Update gyroscope plot
                self.ax_gyro.clear()
                self.ax_gyro.plot(times, list(self.gyro_buffer['x']), 'r-', label='X', linewidth=1.5)
                self.ax_gyro.plot(times, list(self.gyro_buffer['y']), 'g-', label='Y', linewidth=1.5)
                self.ax_gyro.plot(times, list(self.gyro_buffer['z']), 'b-', label='Z', linewidth=1.5)
                self.ax_gyro.set_title('Gyroscope (°/s)')
                self.ax_gyro.set_xlabel('Time (s)')
                self.ax_gyro.set_ylim(-500, 500)
                self.ax_gyro.legend(loc='upper right')
                self.ax_gyro.grid(True, alpha=0.3)

                self.canvas.draw()

        except Exception as e:
            self.log(f"Plot update error: {str(e)}")

        # Schedule next update
        self.root.after(250, self.update_plots)  # Update at ~4 Hz

    def start_recording(self):
        """Start recording IMU data"""
        if not self.is_connected or not self.client or not self.client.is_connected:
            messagebox.showwarning("Not Connected", "Please connect to device first")
            return

        try:
            # Send start command to device
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(
                self.client.write_gatt_char(CTRL_CHAR_UUID, bytearray([0x01]))
            )
        except Exception as e:
            self.log(f"ERROR: Failed to start recording - {str(e)}")
            messagebox.showerror("Recording Error",
                f"Failed to send start command to device:\n{str(e)}\n\n" +
                "Try disconnecting and reconnecting.")
            return

        self.is_recording = True
        self.current_session = []
        self.recording_start_time = datetime.now()
        self.start_time = datetime.now()

        # Update UI
        self.record_btn.config(state='disabled')
        self.stop_btn.config(state='normal')
        self.record_status.config(text=f"⏺ Recording: {self.phase_var.get()} / {self.posture_var.get()}",
                                 foreground='red')
        self.timer_label.config(text="Duration: 0.0s", foreground='red')

        self.log(f"Recording started - Phase: {self.phase_var.get()}, Posture: {self.posture_var.get()}")

        # Start timer update
        self.update_timer()

    def update_timer(self):
        """Update the recording timer display"""
        if self.is_recording and self.recording_start_time:
            elapsed = (datetime.now() - self.recording_start_time).total_seconds()
            self.timer_label.config(text=f"Duration: {elapsed:.1f}s", foreground='red')
            # Schedule next update
            self.root.after(100, self.update_timer)  # Update every 100ms

    def stop_recording(self):
        """Stop recording and save session"""
        if not self.is_recording:
            return

        # Calculate final duration
        final_duration = (datetime.now() - self.recording_start_time).total_seconds() if self.recording_start_time else 0

        # Send stop command to device
        try:
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(
                self.client.write_gatt_char(CTRL_CHAR_UUID, bytearray([0x00]))
            )
        except Exception as e:
            self.log(f"Warning: Failed to send stop command - {str(e)}")

        self.is_recording = False

        # Update UI
        self.record_btn.config(state='normal')
        self.stop_btn.config(state='disabled')
        self.timer_label.config(text=f"Duration: {final_duration:.1f}s", foreground='green')

        if len(self.current_session) == 0:
            self.record_status.config(text="No data recorded", foreground='orange')
            self.timer_label.config(text="Duration: 0.0s", foreground='black')
            self.log("No data captured in this session")
            return

        # Save session with metadata
        session_data = {
            'session_id': len(self.all_sessions),
            'timestamp': datetime.now().isoformat(),
            'participant_id': self.participant_entry.get(),
            'imu_placement': self.placement_combo.get(),
            'notes': self.notes_text.get('1.0', tk.END).strip(),
            'phase_label': self.phase_var.get(),
            'posture_label': self.posture_var.get(),
            'sample_count': len(self.current_session),
            'duration_sec': (datetime.now() - self.start_time).total_seconds(),
            'sample_rate_hz': SAMPLE_RATE,
            'data': self.current_session
        }

        self.all_sessions.append(session_data)

        # Update summary
        self.summary_label.config(text=f"{len(self.all_sessions)} sessions recorded")
        self.session_listbox.insert(tk.END,
            f"#{session_data['session_id']}: {session_data['phase_label']}/{session_data['posture_label']} " +
            f"({session_data['sample_count']} samples, {session_data['duration_sec']:.1f}s)")

        self.record_status.config(
            text=f"✓ Saved {len(self.current_session)} samples ({session_data['duration_sec']:.1f}s)",
            foreground='green')

        self.log(f"Session saved: {len(self.current_session)} samples in {session_data['duration_sec']:.1f}s")

    def export_data(self):
        """Export all collected data to JSON"""
        if len(self.all_sessions) == 0:
            messagebox.showwarning("No Data", "No sessions recorded yet")
            return

        # Ask for save location
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"pushup_data_{timestamp}.json"

        filepath = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=default_filename
        )

        if not filepath:
            return

        # Prepare export data
        export_data = {
            'metadata': {
                'export_timestamp': datetime.now().isoformat(),
                'total_sessions': len(self.all_sessions),
                'sample_rate_hz': SAMPLE_RATE,
                'format_version': '1.0'
            },
            'sessions': self.all_sessions
        }

        # Write to file
        try:
            with open(filepath, 'w') as f:
                json.dump(export_data, f, indent=2)

            self.log(f"✓ Exported {len(self.all_sessions)} sessions to {filepath}")
            messagebox.showinfo("Export Successful",
                f"Data exported successfully!\n\nFile: {filepath}\n" +
                f"Sessions: {len(self.all_sessions)}\n" +
                f"Total samples: {sum(s['sample_count'] for s in self.all_sessions)}")
        except Exception as e:
            self.log(f"ERROR: Export failed - {str(e)}")
            messagebox.showerror("Export Error", f"Failed to export data:\n{str(e)}")

    def clear_all(self):
        """Clear all recorded sessions"""
        if len(self.all_sessions) == 0:
            return

        if messagebox.askyesno("Clear All Data",
            f"Are you sure you want to delete all {len(self.all_sessions)} sessions?\n" +
            "This cannot be undone!"):

            self.all_sessions = []
            self.current_session = []
            self.session_listbox.delete(0, tk.END)
            self.summary_label.config(text="0 sessions recorded")
            self.record_status.config(text="All data cleared. Idle.", foreground='black')
            self.log("All sessions cleared")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = PushupDataCollector(root)

    def on_closing():
        if app.is_connected:
            app.disconnect_device()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()
