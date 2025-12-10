from scipy.signal import butter
import numpy as np

# 10 Hz lowpass at 40 Hz sample rate (for accelerometer)
sos_accel = butter(4, 10, 'low', fs=40, output='sos')
print("=== ACCEL LOWPASS (10 Hz) ===")
print("Section 1 (b0, b1, b2, a0, a1, a2):")
print(sos_accel[0])
print("\nSection 2 (b0, b1, b2, a0, a1, a2):")
print(sos_accel[1])

# 0.2 Hz highpass at 40 Hz sample rate (for gyroscope)
sos_gyro = butter(4, 0.2, 'high', fs=40, output='sos')
print("\n=== GYRO HIGHPASS (0.2 Hz) ===")
print("Section 1:")
print(sos_gyro[0])
print("\nSection 2:")
print(sos_gyro[1])

# 0.5 Hz lowpass at 40 Hz sample rate (for gravity)
sos_grav = butter(4, 0.5, 'low', fs=40, output='sos')
print("\n=== GRAVITY LOWPASS (0.5 Hz) ===")
print("Section 1:")
print(sos_grav[0])
print("\nSection 2:")
print(sos_grav[1])