#ifndef PREPROCESSING_H_
#define PREPROCESSING_H_

#include <cstdint>

// Preprocessing pipeline configuration
constexpr int MEDIAN_KERNEL_SIZE = 3;
constexpr int NUM_IMU_CHANNELS = 6;  // ax, ay, az, gx, gy, gz
constexpr int ACCEL_CHANNELS = 3;    // ax, ay, az
constexpr int GYRO_CHANNELS = 3;     // gx, gy, gz

// Butterworth filter state for 4th-order filter (2 biquad sections)
struct BiquadState {
    float w1;  // State variable 1
    float w2;  // State variable 2
};

struct ButterworthFilter {
    BiquadState section1;
    BiquadState section2;
    // Section 1 coefficients
    float b0_1, b1_1, b2_1;  // Numerator coefficients for section 1
    float a1_1, a2_1;         // Denominator coefficients for section 1 (a0 = 1)
    // Section 2 coefficients
    float b0_2, b1_2, b2_2;  // Numerator coefficients for section 2
    float a1_2, a2_2;         // Denominator coefficients for section 2 (a0 = 1)
};

// Preprocessing filters
class Preprocessor {
public:
    Preprocessor();

    // Initialize all filters
    void Init();

    // Reset filter states
    void Reset();

    // Process single IMU sample
    // Input: raw_accel[3], raw_gyro[3] in physical units (g, deg/s)
    // Output: processed_sample[6] = [ax, ay, az, gx, gy, gz] with gravity removed
    void ProcessSample(const float* raw_accel, const float* raw_gyro, float* processed_sample);

private:
    // Median filter buffers (rolling window of size 3)
    float accel_median_buffer[ACCEL_CHANNELS][MEDIAN_KERNEL_SIZE];
    float gyro_median_buffer[GYRO_CHANNELS][MEDIAN_KERNEL_SIZE];
    int median_index;

    // Butterworth lowpass filters for accelerometer (10 Hz @ 40 Hz sample rate)
    ButterworthFilter accel_lowpass[ACCEL_CHANNELS];

    // Butterworth highpass filters for gyroscope (0.2 Hz @ 40 Hz sample rate)
    ButterworthFilter gyro_highpass[GYRO_CHANNELS];

    // Butterworth lowpass filter for gravity estimation (0.5 Hz @ 40 Hz sample rate)
    ButterworthFilter gravity_filter[ACCEL_CHANNELS];

    // Helper functions
    float ApplyMedianFilter(float* buffer, float new_value);
    float ApplyBiquad(BiquadState* state, float input, float b0, float b1, float b2, float a1, float a2);
    float ApplyButterworthFilter(ButterworthFilter* filter, float input);

    // Initialize filter coefficients
    void InitAccelLowpass();
    void InitGyroHighpass();
    void InitGravityFilter();
};

#endif  // PREPROCESSING_H_
