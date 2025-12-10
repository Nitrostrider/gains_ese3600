#include "preprocessing.h"
#include <Arduino.h>
#include <cmath>
#include <algorithm>

// ============================================================================
// CONSTRUCTOR
// ============================================================================

Preprocessor::Preprocessor() {
    Init();
}

// ============================================================================
// INITIALIZATION
// ============================================================================

void Preprocessor::Init() {
    Reset();
    InitAccelLowpass();
    InitGyroHighpass();
    InitGravityFilter();
}

void Preprocessor::Reset() {
    // Clear median filter buffers
    median_index = 0;
    for (int i = 0; i < ACCEL_CHANNELS; i++) {
        for (int j = 0; j < MEDIAN_KERNEL_SIZE; j++) {
            accel_median_buffer[i][j] = 0.0f;
        }
    }
    for (int i = 0; i < GYRO_CHANNELS; i++) {
        for (int j = 0; j < MEDIAN_KERNEL_SIZE; j++) {
            gyro_median_buffer[i][j] = 0.0f;
        }
    }

    // Clear filter states
    for (int i = 0; i < ACCEL_CHANNELS; i++) {
        accel_lowpass[i].section1.w1 = 0.0f;
        accel_lowpass[i].section1.w2 = 0.0f;
        accel_lowpass[i].section2.w1 = 0.0f;
        accel_lowpass[i].section2.w2 = 0.0f;

        gravity_filter[i].section1.w1 = 0.0f;
        gravity_filter[i].section1.w2 = 0.0f;
        gravity_filter[i].section2.w1 = 0.0f;
        gravity_filter[i].section2.w2 = 0.0f;
    }
    for (int i = 0; i < GYRO_CHANNELS; i++) {
        gyro_highpass[i].section1.w1 = 0.0f;
        gyro_highpass[i].section1.w2 = 0.0f;
        gyro_highpass[i].section2.w1 = 0.0f;
        gyro_highpass[i].section2.w2 = 0.0f;
    }
}

// ============================================================================
// FILTER COEFFICIENT INITIALIZATION
// ============================================================================

void Preprocessor::InitAccelLowpass() {
    // 4th-order Butterworth lowpass filter: 10 Hz cutoff @ 40 Hz sample rate
    // Designed using scipy.signal.butter(4, 10, 'low', fs=40, output='sos')
    // Implemented as cascade of two 2nd-order sections (biquads)

    // Section 1 coefficients
    const float b0_1 = 0.0939808514f;
    const float b1_1 = 0.187961703f;
    const float b2_1 = 0.0939808514f;
    const float a1_1 = 0.0f;  // Nearly zero from scipy (1.38e-16)
    const float a2_1 = 0.0395661299f;

    // Section 2 coefficients
    const float b0_2 = 1.0f;
    const float b1_2 = 2.0f;
    const float b2_2 = 1.0f;
    const float a1_2 = 0.0f;  // Nearly zero from scipy (1.61e-16)
    const float a2_2 = 0.446462692f;

    for (int i = 0; i < ACCEL_CHANNELS; i++) {
        accel_lowpass[i].b0_1 = b0_1;
        accel_lowpass[i].b1_1 = b1_1;
        accel_lowpass[i].b2_1 = b2_1;
        accel_lowpass[i].a1_1 = a1_1;
        accel_lowpass[i].a2_1 = a2_1;

        accel_lowpass[i].b0_2 = b0_2;
        accel_lowpass[i].b1_2 = b1_2;
        accel_lowpass[i].b2_2 = b2_2;
        accel_lowpass[i].a1_2 = a1_2;
        accel_lowpass[i].a2_2 = a2_2;
    }
}

void Preprocessor::InitGyroHighpass() {
    // 4th-order Butterworth highpass filter: 0.2 Hz cutoff @ 40 Hz sample rate
    // Designed using scipy.signal.butter(4, 0.2, 'high', fs=40, output='sos')
    // Removes low-frequency drift from gyroscope

    // Section 1 coefficients
    const float b0_1 = 0.95978223f;
    const float b1_1 = -1.91956446f;
    const float b2_1 = 0.95978223f;
    const float a1_1 = -1.94263823f;
    const float a2_1 = 0.94359728f;

    // Section 2 coefficients
    const float b0_2 = 1.0f;
    const float b1_2 = -2.0f;
    const float b2_2 = 1.0f;
    const float a1_2 = -1.97526963f;
    const float a2_2 = 0.97624479f;

    for (int i = 0; i < GYRO_CHANNELS; i++) {
        gyro_highpass[i].b0_1 = b0_1;
        gyro_highpass[i].b1_1 = b1_1;
        gyro_highpass[i].b2_1 = b2_1;
        gyro_highpass[i].a1_1 = a1_1;
        gyro_highpass[i].a2_1 = a2_1;

        gyro_highpass[i].b0_2 = b0_2;
        gyro_highpass[i].b1_2 = b1_2;
        gyro_highpass[i].b2_2 = b2_2;
        gyro_highpass[i].a1_2 = a1_2;
        gyro_highpass[i].a2_2 = a2_2;
    }
}

void Preprocessor::InitGravityFilter() {
    // 4th-order Butterworth lowpass filter: 0.5 Hz cutoff @ 40 Hz sample rate
    // Designed using scipy.signal.butter(4, 0.5, 'low', fs=40, output='sos')
    // Extracts gravity component from accelerometer

    // Section 1 coefficients
    const float b0_1 = 2.15056874e-06f;
    const float b1_1 = 4.30113747e-06f;
    const float b2_1 = 2.15056874e-06f;
    const float a1_1 = -1.85907627f;
    const float a2_1 = 0.864824899f;

    // Section 2 coefficients
    const float b0_2 = 1.0f;
    const float b1_2 = 2.0f;
    const float b2_2 = 1.0f;
    const float a1_2 = -1.93571484f;
    const float a2_2 = 0.94170045f;

    for (int i = 0; i < ACCEL_CHANNELS; i++) {
        gravity_filter[i].b0_1 = b0_1;
        gravity_filter[i].b1_1 = b1_1;
        gravity_filter[i].b2_1 = b2_1;
        gravity_filter[i].a1_1 = a1_1;
        gravity_filter[i].a2_1 = a2_1;

        gravity_filter[i].b0_2 = b0_2;
        gravity_filter[i].b1_2 = b1_2;
        gravity_filter[i].b2_2 = b2_2;
        gravity_filter[i].a1_2 = a1_2;
        gravity_filter[i].a2_2 = a2_2;
    }
}

// ============================================================================
// FILTER APPLICATION
// ============================================================================

float Preprocessor::ApplyMedianFilter(float* buffer, float new_value) {
    // Store new value in circular buffer
    buffer[median_index] = new_value;

    // Create sorted copy for median calculation
    float sorted[MEDIAN_KERNEL_SIZE];
    for (int i = 0; i < MEDIAN_KERNEL_SIZE; i++) {
        sorted[i] = buffer[i];
    }

    // Simple bubble sort for small array
    for (int i = 0; i < MEDIAN_KERNEL_SIZE - 1; i++) {
        for (int j = 0; j < MEDIAN_KERNEL_SIZE - i - 1; j++) {
            if (sorted[j] > sorted[j + 1]) {
                float temp = sorted[j];
                sorted[j] = sorted[j + 1];
                sorted[j + 1] = temp;
            }
        }
    }

    // Return median (middle element)
    return sorted[MEDIAN_KERNEL_SIZE / 2];
}

float Preprocessor::ApplyBiquad(BiquadState* state, float input,
                                 float b0, float b1, float b2,
                                 float a1, float a2) {
    // Direct Form II transposed structure
    // More numerically stable than Direct Form I

    float output = b0 * input + state->w1;
    state->w1 = b1 * input - a1 * output + state->w2;
    state->w2 = b2 * input - a2 * output;

    return output;
}

float Preprocessor::ApplyButterworthFilter(ButterworthFilter* filter, float input) {
    // Apply cascade of two biquad sections for 4th-order filter
    // Section 1 uses coefficients b0_1, b1_1, b2_1, a1_1, a2_1
    float intermediate = ApplyBiquad(&filter->section1, input,
                                     filter->b0_1, filter->b1_1, filter->b2_1,
                                     filter->a1_1, filter->a2_1);

    // Section 2 uses coefficients b0_2, b1_2, b2_2, a1_2, a2_2
    float output = ApplyBiquad(&filter->section2, intermediate,
                               filter->b0_2, filter->b1_2, filter->b2_2,
                               filter->a1_2, filter->a2_2);

    return output;
}

// ============================================================================
// MAIN PROCESSING FUNCTION
// ============================================================================

void Preprocessor::ProcessSample(const float* raw_accel, const float* raw_gyro,
                                   float* processed_sample) {
    // Step 1: Median filter (denoise)
    float accel_median[ACCEL_CHANNELS];
    float gyro_median[GYRO_CHANNELS];

    for (int i = 0; i < ACCEL_CHANNELS; i++) {
        accel_median[i] = ApplyMedianFilter(accel_median_buffer[i], raw_accel[i]);
    }
    for (int i = 0; i < GYRO_CHANNELS; i++) {
        gyro_median[i] = ApplyMedianFilter(gyro_median_buffer[i], raw_gyro[i]);
    }

    // Update median buffer index
    median_index = (median_index + 1) % MEDIAN_KERNEL_SIZE;

    // Step 2: Apply lowpass filter to accelerometer (10 Hz)
    float accel_lowpass_out[ACCEL_CHANNELS];
    for (int i = 0; i < ACCEL_CHANNELS; i++) {
        accel_lowpass_out[i] = ApplyButterworthFilter(&accel_lowpass[i], accel_median[i]);
    }

    // Step 3: Apply highpass filter to gyroscope (0.2 Hz drift removal)
    float gyro_highpass_out[GYRO_CHANNELS];
    for (int i = 0; i < GYRO_CHANNELS; i++) {
        gyro_highpass_out[i] = ApplyButterworthFilter(&gyro_highpass[i], gyro_median[i]);
    }

    // Step 4: Estimate gravity (0.5 Hz lowpass on filtered accel)
    float gravity[ACCEL_CHANNELS];
    for (int i = 0; i < ACCEL_CHANNELS; i++) {
        gravity[i] = ApplyButterworthFilter(&gravity_filter[i], accel_lowpass_out[i]);
    }

    // Step 5: Remove gravity to get linear acceleration
    float linear_accel[ACCEL_CHANNELS];
    for (int i = 0; i < ACCEL_CHANNELS; i++) {
        linear_accel[i] = accel_lowpass_out[i] - gravity[i];
    }

    // Output: [ax, ay, az, gx, gy, gz] with gravity removed
    processed_sample[0] = linear_accel[0];
    processed_sample[1] = linear_accel[1];
    processed_sample[2] = linear_accel[2];
    processed_sample[3] = gyro_highpass_out[0];
    processed_sample[4] = gyro_highpass_out[1];
    processed_sample[5] = gyro_highpass_out[2];
}
