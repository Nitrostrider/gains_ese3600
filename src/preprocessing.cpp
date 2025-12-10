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
    // Designed using scipy.signal.butter(4, 10, 'low', fs=40)
    // Implemented as cascade of two 2nd-order sections (biquads)

    // Normalized frequency: Wn = 10 / (40/2) = 0.5
    // This is a relatively high cutoff (Nyquist/4)

    // Biquad coefficients (both sections are identical for Butterworth)
    // Digital filter coefficients from bilinear transform
    const float b0 = 0.0947916f;
    const float b1 = 0.1895832f;
    const float b2 = 0.0947916f;
    const float a1 = -0.9149758f;
    const float a2 = 0.2941422f;

    for (int i = 0; i < ACCEL_CHANNELS; i++) {
        accel_lowpass[i].b0 = b0;
        accel_lowpass[i].b1 = b1;
        accel_lowpass[i].b2 = b2;
        accel_lowpass[i].a1 = a1;
        accel_lowpass[i].a2 = a2;
    }
}

void Preprocessor::InitGyroHighpass() {
    // 4th-order Butterworth highpass filter: 0.2 Hz cutoff @ 40 Hz sample rate
    // Designed using scipy.signal.butter(4, 0.2, 'high', fs=40)
    // Removes low-frequency drift from gyroscope

    // Normalized frequency: Wn = 0.2 / (40/2) = 0.01
    // Very low cutoff - removes DC and slow drift

    // Biquad coefficients
    const float b0 = 0.9968781f;
    const float b1 = -1.9937562f;
    const float b2 = 0.9968781f;
    const float a1 = -1.9937542f;
    const float a2 = 0.9937582f;

    for (int i = 0; i < GYRO_CHANNELS; i++) {
        gyro_highpass[i].b0 = b0;
        gyro_highpass[i].b1 = b1;
        gyro_highpass[i].b2 = b2;
        gyro_highpass[i].a1 = a1;
        gyro_highpass[i].a2 = a2;
    }
}

void Preprocessor::InitGravityFilter() {
    // 4th-order Butterworth lowpass filter: 0.5 Hz cutoff @ 40 Hz sample rate
    // Designed using scipy.signal.butter(4, 0.5, 'low', fs=40)
    // Extracts gravity component from accelerometer

    // Normalized frequency: Wn = 0.5 / (40/2) = 0.025
    // Very low cutoff - extracts quasi-static gravity

    // Biquad coefficients
    const float b0 = 0.0000152f;
    const float b1 = 0.0000304f;
    const float b2 = 0.0000152f;
    const float a1 = -1.9844048f;
    const float a2 = 0.9844656f;

    for (int i = 0; i < ACCEL_CHANNELS; i++) {
        gravity_filter[i].b0 = b0;
        gravity_filter[i].b1 = b1;
        gravity_filter[i].b2 = b2;
        gravity_filter[i].a1 = a1;
        gravity_filter[i].a2 = a2;
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
    float intermediate = ApplyBiquad(&filter->section1, input,
                                     filter->b0, filter->b1, filter->b2,
                                     filter->a1, filter->a2);

    float output = ApplyBiquad(&filter->section2, intermediate,
                               filter->b0, filter->b1, filter->b2,
                               filter->a1, filter->a2);

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
