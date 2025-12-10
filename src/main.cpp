/* GAINS Pushup Posture Classification
 * Using TensorFlow Lite Micro with CNN model
 * Classifies 4 posture types: good-form, hips-high, hips-sagging, partial-rom
 *
 * Hardware: Seeed Studio XIAO ESP32S3 + ICM-20600 IMU
 */
#include <Arduino.h>
#include "oled_display.h" 
#include "esp_task_wdt.h"
#include "esp_system.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"

#include "imu_provider.h"
#include "pushup_model_data.h"
#include "preprocessing.h"

// Note definitions for the speaker
#define NOTE_C4 262
#define NOTE_D4 294
#define NOTE_E4 330
#define NOTE_F4 349
#define NOTE_G4 392
#define NOTE_A4 440
#define NOTE_AS4 466
#define NOTE_C5 523

/* Pins on the Seeed Studio XIAO are labeled incorrectly.
   Pin 'n' here refers to pin 'n-1' on the board. Pins 5 and 6 (4 and 5 on the board) cannot be used since
   they are I2C pins between the ESP-32 Microcontroller and the OLED display */

const int BUZZER_PIN = 2;  // Buzzer pin
const int BUTTON_PIN = 3;  // pushbutton pin
const int RECORDING_LED_PIN = 4;  // LED pin

// LOW = false; HIGH = true
volatile bool buttonState = LOW;     // updated in ISR
bool lastButtonState = LOW;          // tracks last state for change detection
unsigned long lastOLEDUpdate = 0;    // timestamp for OLED throttling
const unsigned long OLED_UPDATE_INTERVAL = 200; // ms

void IRAM_ATTR handleButtonInterrupt() {
    // Read button quickly in ISR
    buttonState = digitalRead(BUTTON_PIN);
}

// ===== MODEL CONFIGURATION =====
// Update these based on your trained model metadata
constexpr int WINDOW_SIZE = 50;         // Number of IMU samples per window (50 @ 40Hz = 1.25s)
constexpr int NUM_CHANNELS = 6;         // ax, ay, az, gx, gy, gz
constexpr int NUM_POSTURE_CLASSES = 4;  // 4 posture types

// Posture labels - MUST match model output order!
// From pushup_model_metadata.json: ["good-form", "hips-high", "hips-sagging", "partial-rom"]
const char* posture_labels[NUM_POSTURE_CLASSES] = {
    "good-form",    // Class 0
    "hips-high",    // Class 1
    "hips-sagging", // Class 2
    "partial-rom"   // Class 3
};

// ===== INFERENCE RESULT STORAGE =====
// Structure to store individual inference results for voting
struct InferenceResult {
    float probabilities[NUM_POSTURE_CLASSES];  // All 4 class probabilities
    float max_confidence;                       // Best class confidence
    int best_class;                             // Best class index
    unsigned long timestamp;                    // When inference ran
};

constexpr int MAX_INFERENCE_RESULTS = 15;  // Support pushups up to 15s
InferenceResult inference_buffer[MAX_INFERENCE_RESULTS];
int inference_count = 0;

// ===== RECORDING STATE MACHINE =====
// Enhanced state machine (replaces simple bool inferenceEnabled)
enum RecordingState {
    IDLE,              // Ready to start
    RECORDING,         // Collecting inferences
    DISPLAYING_RESULT  // Showing voted result
};
RecordingState recording_state = IDLE;

// Final voted result storage
int final_voted_class = 0;
float final_voted_confidence = 0.0f;
int final_sample_count = 0;

// ===== NORMALIZATION PARAMETERS =====
// Replace with values from pushup_model_metadata.json generated during training
// These are computed from your training data: mean and std per channel
float imu_mean[NUM_CHANNELS] = {
    0.00042208,
    -0.00188804,
    -0.00668184,
    -0.73875794,
    1.9212489,
    -0.94398156
};
float imu_std[NUM_CHANNELS] = {
    0.09626791,
    0.04948182,
    0.20323046,
    7.33304658,
    27.609867,
    5.87350077
};

// ===== IMU BUFFER =====
// Circular buffer for sliding window
constexpr int BUFFER_SIZE = WINDOW_SIZE;
float imu_buffer[BUFFER_SIZE][NUM_CHANNELS];  // [50][6]
int buffer_index = 0;
int samples_collected = 0;

// ===== TENSORFLOW LITE MICRO =====
constexpr int kTensorArenaSize = 120 * 1024;  // 120 KB for CNN model (58 KB model + working memory)
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

// ===== INFERENCE CONTROL =====
constexpr int INFERENCE_INTERVAL_MS = 200;  // Run inference every  second when enabled
unsigned long lastInferenceTime = 0;

// ===== PREPROCESSING =====
Preprocessor preprocessor;  // Global preprocessor instance for filtering pipeline

// ===== HELPER FUNCTIONS =====

// Clear inference buffer when starting new recording
void ClearInferenceBuffer() {
    inference_count = 0;
    memset(inference_buffer, 0, sizeof(inference_buffer));
    Serial.println("[BUFFER] Inference buffer cleared");
}

// Store inference result in buffer
void StoreInferenceResult(const float* probs, int best, float max_prob) {
    if (inference_count >= MAX_INFERENCE_RESULTS) {
        Serial.println("[BUFFER WARNING] Maximum inferences reached (15)");
        return;
    }

    InferenceResult* result = &inference_buffer[inference_count];
    memcpy(result->probabilities, probs, sizeof(float) * NUM_POSTURE_CLASSES);
    result->max_confidence = max_prob;
    result->best_class = best;
    result->timestamp = millis();

    inference_count++;
    Serial.printf("[BUFFER] Stored inference #%d (class=%s, conf=%.1f%%)\n",
                  inference_count, posture_labels[best], max_prob * 100);
}

// Compute confidence-weighted vote across all stored inferences
bool ComputeWeightedVote(int& voted_class, float& voted_confidence) {
    const int MIN_SAMPLES = 2;

    if (inference_count < MIN_SAMPLES) {
        Serial.printf("[VOTE ERROR] Insufficient samples: %d (need %d)\n",
                      inference_count, MIN_SAMPLES);
        return false;
    }

    float weighted_scores[NUM_POSTURE_CLASSES] = {0};
    float total_weight = 0;

    // Compute weighted scores
    for (int i = 0; i < inference_count; i++) {
        float weight = inference_buffer[i].max_confidence;
        total_weight += weight;

        for (int c = 0; c < NUM_POSTURE_CLASSES; c++) {
            weighted_scores[c] += inference_buffer[i].probabilities[c] * weight;
        }
    }

    // Normalize and find winner
    // voted_class = 0;
    // voted_confidence = 0;
    // for (int c = 0; c < NUM_POSTURE_CLASSES; c++) {
    //     weighted_scores[c] /= total_weight;
    //     if (weighted_scores[c] > voted_confidence) {
    //         voted_confidence = weighted_scores[c];
    //         voted_class = c;
    //     }
    // }

    const InferenceResult& last = inference_buffer[inference_count - 1];

    voted_class = last.best_class;
    voted_confidence = last.max_confidence;

    // Debug output
    Serial.println("\n========== WEIGHTED VOTE ==========");
    Serial.printf("Samples: %d\n", inference_count);
    for (int c = 0; c < NUM_POSTURE_CLASSES; c++) {
        Serial.printf("  %s: %.1f%%\n", posture_labels[c],
                      weighted_scores[c] * 100);
    }
    Serial.printf("Winner: %s (%.1f%% confident)\n",
                  posture_labels[voted_class], voted_confidence * 100);
    Serial.println("===================================\n");

    return true;
}

// Display recording status with sample count
void DisplayRecordingStatus() {
    oled_display_clear();
    oled_display_text(0, 0, "GAINS");
    oled_display_text(0, 16, "Recording...");

    char sample_line[32];
    snprintf(sample_line, sizeof(sample_line), "%d samples", inference_count);
    oled_display_text(0, 32, sample_line);

    oled_display_update();
}

// Display voted result with confidence
void DisplayVotedResult(int voted_class, float voted_conf, int sample_count) {
    oled_display_clear();
    oled_display_text(0, 0, "GAINS");
    oled_display_text(0, 12, "Result:");

    // Posture label (may wrap to two lines)
    oled_display_text(0, 24, posture_labels[voted_class]);

    // Confidence
    char conf_line[32];
    snprintf(conf_line, sizeof(conf_line), "%.0f%% confident", voted_conf * 100);
    oled_display_text(0, 36, conf_line);

    // Sample count
    char sample_line[32];
    snprintf(sample_line, sizeof(sample_line), "(%d samples)", sample_count);
    oled_display_text(0, 48, sample_line);

    oled_display_update();
}

// Display error when insufficient samples collected
void DisplayInsufficientSamplesError(int sample_count) {
    oled_display_clear();
    oled_display_text(0, 0, "GAINS");
    oled_display_text(0, 16, "ERROR");
    oled_display_text(0, 32, "Need 2+ samples");

    char got_line[32];
    snprintf(got_line, sizeof(got_line), "Got: %d", sample_count);
    oled_display_text(0, 48, got_line);

    oled_display_update();
}

// Handle recording state transitions (unified button/serial handler)
void HandleRecordingToggle(bool is_button) {
    const char* source = is_button ? "button" : "serial";

    switch (recording_state) {
        case IDLE:
            // Start recording
            recording_state = RECORDING;
            ClearInferenceBuffer();

            Serial.printf("[STATE] IDLE -> RECORDING (via %s)\n", source);

            // Visual feedback
            digitalWrite(RECORDING_LED_PIN, HIGH);
            oled_display_clear();
            oled_display_text(0, 0, "GAINS");
            oled_display_text(0, 20, "Recording...");
            oled_display_text(0, 40, "0 samples");
            oled_display_update();

            // Audio feedback
            tone(BUZZER_PIN, NOTE_D4, 100);
            delay(100);
            noTone(BUZZER_PIN);
            delay(50);
            tone(BUZZER_PIN, NOTE_D4, 100);
            delay(100);
            noTone(BUZZER_PIN);
            break;

        case RECORDING:
            // Stop recording and compute vote
            recording_state = DISPLAYING_RESULT;

            Serial.printf("[STATE] RECORDING -> DISPLAYING_RESULT (via %s)\n", source);

            // Compute vote
            if (ComputeWeightedVote(final_voted_class, final_voted_confidence)) {
                final_sample_count = inference_count;
                DisplayVotedResult(final_voted_class, final_voted_confidence,
                                  final_sample_count);
            } else {
                // Insufficient samples
                final_sample_count = inference_count;
                DisplayInsufficientSamplesError(final_sample_count);
            }

            // Visual feedback
            digitalWrite(RECORDING_LED_PIN, LOW);

            // Audio feedback
            tone(BUZZER_PIN, NOTE_C4, 200);
            delay(200);
            noTone(BUZZER_PIN);
            break;

        case DISPLAYING_RESULT:
            // Return to idle
            recording_state = IDLE;

            Serial.printf("[STATE] DISPLAYING_RESULT -> IDLE (via %s)\n", source);

            oled_display_clear();
            oled_display_text(0, 10, "GAINS");
            oled_display_text(0, 30, "Press to start");
            oled_display_update();
            break;
    }

    lastOLEDUpdate = millis();
}

void NormalizeWindow(float normalized_window[WINDOW_SIZE][NUM_CHANNELS]) {
    // Normalize the sliding window using saved mean/std
    for (int i = 0; i < WINDOW_SIZE; i++) {
        int buf_idx = (buffer_index - WINDOW_SIZE + i + BUFFER_SIZE) % BUFFER_SIZE;
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            normalized_window[i][ch] = (imu_buffer[buf_idx][ch] - imu_mean[ch]) / (imu_std[ch] + 1e-8f);
        }
    }
}

void RunInference() {
    if (samples_collected < WINDOW_SIZE) {
        // Not enough samples yet
        return;
    }

    // Feed watchdog to prevent reset during inference
    esp_task_wdt_reset();

    // Create normalized window
    float normalized_window[WINDOW_SIZE][NUM_CHANNELS];
    NormalizeWindow(normalized_window);

    // Get input tensor
    TfLiteTensor* model_input = interpreter->input(0);

    // Get quantization parameters
    const float input_scale = model_input->params.scale;
    const int input_zp = model_input->params.zero_point;

    // Quantize and copy data to model input
    // Model expects shape: [1, WINDOW_SIZE, NUM_CHANNELS]
    for (int t = 0; t < WINDOW_SIZE; t++) {
        for (int ch = 0; ch < NUM_CHANNELS; ch++) {
            int idx = t * NUM_CHANNELS + ch;
            float val = normalized_window[t][ch];

            // Quantize: q = round(val / scale) + zero_point
            int32_t q = static_cast<int32_t>(roundf(val / input_scale)) + input_zp;

            // Clamp to int8 range
            if (q < -128) q = -128;
            if (q > 127) q = 127;

            model_input->data.int8[idx] = static_cast<int8_t>(q);
        }
    }

    // Run inference
    Serial.println("[INFERENCE] Starting model invoke...");
    unsigned long start_time = millis();

    TfLiteStatus invoke_status = interpreter->Invoke();

    unsigned long inference_time = millis() - start_time;
    Serial.printf("[INFERENCE] Completed in %lu ms\n", inference_time);

    if (invoke_status != kTfLiteOk) {
        Serial.println("ERROR: Inference failed!");
        return;
    }

    // Feed watchdog again after inference
    esp_task_wdt_reset();

    // Get output tensor (single-task model with 1 output: posture)
    TfLiteTensor* posture_output = interpreter->output(0);

    // Dequantize posture predictions
    const float posture_scale = posture_output->params.scale;
    const int posture_zp = posture_output->params.zero_point;

    float posture_probs[NUM_POSTURE_CLASSES];
    int best_posture = 0;
    float max_posture_prob = -1.0f;

    for (int i = 0; i < NUM_POSTURE_CLASSES; i++) {
        float p = posture_scale * (static_cast<int>(posture_output->data.int8[i]) - posture_zp);
        posture_probs[i] = p;
        if (p > max_posture_prob) {
            max_posture_prob = p;
            best_posture = i;
        }
    }

    // Print results to serial
    Serial.println("\n========== PREDICTION ==========");
    Serial.print("Posture: ");
    Serial.print(posture_labels[best_posture]);
    Serial.print(" (");
    Serial.print(max_posture_prob * 100, 1);
    Serial.println("%)");

    // Print all probabilities for debugging
    Serial.println("\nAll Posture Probabilities:");
    for (int i = 0; i < NUM_POSTURE_CLASSES; i++) {
        Serial.print("  ");
        Serial.print(posture_labels[i]);
        Serial.print(": ");
        Serial.print(posture_probs[i] * 100, 1);
        Serial.println("%");
    }
    Serial.println("================================\n");

    // Store result and update display if recording
    if (recording_state == RECORDING) {
        // Store result in buffer
        StoreInferenceResult(posture_probs, best_posture, max_posture_prob);

        // Update display with sample count (throttled)
        unsigned long currentTime = millis();
        if (currentTime - lastOLEDUpdate >= OLED_UPDATE_INTERVAL) {
            DisplayRecordingStatus();
            lastOLEDUpdate = currentTime;
        }
    }
}

// ====================================================================
// setup()
// ====================================================================
void setup() {
    Serial.begin(115200);
    delay(2000);  // Longer delay for serial to stabilize

    // VERY FIRST debug output
    Serial.println("\n\n\n");
    Serial.println("##############################");
    Serial.println("#  SERIAL OUTPUT WORKING!   #");
    Serial.println("##############################");

    // Check reset reason
    esp_reset_reason_t reset_reason = esp_reset_reason();
    Serial.print("Reset reason: ");
    switch(reset_reason) {
        case ESP_RST_POWERON: Serial.println("Power on"); break;
        case ESP_RST_SW: Serial.println("Software reset"); break;
        case ESP_RST_PANIC: Serial.println("Exception/panic"); break;
        case ESP_RST_INT_WDT: Serial.println("Interrupt watchdog"); break;
        case ESP_RST_TASK_WDT: Serial.println("Task watchdog"); break;
        case ESP_RST_WDT: Serial.println("Other watchdog"); break;
        default: Serial.printf("Other (%d)\n", reset_reason); break;
    }

    Serial.flush();
    delay(500);

    tflite::InitializeTarget();

    // Configure watchdog timer (10 second timeout)
    esp_task_wdt_init(10, true);  // 10 second timeout, panic on timeout
    esp_task_wdt_add(NULL);       // Add current task to watchdog
    Serial.println("✓ Watchdog timer configured");

    Serial.println("\n========================================");
    Serial.println("  GAINS - Pushup Posture Classifier");
    Serial.println("========================================");
    Serial.flush();

    // Initialize OLED
    oled_display_init();
    oled_display_text(0, 10, "GAINS");
    oled_display_update();

    // Initialize button and LED
    pinMode(BUTTON_PIN, INPUT_PULLUP);  // Enable pull-up to prevent floating pin
    pinMode(BUZZER_PIN, OUTPUT);
    pinMode(RECORDING_LED_PIN, OUTPUT);

    // Read initial button state before attaching interrupt
    buttonState = digitalRead(BUTTON_PIN);
    lastButtonState = buttonState;

    attachInterrupt(digitalPinToInterrupt(BUTTON_PIN), handleButtonInterrupt, CHANGE);

    // Initialize IMU
    if (!SetupIMU()) {
        Serial.println("ERROR: IMU failed!");
        oled_display_clear();
        oled_display_text(0, 10, "ERROR");
        oled_display_text(0, 30, "IMU Failed!");
        oled_display_update();
        while (1) delay(1000);
    }
    Serial.println("✓ IMU ready");

    // Initialize preprocessing pipeline
    preprocessor.Init();
    Serial.println("✓ Preprocessing filters initialized");

    // Load TFLite model
    model = tflite::GetModel(g_pushup_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("ERROR: Model version mismatch!");
        Serial.printf("Model version: %d, Expected: %d\n",
                      model->version(), TFLITE_SCHEMA_VERSION);
        oled_display_clear();
        oled_display_text(0, 10, "ERROR");
        oled_display_text(0, 30, "Model Version!");
        oled_display_update();
        while (1) delay(1000);
    }
    Serial.println("✓ Model loaded");

    // Setup TFLite interpreter
    static tflite::AllOpsResolver micro_op_resolver;

    static tflite::MicroInterpreter static_interpreter(
        model, micro_op_resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("ERROR: Tensor allocation failed!");
        oled_display_clear();
        oled_display_text(0, 10, "ERROR");
        oled_display_text(0, 30, "Tensor Alloc!");
        oled_display_update();
        while (1) delay(1000);
    }
    Serial.println("✓ Model ready");

    // Print model info
    TfLiteTensor* input = interpreter->input(0);
    Serial.printf("Input shape: [%d, %d, %d]\n",
                  input->dims->data[0],
                  input->dims->data[1],
                  input->dims->data[2]);
    Serial.printf("Input type: %s\n",
                  input->type == kTfLiteInt8 ? "INT8" : "FLOAT32");

    Serial.println("========================================");
    Serial.println("System ready!");
    Serial.println("Press button or 'r' key to START recording");
    Serial.println("Press again to STOP and get result");
    Serial.println("Press third time to return to IDLE");
    Serial.println("========================================\n");

    // Initialize state machine
    recording_state = IDLE;
    oled_display_clear();
    oled_display_text(0, 10, "GAINS");
    oled_display_text(0, 30, "Press to start");
    oled_display_update();

    // Clear any serial data sent during connection (shell prompts, etc)
    delay(500);
    while (Serial.available() > 0) {
        Serial.read();
    }
    Serial.println("Serial buffer cleared - ready for input!");
}

// ====================================================================
// loop()
// ====================================================================
void loop() {
    // Timing diagnostics: Track loop duration
    static unsigned long last_loop_time = 0;
    unsigned long loop_start = millis();
    unsigned long loop_duration = loop_start - last_loop_time;

    // Warn if loop is taking too long (> 100ms indicates blocking)
    if (last_loop_time > 0 && loop_duration > 100) {
        Serial.printf("[TIMING WARNING] Loop took %lu ms (expected ~10ms)\n", loop_duration);
    }

    // Handle button state changes
    bool currentState = buttonState;

    if (currentState != lastButtonState) {
        lastButtonState = currentState;
        // Print button/LED state for debugging
        Serial.printf("Button: %s, LED: %s\n",
                  currentState ? "HIGH" : "LOW",
                  currentState ? "ON" : "OFF");

        if (currentState == HIGH) {
            HandleRecordingToggle(true);  // true = button source
        }
    }

    // Handle serial input - only respond to 'r' or 'R' key
    if (Serial.available() > 0) {
        char key = Serial.read();
        // Clear any remaining characters
        while (Serial.available() > 0) {
            Serial.read();
        }

        // Only toggle recording on 'r' or 'R' key
        if (key == 'r' || key == 'R') {
            Serial.printf("Key pressed: '%c' (0x%02X) - toggling recording\n", key, key);
            HandleRecordingToggle(false);  // false = serial source
        } else {
            Serial.printf("Key pressed: '%c' (0x%02X) - ignored (press 'r' to toggle)\n", key, key);
        }
    }

    // Always read IMU data (keep buffer updated)
    float raw_accel[3], raw_gyro[3];
    if (ReadIMU(raw_accel, raw_gyro)) {
        // Apply preprocessing pipeline:
        // 1. Median filter (denoise)
        // 2. Lowpass filter on accel (10 Hz)
        // 3. Highpass filter on gyro (0.2 Hz)
        // 4. Gravity removal from accel (0.5 Hz lowpass estimate)
        float processed_sample[NUM_CHANNELS];
        preprocessor.ProcessSample(raw_accel, raw_gyro, processed_sample);

        // Store preprocessed data in circular buffer: [ax, ay, az, gx, gy, gz]
        // This data is now: linear accel (no gravity) + drift-free gyro
        imu_buffer[buffer_index][0] = processed_sample[0];
        imu_buffer[buffer_index][1] = processed_sample[1];
        imu_buffer[buffer_index][2] = processed_sample[2];
        imu_buffer[buffer_index][3] = processed_sample[3];
        imu_buffer[buffer_index][4] = processed_sample[4];
        imu_buffer[buffer_index][5] = processed_sample[5];

        buffer_index = (buffer_index + 1) % BUFFER_SIZE;
        if (samples_collected < WINDOW_SIZE) {
            samples_collected++;
        }

        // Debug: Print raw vs processed data every 20 samples (every 0.5 seconds @ 40Hz)
        static int debug_count = 0;
        // if (samples_collected >= WINDOW_SIZE && (++debug_count % 20 == 0)) {
        //     Serial.printf("[RAW] ax=%.3f, ay=%.3f, az=%.3f | gx=%.3f, gy=%.3f, gz=%.3f\n",
        //                   raw_accel[0], raw_accel[1], raw_accel[2],
        //                   raw_gyro[0], raw_gyro[1], raw_gyro[2]);
        //     Serial.printf("[PROCESSED] ax=%.3f, ay=%.3f, az=%.3f | gx=%.3f, gy=%.3f, gz=%.3f\n",
        //                   processed_sample[0], processed_sample[1], processed_sample[2],
        //                   processed_sample[3], processed_sample[4], processed_sample[5]);
        //     Serial.println();
        // }
    }

    // Run inference periodically ONLY when recording
    if (recording_state == RECORDING) {
        unsigned long currentTime = millis();
        if (currentTime - lastInferenceTime >= INFERENCE_INTERVAL_MS) {
            lastInferenceTime = currentTime;
            RunInference();
        }
    }

    // Feed watchdog in main loop to prevent timeout when inference is disabled
    // or during buffer warmup period (first 50 samples)
    esp_task_wdt_reset();

    delay(10);  // ~100 Hz sampling rate

    // Update timing for next iteration
    last_loop_time = millis();
}
