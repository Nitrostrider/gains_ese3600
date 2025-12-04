/* GAINS Pushup Posture & Phase Classification
 * Using TensorFlow Lite Micro with multi-task CNN-LSTM model
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

/* Pins on the Seeed Studio XIAO are labeled incorrectly.
   Pin 'n' here refers to pin 'n-1' on the board. Pins 5 and 6 cannot be used since
   they are I2C pins between the ESP-32 Microcontroller and the OLED display */
const int buttonPin = 4;  // pushbutton pin
const int ledPin    = 3;  // LED pin

// ===== MODEL CONFIGURATION =====
// TODO: Update these based on your trained model metadata
constexpr int WINDOW_SIZE = 100;        // Number of IMU samples per window
constexpr int NUM_CHANNELS = 6;         // ax, ay, az, gx, gy, gz
constexpr int NUM_PHASE_CLASSES = 4;    // Updated to match your actual model
constexpr int NUM_POSTURE_CLASSES = 2;  // Updated to match your actual model

// Phase labels - MUST match model output order!
const char* phase_labels[NUM_PHASE_CLASSES] = {
    "moving-down",
    "moving-up",
    "not-in-pushup",
    "top"
};

// Posture labels - MUST match model output order!
const char* posture_labels[NUM_POSTURE_CLASSES] = {
    "good-form",
    "hips-sagging"
};

// ===== NORMALIZATION PARAMETERS =====
// TODO: Replace with values from norm_params.json generated during training
// These are computed from your training data: mean and std per channel
float imu_mean[NUM_CHANNELS] = { 0.5408033083196272,
        0.32900583367598685,
        0.49756270559210525,
        1.299851319710234,
        -0.6161175886499124,
        0.024165001748325792};
float imu_std[NUM_CHANNELS] = {0.4391443151852503,
        0.35146379602399636,
        0.2542886946241483,
        28.78840837535021,
        21.548818392843355,
        22.036605220632804};

// ===== IMU BUFFER =====
// Circular buffer for sliding window
constexpr int BUFFER_SIZE = WINDOW_SIZE;
float imu_buffer[BUFFER_SIZE][NUM_CHANNELS];  // [100][6]
int buffer_index = 0;
int samples_collected = 0;

// ===== TENSORFLOW LITE MICRO =====
constexpr int kTensorArenaSize = 120 * 1024;  // 120 KB for CNN-LSTM model
alignas(16) uint8_t tensor_arena[kTensorArenaSize];

const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;

// ===== BUTTON STATE =====
volatile bool buttonState = LOW;
bool lastButtonState = LOW;
bool inferenceEnabled = false;  // Changed from isRecording - controls inference
unsigned long lastOLEDUpdate = 0;
const unsigned long OLED_UPDATE_INTERVAL = 200; // ms

// ===== INFERENCE CONTROL =====
constexpr int INFERENCE_INTERVAL_MS = 1000;  // Run inference every 1 second when enabled
unsigned long lastInferenceTime = 0;

void IRAM_ATTR handleButtonInterrupt() {
    buttonState = digitalRead(buttonPin);
}

// ===== HELPER FUNCTIONS =====

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

    // Get output tensors (multi-task model has 2 outputs)
    TfLiteTensor* phase_output = interpreter->output(0);
    TfLiteTensor* posture_output = interpreter->output(1);

    // Dequantize phase predictions
    const float phase_scale = phase_output->params.scale;
    const int phase_zp = phase_output->params.zero_point;

    float phase_probs[NUM_PHASE_CLASSES];
    int best_phase = 0;
    float max_phase_prob = -1.0f;

    for (int i = 0; i < NUM_PHASE_CLASSES; i++) {
        float p = phase_scale * (static_cast<int>(phase_output->data.int8[i]) - phase_zp);
        phase_probs[i] = p;
        if (p > max_phase_prob) {
            max_phase_prob = p;
            best_phase = i;
        }
    }

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

    Serial.print("Phase: ");
    Serial.print(phase_labels[best_phase]);
    Serial.print(" (");
    Serial.print(max_phase_prob * 100, 1);
    Serial.println("%)");

    Serial.print("Posture: ");
    Serial.print(posture_labels[best_posture]);
    Serial.print(" (");
    Serial.print(max_posture_prob * 100, 1);
    Serial.println("%)");

    // Print all probabilities for debugging
    Serial.println("\nAll Phase Probabilities:");
    for (int i = 0; i < NUM_PHASE_CLASSES; i++) {
        if (phase_probs[i] > 0.05f) {  // Only show >5%
            Serial.print("  ");
            Serial.print(phase_labels[i]);
            Serial.print(": ");
            Serial.print(phase_probs[i] * 100, 1);
            Serial.println("%");
        }
    }

    Serial.println("\nAll Posture Probabilities:");
    for (int i = 0; i < NUM_POSTURE_CLASSES; i++) {
        if (posture_probs[i] > 0.05f) {
            Serial.print("  ");
            Serial.print(posture_labels[i]);
            Serial.print(": ");
            Serial.print(posture_probs[i] * 100, 1);
            Serial.println("%");
        }
    }
    Serial.println("================================\n");

    // Update OLED display with results
    oled_display_clear();
    oled_display_text(0, 0, "GAINS");

    char line1[32];
    snprintf(line1, sizeof(line1), "Ph: %s", phase_labels[best_phase]);
    oled_display_text(0, 16, line1);

    char line2[32];
    snprintf(line2, sizeof(line2), "    %.0f%%", max_phase_prob * 100);
    oled_display_text(0, 26, line2);

    char line3[32];
    snprintf(line3, sizeof(line3), "Po: %s", posture_labels[best_posture]);
    oled_display_text(0, 40, line3);

    char line4[32];
    snprintf(line4, sizeof(line4), "    %.0f%%", max_posture_prob * 100);
    oled_display_text(0, 50, line4);

    oled_display_update();
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

    Serial.println("\n========================================");
    Serial.println("  GAINS - Pushup Posture Classifier");
    Serial.println("========================================");
    Serial.flush();

    // Initialize OLED
    oled_display_init();
    oled_display_text(0, 10, "GAINS");
    oled_display_text(0, 30, "Initializing...");
    oled_display_update();

    // Initialize button and LED
    pinMode(ledPin, OUTPUT);
    pinMode(buttonPin, INPUT);
    attachInterrupt(digitalPinToInterrupt(buttonPin), handleButtonInterrupt, CHANGE);

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
    Serial.println("Press button or key to START inference");
    Serial.println("Press again to STOP inference");
    Serial.println("========================================\n");

    oled_display_clear();
    oled_display_text(0, 10, "GAINS");
    oled_display_text(0, 30, "Ready!");
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
    // Handle button state changes
    bool currentState = buttonState;
    digitalWrite(ledPin, currentState ? HIGH : LOW);

    if (currentState != lastButtonState) {
        lastButtonState = currentState;
        lastOLEDUpdate = millis();

        if (currentState == HIGH) {
            inferenceEnabled = !inferenceEnabled;

            if (inferenceEnabled) {
                Serial.println("\n[INFERENCE STARTED]");
                oled_display_clear();
                oled_display_text(0, 10, "GAINS");
                oled_display_text(0, 30, "Running...");
                oled_display_update();
            } else {
                Serial.println("\n[INFERENCE STOPPED]");
                oled_display_clear();
                oled_display_text(0, 10, "GAINS");
                oled_display_text(0, 30, "Stopped");
                oled_display_update();
            }
        }
    }

    // Handle serial input
    if (Serial.available() > 0) {
        char key = Serial.read();
        Serial.printf("Key pressed: '%c' (0x%02X)\n", key, key);

        inferenceEnabled = !inferenceEnabled;

        oled_display_clear();
        oled_display_text(0, 10, "GAINS");
        if (inferenceEnabled) {
            oled_display_text(0, 30, "Running...");
            Serial.println("[INFERENCE STARTED via serial]");
        } else {
            oled_display_text(0, 30, "Stopped");
            Serial.println("[INFERENCE STOPPED via serial]");
        }
        oled_display_update();
        lastOLEDUpdate = millis();
    }

    // Always read IMU data (keep buffer updated)
    float accel[3], gyro[3];
    if (ReadIMU(accel, gyro)) {
        // Store in circular buffer: [ax, ay, az, gx, gy, gz]
        imu_buffer[buffer_index][0] = accel[0];
        imu_buffer[buffer_index][1] = accel[1];
        imu_buffer[buffer_index][2] = accel[2];
        imu_buffer[buffer_index][3] = gyro[0];
        imu_buffer[buffer_index][4] = gyro[1];
        imu_buffer[buffer_index][5] = gyro[2];

        buffer_index = (buffer_index + 1) % BUFFER_SIZE;
        if (samples_collected < WINDOW_SIZE) {
            samples_collected++;
        }
    }

    // Run inference periodically ONLY when enabled
    if (inferenceEnabled) {
        unsigned long currentTime = millis();
        if (currentTime - lastInferenceTime >= INFERENCE_INTERVAL_MS) {
            lastInferenceTime = currentTime;
            RunInference();
        }
    }

    delay(10);  // ~100 Hz sampling rate
}
