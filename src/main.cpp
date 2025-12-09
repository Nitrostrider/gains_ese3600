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

// Note definitions for the speaker
#define NOTE_C4 262
#define NOTE_D4 294
#define NOTE_E4 330
#define NOTE_F4 349
#define NOTE_G4 392
#define NOTE_A4 440
#define NOTE_AS4 466
#define NOTE_C5 523
#define NOTE_E5 659

/* Pins on the Seeed Studio XIAO are labeled incorrectly.
   Pin 'n' here refers to pin 'n-1' on the board. Pins 5 and 6 (4 and 5 on the board) cannot be used since
   they are I2C pins between the ESP-32 Microcontroller and the OLED display */

const int BUZZER_PIN = 2;  // Buzzer pin
const int BUTTON_PIN = 3;  // pushbutton pin
const int RECORDING_LED_PIN = 4;  // LED pin

// LOW = false; HIGH = true
volatile bool buttonState = LOW;     // updated in ISR
bool lastButtonState = LOW;          // tracks last state for change detection
bool inferenceEnabled = false;        // recording state (aka inferenceEnabled)
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
constexpr int NUM_PHASE_CLASSES = 3;    // 3 phase types

// Posture labels - MUST match model output order!
// From pushup_model_metadata.json: ["good-form", "hips-high", "hips-sagging", "partial-rom"]
const char* posture_labels[NUM_POSTURE_CLASSES] = {
    "good-form",    // Class 0
    "hips-high",    // Class 1
    "hips-sagging", // Class 2
    "partial-rom"   // Class 3
};

// Phase labels - MUST match model output order!
// Multi-task model output: ["at-top", "moving", "at-bottom"]
const char* phase_labels[NUM_PHASE_CLASSES] = {
    "at-top",      // Class 0
    "moving",      // Class 1
    "at-bottom"    // Class 2
};

// ===== REP COUNTING STATE MACHINE =====
enum PushupState {
    STATE_IDLE = 0,
    STATE_AT_TOP,
    STATE_DESCENDING,
    STATE_AT_BOTTOM,
    STATE_ASCENDING
};

struct RepCounter {
    int total_reps;
    int good_form_reps;
    int poor_form_reps;
    PushupState state;
    int phase_confirm_count;      // Hysteresis counter
    int good_form_count;           // Form tracking during rep
    int poor_form_count;
    unsigned long state_enter_time;  // Timeout tracking
};

RepCounter rep_counter = {0, 0, 0, STATE_IDLE, 0, 0, 0, 0};

const unsigned long STATE_TIMEOUT_MS = 10000;  // 10 seconds

// ===== PUSHUP CYCLE TRACKING FOR PREDICTION AGGREGATION =====
enum PushupCycleState {
    CYCLE_IDLE,          // No pushup in progress
    CYCLE_IN_PROGRESS,   // Pushup started
    CYCLE_DESCENDING,    // Going down
    CYCLE_AT_BOTTOM,     // At bottom
    CYCLE_ASCENDING      // Coming back up
};

struct PredictionRecord {
    uint8_t posture_class;    // 0-3 (good-form, hips-high, hips-sagging, partial-rom)
    uint8_t phase_class;      // 0-2 (at-top, moving, at-bottom)
    uint32_t timestamp_ms;    // For debugging
};

constexpr int MAX_PREDICTIONS_PER_CYCLE = 6;

struct PushupCycleTracker {
    PushupCycleState cycle_state;
    PredictionRecord predictions[MAX_PREDICTIONS_PER_CYCLE];
    uint8_t prediction_count;
    uint32_t cycle_start_time;
    uint32_t cycle_end_time;
};

PushupCycleTracker cycle_tracker = {CYCLE_IDLE, {}, 0, 0, 0};
constexpr unsigned long CYCLE_TIMEOUT_MS = 10000;  // 10 seconds

// ===== NORMALIZATION PARAMETERS =====
// Replace with values from pushup_model_metadata.json generated during training
// These are computed from your training data: mean and std per channel
float imu_mean[NUM_CHANNELS] = {
    0.21002981838521975,
    -0.012597600868383742,
    0.9683456599734906,
    0.17634020483573734,
    1.1392008499625985,
    0.9496813799078279
};
float imu_std[NUM_CHANNELS] = {0.2788689179258505,
    0.11287905247385777,
    0.28730744260394775,
    13.642031321723314,
    36.710716288881166,
    6.244813331362449};

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
constexpr int INFERENCE_INTERVAL_MS = 1000;  // Run inference every 1 second when enabled
unsigned long lastInferenceTime = 0;

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

int GetBestClass(TfLiteTensor* output, int num_classes) {
    // Dequantize and find class with highest probability
    const float scale = output->params.scale;
    const int zp = output->params.zero_point;

    int best_class = 0;
    float max_prob = -1.0f;

    for (int i = 0; i < num_classes; i++) {
        float prob = scale * (static_cast<int>(output->data.int8[i]) - zp);
        if (prob > max_prob) {
            max_prob = prob;
            best_class = i;
        }
    }

    return best_class;
}

int AggregatePredictions(PushupCycleTracker* tracker) {
    // Require at least 2 predictions to aggregate
    if (tracker->prediction_count < 2) {
        Serial.println("WARNING: Too few predictions, using most recent");
        return tracker->predictions[tracker->prediction_count - 1].posture_class;
    }

    // Count votes for each posture class
    int votes[NUM_POSTURE_CLASSES] = {0, 0, 0, 0};

    // Only count predictions from meaningful phases
    for (int i = 0; i < tracker->prediction_count; i++) {
        PredictionRecord* pred = &tracker->predictions[i];

        // Filter out "at-top" predictions (less informative)
        // Focus on "moving" and "at-bottom" predictions
        if (pred->phase_class != 0) {  // Not at-top
            votes[pred->posture_class]++;
        }
    }

    // Find majority
    int best_class = 0;
    int max_votes = votes[0];
    for (int i = 1; i < NUM_POSTURE_CLASSES; i++) {
        if (votes[i] > max_votes) {
            max_votes = votes[i];
            best_class = i;
        }
    }

    return best_class;
}

void CompleteRepWithAggregation(int aggregated_posture) {
    rep_counter.total_reps++;

    // Classify based on AGGREGATED prediction
    bool is_good_form = (aggregated_posture == 0);  // Class 0 = good-form

    if (is_good_form) {
        rep_counter.good_form_reps++;
    } else {
        rep_counter.poor_form_reps++;
    }

    // Feed watchdog to prevent timeout
    esp_task_wdt_reset();

    // Concise serial output
    Serial.printf("REP #%d | %s (%d preds, %lums) | Total:%d Good:%d Poor:%d\n",
                  rep_counter.total_reps,
                  posture_labels[aggregated_posture],
                  cycle_tracker.prediction_count,
                  cycle_tracker.cycle_end_time - cycle_tracker.cycle_start_time,
                  rep_counter.total_reps,
                  rep_counter.good_form_reps,
                  rep_counter.poor_form_reps);

    // Reset form counters for next rep
    rep_counter.good_form_count = 0;
    rep_counter.poor_form_count = 0;
}

void UpdateRepCounter(int phase, int posture) {
    // Track posture quality during rep
    if (posture == 0) {  // good-form
        rep_counter.good_form_count++;
    } else {
        rep_counter.poor_form_count++;
    }

    // Buffer prediction during cycle with overflow protection
    if (cycle_tracker.cycle_state != CYCLE_IDLE) {
        if (cycle_tracker.prediction_count < MAX_PREDICTIONS_PER_CYCLE) {
            PredictionRecord* pred = &cycle_tracker.predictions[cycle_tracker.prediction_count++];
            pred->posture_class = posture;
            pred->phase_class = phase;
            pred->timestamp_ms = millis();
        } else {
            // Buffer full - shift left to keep most recent
            for (int i = 0; i < MAX_PREDICTIONS_PER_CYCLE - 1; i++) {
                cycle_tracker.predictions[i] = cycle_tracker.predictions[i + 1];
            }
            PredictionRecord* pred = &cycle_tracker.predictions[MAX_PREDICTIONS_PER_CYCLE - 1];
            pred->posture_class = posture;
            pred->phase_class = phase;
            pred->timestamp_ms = millis();
        }
    }

    // State transitions with hysteresis
    PushupState old_state = rep_counter.state;

    switch (rep_counter.state) {
        case STATE_IDLE:
            if (phase == 0) {  // at-top
                rep_counter.state = STATE_AT_TOP;
            }
            break;

        case STATE_AT_TOP:
            if (phase == 1) {  // moving
                rep_counter.phase_confirm_count++;
                if (rep_counter.phase_confirm_count >= 2) {
                    rep_counter.state = STATE_DESCENDING;
                    rep_counter.phase_confirm_count = 0;

                    // Start cycle tracking
                    cycle_tracker.cycle_state = CYCLE_IN_PROGRESS;
                    cycle_tracker.prediction_count = 0;
                    cycle_tracker.cycle_start_time = millis();
                }
            } else {
                rep_counter.phase_confirm_count = 0;
            }
            break;

        case STATE_DESCENDING:
            if (phase == 2) {  // at-bottom
                rep_counter.phase_confirm_count++;
                if (rep_counter.phase_confirm_count >= 2) {
                    rep_counter.state = STATE_AT_BOTTOM;
                    rep_counter.phase_confirm_count = 0;
                }
            } else {
                rep_counter.phase_confirm_count = 0;
            }
            break;

        case STATE_AT_BOTTOM:
            if (phase == 1) {  // moving
                rep_counter.phase_confirm_count++;
                if (rep_counter.phase_confirm_count >= 2) {
                    rep_counter.state = STATE_ASCENDING;
                    rep_counter.phase_confirm_count = 0;
                }
            } else {
                rep_counter.phase_confirm_count = 0;
            }
            break;

        case STATE_ASCENDING:
            if (phase == 0) {  // at-top
                rep_counter.phase_confirm_count++;
                if (rep_counter.phase_confirm_count >= 2) {
                    // End cycle and aggregate
                    cycle_tracker.cycle_end_time = millis();
                    int final_posture = AggregatePredictions(&cycle_tracker);

                    // Complete rep with aggregated prediction
                    CompleteRepWithAggregation(final_posture);

                    // Reset cycle tracker
                    cycle_tracker.cycle_state = CYCLE_IDLE;
                    cycle_tracker.prediction_count = 0;

                    rep_counter.state = STATE_AT_TOP;
                    rep_counter.phase_confirm_count = 0;
                }
            } else {
                rep_counter.phase_confirm_count = 0;
            }
            break;
    }

    // Track state entry time for timeout detection
    // Update cycle state based on pushup state
    if (rep_counter.state != old_state) {
        rep_counter.state_enter_time = millis();

        // Print state transition
        const char* state_names[] = {"IDLE", "AT_TOP", "DESCENDING", "AT_BOTTOM", "ASCENDING"};
        Serial.printf("State: %s -> %s | Reps: %d\n",
                     state_names[old_state],
                     state_names[rep_counter.state],
                     rep_counter.total_reps);

        switch (rep_counter.state) {
            case STATE_DESCENDING:
                cycle_tracker.cycle_state = CYCLE_DESCENDING;
                break;
            case STATE_AT_BOTTOM:
                cycle_tracker.cycle_state = CYCLE_AT_BOTTOM;
                break;
            case STATE_ASCENDING:
                cycle_tracker.cycle_state = CYCLE_ASCENDING;
                break;
            default:
                break;
        }
    }
}

// Statistical phase detection based on Z-acceleration patterns
int DetectPhase() {
    // Phase detection thresholds (from auto_label_phases.py)
    const float VALLEY_MAX = 0.70f;      // Moving phase (low az)
    const float PLATEAU_MIN = 0.72f;     // Top plateau
    const float PLATEAU_MAX = 1.10f;     // Top plateau
    const float PEAK_MIN = 1.12f;        // Bottom peak (high az)
    const float STABLE_STD_MAX = 0.10f;  // Low variance = stable
    const float STATIC_GY_MAX = 15.0f;   // Low gyro = static

    // Calculate statistics from current window
    float az_sum = 0.0f;
    float gy_sum = 0.0f;

    for (int i = 0; i < WINDOW_SIZE; i++) {
        int buf_idx = (buffer_index - WINDOW_SIZE + i + BUFFER_SIZE) % BUFFER_SIZE;
        az_sum += imu_buffer[buf_idx][2];  // az is channel 2
        gy_sum += fabsf(imu_buffer[buf_idx][4]);  // |gy| is channel 4
    }

    float az_mean = az_sum / WINDOW_SIZE;
    float gy_max_abs = gy_sum / WINDOW_SIZE;  // Approximation

    // Calculate variance for az
    float az_var_sum = 0.0f;
    for (int i = 0; i < WINDOW_SIZE; i++) {
        int buf_idx = (buffer_index - WINDOW_SIZE + i + BUFFER_SIZE) % BUFFER_SIZE;
        float diff = imu_buffer[buf_idx][2] - az_mean;
        az_var_sum += diff * diff;
    }
    float az_std = sqrtf(az_var_sum / WINDOW_SIZE);

    // Phase detection logic (from Python auto_label_phase)
    // RULE 1: Clear valley = moving
    if (az_mean < VALLEY_MAX) {
        return 1;  // moving
    }

    // RULE 2: Clear peak = at-bottom
    if (az_mean >= PEAK_MIN) {
        return 2;  // at-bottom
    }

    // RULE 3: High gyro activity = moving
    if (gy_max_abs > 20.0f) {
        return 1;  // moving
    }

    // RULE 4: Stable plateau = at-top
    if (az_mean >= PLATEAU_MIN && az_mean <= PLATEAU_MAX &&
        az_std < STABLE_STD_MAX && gy_max_abs < STATIC_GY_MAX) {
        return 0;  // at-top
    }

    // RULE 5: High variance = moving
    if (az_std > 0.15f) {
        return 1;  // moving
    }

    // Default: at-top if in plateau range, otherwise moving
    if (az_mean >= PLATEAU_MIN && az_mean <= PLATEAU_MAX) {
        return 0;  // at-top
    }

    return 1;  // moving (default)
}

void DisplayRepCount() {
    // Feed watchdog before display update
    esp_task_wdt_reset();

    oled_display_clear();
    oled_display_text(0, 0, "PUSHUP TRACKER");

    char rep_line[32];
    snprintf(rep_line, sizeof(rep_line), "Total: %d", rep_counter.total_reps);
    oled_display_text(0, 16, rep_line);

    char form_line[32];
    snprintf(form_line, sizeof(form_line), "Good:%d Poor:%d",
             rep_counter.good_form_reps,
             rep_counter.poor_form_reps);
    oled_display_text(0, 32, form_line);

    // Show cycle progress
    char cycle_line[32];
    if (cycle_tracker.cycle_state != CYCLE_IDLE) {
        const char* cycle_names[] = {"IDLE", "ACTIVE", "DOWN", "BOTTOM", "UP"};
        snprintf(cycle_line, sizeof(cycle_line), "%s [%d pred]",
                 cycle_names[cycle_tracker.cycle_state],
                 cycle_tracker.prediction_count);
    } else {
        const char* state_names[] = {"IDLE", "TOP", "DOWN", "BOTTOM", "UP"};
        snprintf(cycle_line, sizeof(cycle_line), "%s", state_names[rep_counter.state]);
    }
    oled_display_text(0, 48, cycle_line);

    oled_display_update();
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
    TfLiteStatus invoke_status = interpreter->Invoke();

    if (invoke_status != kTfLiteOk) {
        Serial.println("ERROR: Inference failed!");
        return;
    }

    // Feed watchdog again after inference
    esp_task_wdt_reset();

    // Get output tensor (single-output posture model)
    TfLiteTensor* posture_output = interpreter->output(0);  // Posture: [4 classes]

    // Get best posture prediction
    int best_posture = GetBestClass(posture_output, NUM_POSTURE_CLASSES);

    // Detect phase using statistical analysis of IMU data
    int best_phase = DetectPhase();

    // Update state machine with predictions
    UpdateRepCounter(best_phase, best_posture);

    // Update OLED display with rep counter
    DisplayRepCount();
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
    pinMode(BUTTON_PIN, INPUT);
    pinMode(RECORDING_LED_PIN, OUTPUT);

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

    // Check number of outputs
    int num_outputs = interpreter->outputs_size();
    Serial.printf("Number of outputs: %d\n", num_outputs);
    Serial.println("Using statistical phase detection");

    Serial.println("========================================");
    Serial.println("System ready!");
    Serial.println("Press button or key to START inference");
    Serial.println("Press again to STOP inference");
    Serial.println("========================================\n");

    oled_display_clear();
    oled_display_text(0, 10, "GAINS");
    oled_display_text(0, 30, "Press button to start recording.");
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

    // Only update OLED if the button state has changed
    if (currentState != lastButtonState) {
        lastButtonState = currentState;
        lastOLEDUpdate = millis();  // reset OLED timer

        if (currentState == HIGH) {
            // Toggle inference state
            inferenceEnabled = !inferenceEnabled;

            if (inferenceEnabled) {
                Serial.println("Started");
                oled_display_clear();
                oled_display_text(0, 10, "GAINS");
                oled_display_text(0, 30, "Started");
                oled_display_update();
                digitalWrite(RECORDING_LED_PIN, HIGH);
            } else {
                Serial.println("Stopped");
                oled_display_clear();
                oled_display_text(0, 10, "GAINS");
                oled_display_text(0, 30, "Stopped");
                oled_display_update();
                digitalWrite(RECORDING_LED_PIN, LOW);
            }
        }
    }

    // Handle serial input
    if (Serial.available() > 0) {
        char key = Serial.read();
        while (Serial.available() > 0) {
            Serial.read();
        }

        inferenceEnabled = !inferenceEnabled;

        oled_display_clear();
        oled_display_text(0, 10, "GAINS");
        if (inferenceEnabled) {
            oled_display_text(0, 30, "Running");
            Serial.println("Started");
        } else {
            oled_display_text(0, 30, "Stopped");
            Serial.println("Stopped");
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

        // State machine timeout - handle stuck states
        if (rep_counter.state != STATE_IDLE &&
            rep_counter.state != STATE_AT_TOP &&
            currentTime - rep_counter.state_enter_time > STATE_TIMEOUT_MS) {

            // If stuck in ASCENDING state, likely finished the pushup - complete it
            if (rep_counter.state == STATE_ASCENDING && cycle_tracker.prediction_count >= 2) {
                Serial.println("Timeout in ASCENDING - completing rep");
                cycle_tracker.cycle_end_time = millis();
                int final_posture = AggregatePredictions(&cycle_tracker);
                CompleteRepWithAggregation(final_posture);
                cycle_tracker.cycle_state = CYCLE_IDLE;
                cycle_tracker.prediction_count = 0;
                rep_counter.state = STATE_AT_TOP;
                rep_counter.phase_confirm_count = 0;
            } else {
                // For other stuck states, reset
                Serial.printf("Timeout in %s - resetting\n",
                    rep_counter.state == STATE_DESCENDING ? "DESCENDING" :
                    rep_counter.state == STATE_AT_BOTTOM ? "AT_BOTTOM" : "UNKNOWN");
                rep_counter.state = STATE_IDLE;
                rep_counter.phase_confirm_count = 0;
                cycle_tracker.cycle_state = CYCLE_IDLE;
                cycle_tracker.prediction_count = 0;
            }
            rep_counter.state_enter_time = currentTime;
        }

        // Check for cycle timeout (incomplete pushup)
        if (cycle_tracker.cycle_state != CYCLE_IDLE) {
            unsigned long cycle_duration = currentTime - cycle_tracker.cycle_start_time;

            if (cycle_duration > CYCLE_TIMEOUT_MS) {
                Serial.println("CYCLE TIMEOUT - Resetting");

                // Reset cycle without counting rep
                cycle_tracker.cycle_state = CYCLE_IDLE;
                cycle_tracker.prediction_count = 0;
                rep_counter.state = STATE_IDLE;
                rep_counter.phase_confirm_count = 0;
                rep_counter.good_form_count = 0;
                rep_counter.poor_form_count = 0;
            }
        }
    }

    delay(10);  // ~100 Hz sampling rate
}
