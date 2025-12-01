// #include <Arduino.h>

// /* IMPORTANT: The pins on the Seeed Studio XIAO are labeled incorrectly.
// Pin 'n' in this file refer to pin 'n-1' on the board. */
// /* OBSERVATIONS: Pin 7 in this file doesn't work for blinking LED. Pins 5 and 6 work fine. */
// const int buttonPin = 6;     // pushbutton pin
// const int ledPin = 5;       // LED pin

// int buttonState = 0;

// void setup() {
//   Serial.begin(115200);

//   pinMode(ledPin, OUTPUT);
//   pinMode(buttonPin, INPUT);
// }

// void loop() {
//   // Example: blink LED and print state
//   buttonState = digitalRead(buttonPin);
//   Serial.printf("Button state is %s (pin %d)\n", buttonState == HIGH ? "HIGH" : "LOW", buttonPin);

//   if (buttonState == HIGH) {
//     digitalWrite(ledPin, HIGH);
//     Serial.printf("LED state should be HIGH (pin %d)\n", ledPin);
//   } else {
//     digitalWrite(ledPin, LOW);
//     Serial.printf("LED state should be LOW (pin %d)\n", ledPin);
//   }
// }

// #include <Arduino.h>
// // #include "oled_display.h"

// /* IMPORTANT: The pins on the Seeed Studio XIAO are labeled incorrectly.
// Pin 'n' in this file refer to pin 'n-1' on the board. */

// const int buttonPin = 6;     // pushbutton pin
// const int ledPin = 5;        // LED pin

// volatile bool buttonState = LOW;  // volatile because it's modified in an ISR

// void IRAM_ATTR handleButtonInterrupt() {
//     // Read quickly inside ISR — digitalRead is okay on SAMD21
//     buttonState = digitalRead(buttonPin);
// }

// void setup() {
//   Serial.begin(115200);
//   // oled_display_init();

//   pinMode(ledPin, OUTPUT);
//   pinMode(buttonPin, INPUT);

//   // Attach interrupt on button pin (captures both press & release)
//   attachInterrupt(digitalPinToInterrupt(buttonPin), handleButtonInterrupt, CHANGE);
// }

// void loop() {

//   // Handle LED based on interrupt-updated state
//   if (buttonState == HIGH) {
//     digitalWrite(ledPin, HIGH);
//     Serial.printf("LED state is HIGH (pin %d)\n", ledPin);
//     // oled_display_clear();
//     // oled_display_text(0, 10, "GAINS");
//     // oled_display_text(0, 30, "Start Recording!");
//     // oled_display_update();
//   } else {
//     digitalWrite(ledPin, LOW);
//     Serial.printf("LED state is LOW (pin %d)\n", ledPin);
//   }

//   Serial.printf("Button state is %s (pin %d)\n",
//                 buttonState == HIGH ? "HIGH" : "LOW",
//                 buttonPin);

//   delay(200);  // throttle prints
// }

#include <Arduino.h>
#include "oled_display.h"   // uncommented

/* Pins on the Seeed Studio XIAO are labeled incorrectly.
   Pin 'n' here refers to pin 'n-1' on the board. Pins 5 and 6 cannot be used since
   they are I2C pins between the ESP-32 Microcontroller and the OLED display */
const int buttonPin = 4;  // pushbutton pin
const int ledPin    = 3;  // LED pin

volatile bool buttonState = LOW;     // updated in ISR
bool lastButtonState = LOW;          // tracks last state for change detection
bool isRecording = false;        // recording state
unsigned long lastOLEDUpdate = 0;    // timestamp for OLED throttling
const unsigned long OLED_UPDATE_INTERVAL = 200; // ms

void IRAM_ATTR handleButtonInterrupt() {
    // Read button quickly in ISR
    buttonState = digitalRead(buttonPin);
}

void setup() {
    Serial.begin(115200);
    oled_display_init();   // initialize OLED
    oled_display_text(0, 10, "GAINS");
    oled_display_text(0, 30, "Press button to start recording.");
    oled_display_update();

    pinMode(ledPin, OUTPUT);
    pinMode(buttonPin, INPUT);

    attachInterrupt(digitalPinToInterrupt(buttonPin), handleButtonInterrupt, CHANGE);
}

void loop() {
    // Read the current button state atomically
    bool currentState = buttonState;

    // Update LED immediately based on button state
    digitalWrite(ledPin, currentState ? HIGH : LOW);

    // Print button/LED state for debugging
    Serial.printf("Button: %s, LED: %s\n",
                  currentState ? "HIGH" : "LOW",
                  currentState ? "ON" : "OFF");

    // Only update OLED if the button state has changed
    if (currentState != lastButtonState) {
        lastButtonState = currentState;
        lastOLEDUpdate = millis();  // reset OLED timer

        if (currentState == HIGH) {
            if (!isRecording) {
                isRecording = true;
                // OLED update for button pressed
                oled_display_clear();
                oled_display_text(0, 10, "GAINS");
                oled_display_text(0, 30, "Started recording. Press button to stop.");
                oled_display_update();
            } else {
                isRecording = false;
                // OLED update for stopping recording
                oled_display_clear();
                oled_display_text(0, 10, "GAINS");
                oled_display_text(0, 30, "Stopped Recording. Press button to start again.");
                oled_display_update();
            }
        }
    }

    // Optional: throttle OLED updates if the button is held down
    if (currentState == HIGH && millis() - lastOLEDUpdate >= OLED_UPDATE_INTERVAL) {
        lastOLEDUpdate = millis();
        oled_display_update(); // periodic refresh
    }

    delay(20); // small delay to reduce CPU usage and debounce slightly
}