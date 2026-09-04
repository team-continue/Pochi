#include <Arduino.h>

#include "can3.h"
#include "imu.h"
#include "usb_protocol.h"

namespace {

constexpr uint32_t kStatusBlinkIntervalMs = 250;
uint32_t last_blink_ms = 0;
bool led_on = false;

}  // namespace

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);

  Serial.begin(115200);
  can3_init();
  imu_init();
  usb_init();

  // Torque is still OFF here; steady ON only means all encoders are visible.
  led_on = true;
  digitalWrite(LED_BUILTIN, HIGH);
}

void loop() {
  usb_loop();
  imu_loop();
  can3_loop();

  const uint32_t now_ms = millis();
  if (can3_connected_count() == CAN3_MOTOR_COUNT) {
    led_on = true;
    digitalWrite(LED_BUILTIN, HIGH);
  } else if (now_ms - last_blink_ms >= kStatusBlinkIntervalMs) {
    last_blink_ms = now_ms;
    led_on = !led_on;
    digitalWrite(LED_BUILTIN, led_on ? HIGH : LOW);
  }

}
