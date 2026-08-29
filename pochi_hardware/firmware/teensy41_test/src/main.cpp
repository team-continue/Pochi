#include <Arduino.h>

#include "can3.h"
#include "imu.h"
#include "usb_protocol.h"

namespace {

constexpr uint32_t kCommunicationBlinkIntervalMs = 100;
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

  // Initialization complete: steady ON until valid USB commands arrive.
  led_on = true;
  digitalWrite(LED_BUILTIN, HIGH);
}

void loop() {
  usb_loop();
  imu_loop();
  can3_loop();

  const uint32_t now_ms = millis();
  if (!can3_command_alive()) {
    led_on = true;
    digitalWrite(LED_BUILTIN, HIGH);
  } else if (now_ms - last_blink_ms >= kCommunicationBlinkIntervalMs) {
    last_blink_ms = now_ms;
    led_on = !led_on;
    digitalWrite(LED_BUILTIN, led_on ? HIGH : LOW);
  }

}
