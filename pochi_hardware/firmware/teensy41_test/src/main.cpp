#include <Arduino.h>

#include "can3.h"
#include "imu.h"
#include "usb_protocol.h"

namespace {

// A 20 Hz blink has a 50 ms full period, so the output toggles every 25 ms.
constexpr uint32_t kCommunicationBlinkHalfPeriodMs = 25;
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
}

void loop() {
  usb_loop();
  imu_loop();
  can3_loop();

  const uint32_t now_ms = millis();
  if (!can3_host_communication_alive()) {
    last_blink_ms = now_ms;
    if (led_on) {
      led_on = false;
      digitalWrite(LED_BUILTIN, LOW);
    }
  } else if (now_ms - last_blink_ms >= kCommunicationBlinkHalfPeriodMs) {
    last_blink_ms = now_ms;
    led_on = !led_on;
    digitalWrite(LED_BUILTIN, led_on ? HIGH : LOW);
  }
}
