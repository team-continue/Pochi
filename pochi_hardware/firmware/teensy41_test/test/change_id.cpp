#include <Arduino.h>
#include <FlexCAN_T4.h>

#include "robostride.h"

namespace {

// Edit these two values before copying this file over src/main.cpp.
constexpr uint8_t kOldMotorId = 0x05;
constexpr uint8_t kNewMotorId = 0x08;
constexpr uint8_t kMasterId = 0xFD;
constexpr uint32_t kCanBaudRate = 1000000;
constexpr uint8_t kFirstScanId = 0x00;
constexpr uint8_t kLastScanId = 0xFE;
constexpr unsigned long kScanProbeTimeoutMs = 10;
constexpr unsigned long kProbeTimeoutMs = 100;
constexpr unsigned long kSaveTimeoutMs = 1000;
constexpr unsigned long kSerialWaitTimeoutMs = 10000;

static_assert(kOldMotorId >= 1 && kOldMotorId <= 0xFE,
              "OLD motor ID must be in 1..0xFE");
static_assert(kNewMotorId <= 20,
              "NEW motor ID must be in the RS03 persistent range 0..20");
static_assert(kOldMotorId != kNewMotorId,
              "OLD and NEW motor IDs must differ");

using CanBus = FlexCAN_T4<CAN3, RX_SIZE_256, TX_SIZE_16>;
using Motor = RoboStride<CAN3, RX_SIZE_256, TX_SIZE_16>;

CanBus can;
Motor target(&can, kMasterId, kOldMotorId,
             static_cast<int>(ActuatorType::ROBSTRIDE_03), 0.0f);
Motor old_id_check(&can, kMasterId, kOldMotorId,
                   static_cast<int>(ActuatorType::ROBSTRIDE_03), 0.0f);
Motor new_id_check(&can, kMasterId, kNewMotorId,
                   static_cast<int>(ActuatorType::ROBSTRIDE_03), 0.0f);

bool finished = false;
bool passed = false;
const char *final_message = nullptr;

void finish(bool success, const char *message) {
  passed = success;
  finished = true;
  final_message = message;
  digitalWrite(LED_BUILTIN, success ? HIGH : LOW);
  Serial.println(message);
  Serial.println(success ? "RESULT: PASS" : "RESULT: FAIL");
  Serial.println("The test will not run again until the Teensy is reset.");
}

uint8_t scanMotorIds() {
  uint8_t found_count = 0;

  Serial.printf("Scanning CAN3 motor IDs 0x%02X..0x%02X at %lu bps...\n",
                kFirstScanId, kLastScanId,
                static_cast<unsigned long>(kCanBaudRate));
  for (uint16_t id = kFirstScanId; id <= kLastScanId; ++id) {
    Motor candidate(&can, kMasterId, static_cast<uint8_t>(id),
                    static_cast<int>(ActuatorType::ROBSTRIDE_03), 0.0f);
    if (candidate.Probe(kScanProbeTimeoutMs)) {
      ++found_count;
      Serial.printf("  Found motor at ID 0x%02X (%u)\n",
                    static_cast<unsigned int>(id),
                    static_cast<unsigned int>(id));
    }
  }

  if (found_count == 0) {
    Serial.println("  No motor IDs responded on CAN3.");
  } else {
    Serial.printf("CAN3 scan complete: %u ID(s) responded.\n", found_count);
  }
  return found_count;
}

}  // namespace

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  Serial.begin(115200);
  const unsigned long serial_wait_started_ms = millis();
  while (!Serial && millis() - serial_wait_started_ms < kSerialWaitTimeoutMs) {
  }
  delay(250);

  Serial.println();
  Serial.println("RoboStride 03 CAN ID changer (private protocol)");
  Serial.printf("Requested change: 0x%02X -> 0x%02X\n",
                kOldMotorId, kNewMotorId);
  Serial.println("WARNING: Connect exactly one target motor to this CAN bus.");
  Serial.println("The motor must be using the private 29-bit CAN protocol.");

  can.begin();
  can.setBaudRate(kCanBaudRate);
  can.setMaxMB(16);
  can.enableFIFO();
  delay(250);

  scanMotorIds();

  Serial.printf("Checking old ID 0x%02X...\n", kOldMotorId);
  if (!target.Probe(kProbeTimeoutMs)) {
    Serial.println("Old ID did not respond. Checking whether it was already changed...");
    if (new_id_check.Probe(kProbeTimeoutMs)) {
      Serial.println("New ID responds. Saving the current motor settings...");
      if (!new_id_check.Save_Parameters(kSaveTimeoutMs)) {
        Serial.println("WARNING: Save command was not acknowledged; power-cycle verification is required.");
      } else {
        Serial.println("Motor acknowledged the save command.");
      }
      finish(true, "Old ID is absent and the new ID responds; save command was sent.");
      return;
    }
    finish(false, "Neither old nor new ID responded; no change was attempted.");
    return;
  }
  Serial.println("Old ID responded.");

  Serial.printf("Checking that new ID 0x%02X is unused...\n", kNewMotorId);
  if (new_id_check.Probe(kProbeTimeoutMs)) {
    finish(false, "New ID already responded; refusing to create an ID collision.");
    return;
  }
  Serial.println("New ID is unused.");

  Serial.println("Disabling the motor and changing its ID...");
  if (!target.Set_CAN_ID(kNewMotorId, kProbeTimeoutMs)) {
    finish(false,
           "The ID-change command could not be verified. Check both IDs before retrying.");
    return;
  }

  Serial.println("Saving motor settings to non-volatile storage...");
  if (!target.Save_Parameters(kSaveTimeoutMs)) {
    Serial.println("WARNING: Save command was not acknowledged; power-cycle verification is required.");
  } else {
    Serial.println("Motor acknowledged the save command.");
  }

  delay(20);
  Serial.printf("Verifying new ID 0x%02X...\n", kNewMotorId);
  if (!target.Probe(kProbeTimeoutMs)) {
    finish(false, "New ID did not respond after the change.");
    return;
  }

  Serial.printf("Verifying old ID 0x%02X no longer responds...\n", kOldMotorId);
  if (old_id_check.Probe(kProbeTimeoutMs)) {
    finish(false,
           "Old ID still responds. More than one motor may be connected with the old ID.");
    return;
  }

  finish(true, "CAN ID change and connectivity checks completed successfully.");
}

void loop() {
  // A successful run leaves the LED steadily on.  A failure blinks the LED so
  // it is visible even when the USB serial monitor is not open.
  if (!finished) {
    return;
  }

  const unsigned long now_ms = millis();
  static unsigned long last_report_ms = 0;
  if (Serial && now_ms - last_report_ms >= 1000) {
    last_report_ms = now_ms;
    Serial.println(final_message);
    Serial.println(passed ? "RESULT: PASS" : "RESULT: FAIL");
  }

  if (passed) {
    return;
  }
  static unsigned long last_blink_ms = 0;
  static bool led_on = false;
  if (now_ms - last_blink_ms >= 250) {
    last_blink_ms = now_ms;
    led_on = !led_on;
    digitalWrite(LED_BUILTIN, led_on ? HIGH : LOW);
  }
}
