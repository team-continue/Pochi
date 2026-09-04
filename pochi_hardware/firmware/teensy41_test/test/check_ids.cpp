#include <Arduino.h>
#include <FlexCAN_T4.h>

#include "robostride.h"

namespace {

constexpr uint8_t kMasterId = 0xFD;
constexpr uint32_t kCanBaudRate = 1000000;
constexpr uint8_t kFirstScanId = 0x00;
constexpr uint8_t kLastScanId = 0xFE;
constexpr uint8_t kFirstExpectedId = 0x00;
constexpr uint8_t kExpectedMotorCount = 12;
constexpr uint8_t kLastExpectedId =
    static_cast<uint8_t>(kFirstExpectedId + kExpectedMotorCount - 1);
constexpr unsigned long kScanProbeTimeoutMs = 20;
constexpr unsigned long kExpectedIdRetryTimeoutMs = 200;
constexpr unsigned long kSerialWaitTimeoutMs = 10000;

static_assert(kLastExpectedId <= kLastScanId,
              "Expected motor ID range must fit in the scan range");

using CanBus = FlexCAN_T4<CAN3, RX_SIZE_256, TX_SIZE_16>;
using Motor = RoboStride<CAN3, RX_SIZE_256, TX_SIZE_16>;

CanBus can;
bool expected_id_found[kExpectedMotorCount] = {};
bool finished = false;
bool passed = false;

void printResult() {
  Serial.printf("Expected range: 0x%02X..0x%02X (%u IDs)\n",
                kFirstExpectedId, kLastExpectedId, kExpectedMotorCount);
  Serial.println(passed ? "RESULT: PASS" : "RESULT: FAIL");
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
  Serial.println("RoboStride CAN3 ID connectivity check (read only)");
  Serial.printf("Scanning IDs 0x%02X..0x%02X at %lu bps...\n",
                kFirstScanId, kLastScanId,
                static_cast<unsigned long>(kCanBaudRate));

  can.begin();
  can.setBaudRate(kCanBaudRate);
  can.setMaxMB(16);
  can.enableFIFO();
  delay(250);

  uint8_t responding_id_count = 0;
  for (uint16_t id = kFirstScanId; id <= kLastScanId; ++id) {
    Motor candidate(&can, kMasterId, static_cast<uint8_t>(id),
                    static_cast<int>(ActuatorType::ROBSTRIDE_03), 0.0f);
    if (!candidate.Probe(kScanProbeTimeoutMs)) {
      continue;
    }

    ++responding_id_count;
    Serial.printf("  Found ID 0x%02X (%u)%s\n",
                  static_cast<unsigned int>(id),
                  static_cast<unsigned int>(id),
                  (id >= kFirstExpectedId && id <= kLastExpectedId)
                      ? " [expected]"
                      : " [unexpected]");
    if (id >= kFirstExpectedId && id <= kLastExpectedId) {
      expected_id_found[id - kFirstExpectedId] = true;
    }
  }

  uint8_t missing_count = 0;
  for (uint8_t index = 0; index < kExpectedMotorCount; ++index) {
    if (expected_id_found[index]) {
      continue;
    }

    const uint8_t id = static_cast<uint8_t>(kFirstExpectedId + index);
    Motor candidate(&can, kMasterId, id,
                    static_cast<int>(ActuatorType::ROBSTRIDE_03), 0.0f);
    Serial.printf("  Retrying expected ID 0x%02X...\n",
                  static_cast<unsigned int>(id));
    if (candidate.Probe(kExpectedIdRetryTimeoutMs)) {
      expected_id_found[index] = true;
      ++responding_id_count;
      Serial.printf("  Found ID 0x%02X on retry [expected]\n",
                    static_cast<unsigned int>(id));
      continue;
    }

    ++missing_count;
    Serial.printf("  Missing expected ID 0x%02X\n",
                  static_cast<unsigned int>(id));
  }

  Serial.printf("Scan complete: %u ID(s) responded; %u expected ID(s) missing.\n",
                responding_id_count, missing_count);
  passed = missing_count == 0;
  finished = true;
  digitalWrite(LED_BUILTIN, passed ? HIGH : LOW);
  printResult();
}

void loop() {
  if (!finished) {
    return;
  }

  const unsigned long now_ms = millis();
  static unsigned long last_report_ms = 0;
  if (Serial && now_ms - last_report_ms >= 1000) {
    last_report_ms = now_ms;
    Serial.println("Live check of expected IDs 0x00..0x0B:");
    uint8_t live_found_count = 0;
    for (uint8_t id = kFirstExpectedId; id <= kLastExpectedId; ++id) {
      Motor candidate(&can, kMasterId, id,
                      static_cast<int>(ActuatorType::ROBSTRIDE_03), 0.0f);
      const bool found = candidate.Probe(kExpectedIdRetryTimeoutMs);
      live_found_count += found ? 1 : 0;
      Serial.printf("  ID 0x%02X: %s\n", static_cast<unsigned int>(id),
                    found ? "FOUND" : "MISSING");
    }
    Serial.printf("Live expected-ID count: %u/%u\n", live_found_count,
                  kExpectedMotorCount);
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
