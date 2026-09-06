#include <Arduino.h>
#include <FlexCAN_T4.h>

#include <cstdint>

namespace {

constexpr uint8_t kMasterId = 0xFD;
constexpr uint8_t kFirstMotorId = 0;
constexpr uint8_t kMotorCount = 12;
constexpr uint32_t kCanBaudRate = 1000000;
constexpr uint8_t kReadParameterType = 0x11;
constexpr uint8_t kStopMotorType = 0x04;
constexpr uint32_t kReplyTimeoutUs = 4000;
constexpr uint8_t kReadAttempts = 2;
constexpr uint32_t kSerialWaitMs = 30000;

struct RegisterRange {
  uint16_t first;
  uint16_t last;
};

constexpr RegisterRange kRegisterRanges[] = {
    {0x0000, 0x000F},
    {0x1000, 0x101F},
    {0x2000, 0x20FF},
    {0x3000, 0x30FF},
    {0x7000, 0x70FF},
};

using CanBus = FlexCAN_T4<CAN3, RX_SIZE_256, TX_SIZE_16>;
CanBus can;

uint32_t request_count = 0;
uint32_t response_count = 0;
uint32_t success_count = 0;

void drainCan() {
  CAN_message_t message;
  while (can.read(message)) {
  }
}

bool sendFrame(uint8_t type, uint8_t motor_id, uint16_t index = 0) {
  CAN_message_t message{};
  message.id = (static_cast<uint32_t>(type) << 24) |
               (static_cast<uint32_t>(kMasterId) << 8) | motor_id;
  message.flags.extended = 1;
  message.len = 8;
  message.buf[0] = static_cast<uint8_t>(index & 0xFFU);
  message.buf[1] = static_cast<uint8_t>(index >> 8);
  return can.write(message) == 1;
}

void stopAllMotors() {
  for (uint8_t pass = 0; pass < 2; ++pass) {
    for (uint8_t motor_id = kFirstMotorId;
         motor_id < kFirstMotorId + kMotorCount; ++motor_id) {
      sendFrame(kStopMotorType, motor_id);
      delay(2);
    }
    delay(10);
  }
  drainCan();
}

bool receiveParameter(uint8_t motor_id, uint16_t index,
                      CAN_message_t &reply, uint8_t &status) {
  const uint32_t started_us = micros();
  while (static_cast<uint32_t>(micros() - started_us) < kReplyTimeoutUs) {
    CAN_message_t message;
    if (!can.read(message) || !message.flags.extended || message.len < 8) {
      continue;
    }

    const uint8_t type = static_cast<uint8_t>((message.id >> 24) & 0x1FU);
    const uint8_t source = static_cast<uint8_t>((message.id >> 8) & 0xFFU);
    const uint8_t destination = static_cast<uint8_t>(message.id & 0xFFU);
    const uint16_t reply_index =
        static_cast<uint16_t>(message.buf[0]) |
        (static_cast<uint16_t>(message.buf[1]) << 8);
    if (type != kReadParameterType || source != motor_id ||
        destination != kMasterId || reply_index != index) {
      continue;
    }

    status = static_cast<uint8_t>((message.id >> 16) & 0xFFU);
    reply = message;
    return true;
  }
  return false;
}

void printRecord(uint8_t motor_id, uint16_t index, const char *result,
                 int status, const CAN_message_t *reply) {
  Serial.printf("P\t%u\t0x%04X\t%s\t%d\t",
                static_cast<unsigned int>(motor_id),
                static_cast<unsigned int>(index), result, status);
  if (reply == nullptr) {
    Serial.print("--------");
  } else {
    for (uint8_t i = 4; i < 8; ++i) {
      Serial.printf("%02X", reply->buf[i]);
    }
  }
  Serial.println();
}

void readParameter(uint8_t motor_id, uint16_t index) {
  ++request_count;
  for (uint8_t attempt = 0; attempt < kReadAttempts; ++attempt) {
    drainCan();
    if (!sendFrame(kReadParameterType, motor_id, index)) {
      delayMicroseconds(500);
      continue;
    }

    CAN_message_t reply{};
    uint8_t status = 0xFF;
    if (!receiveParameter(motor_id, index, reply, status)) {
      continue;
    }

    ++response_count;
    if (status == 0) {
      ++success_count;
    }
    printRecord(motor_id, index, status == 0 ? "ok" : "error", status,
                &reply);
    return;
  }
  printRecord(motor_id, index, "timeout", -1, nullptr);
}

void runDump() {
  Serial.println("DUMP_BEGIN\tversion=1\tmodel=RS02\tmotors=0-11");
  stopAllMotors();

  for (uint8_t motor_id = kFirstMotorId;
       motor_id < kFirstMotorId + kMotorCount; ++motor_id) {
    Serial.printf("MOTOR_BEGIN\t%u\n", static_cast<unsigned int>(motor_id));
    for (const RegisterRange &range : kRegisterRanges) {
      for (uint32_t index = range.first; index <= range.last; ++index) {
        readParameter(motor_id, static_cast<uint16_t>(index));
      }
    }
    Serial.printf("MOTOR_END\t%u\n", static_cast<unsigned int>(motor_id));
  }

  stopAllMotors();
  Serial.printf("DUMP_END\trequests=%lu\tresponses=%lu\tsuccess=%lu\n",
                static_cast<unsigned long>(request_count),
                static_cast<unsigned long>(response_count),
                static_cast<unsigned long>(success_count));
}

}  // namespace

void setup() {
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LED_BUILTIN, LOW);
  Serial.begin(115200);
  const uint32_t serial_started_ms = millis();
  while (!Serial && millis() - serial_started_ms < kSerialWaitMs) {
  }

  can.begin();
  can.setBaudRate(kCanBaudRate);
  can.setMaxMB(16);
  can.enableFIFO();
  delay(250);

  runDump();
  digitalWrite(LED_BUILTIN, HIGH);
}

void loop() {
  delay(1000);
}
