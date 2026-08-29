#pragma once

#include <Arduino.h>
#include <FlexCAN_T4.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "robostride.h"

constexpr uint8_t CAN3_FIRST_MOTOR_ID = 1;
constexpr size_t CAN3_MOTOR_COUNT = 12;
constexpr uint32_t CAN3_COMMAND_TIMEOUT_MS = 50;

enum : uint8_t {
  CAN3_CONTROL_DISABLED = 0,
  CAN3_CONTROL_MIT = 1,
};

enum : uint16_t {
  CAN3_COMMAND_ENABLE = 1U << 0,
  CAN3_COMMAND_CLEAR_FAULT = 1U << 1,
};

enum : uint8_t {
  CAN3_STATE_INITIALIZED = 1U << 0,
  CAN3_STATE_CONNECTED = 1U << 1,
  CAN3_STATE_FEEDBACK_VALID = 1U << 2,
  CAN3_STATE_ENABLE_REQUESTED = 1U << 3,
  CAN3_STATE_DIAGNOSTICS_VALID = 1U << 4,
  CAN3_STATE_COMMAND_ALIVE = 1U << 5,
  CAN3_STATE_FAULT = 1U << 6,
};

struct Can3MitCommand {
  uint8_t motor_id = 0;
  uint8_t control_mode = CAN3_CONTROL_DISABLED;
  uint16_t flags = 0;
  float position_rad = 0.0f;
  float velocity_rad_s = 0.0f;
  float kp = 0.0f;
  float kd = 0.0f;
  float torque_nm = 0.0f;
};

struct Can3MotorTelemetry {
  uint8_t motor_id = 0;
  uint8_t status = 0;
  uint8_t fault_code = 0;
  uint8_t flags = 0;
  uint32_t last_rx_age_us = UINT32_MAX;
  float position_rad = NAN;
  float velocity_rad_s = NAN;
  float torque_nm = NAN;
  float temp_mos_c = NAN;
  float temp_rotor_c = NAN;
  float bus_voltage_v = NAN;
  float iq_current_a = NAN;
  int32_t rotation_count = INT32_MIN;
  float command_position_rad = 0.0f;
  float command_velocity_rad_s = 0.0f;
  float command_kp = 0.0f;
  float command_kd = 0.0f;
  float command_torque_nm = 0.0f;
  uint32_t command_sequence = 0;
};

namespace can3_detail {

constexpr uint32_t kCanBaudRate = 1000000;
constexpr uint8_t kMasterId = 0xFD;
constexpr uint32_t kMotorSlotIntervalUs = 400;
constexpr uint32_t kConnectionTimeoutUs = 250000;
constexpr uint32_t kDisabledRefreshUs = 100000;
constexpr float kPositionLimitRad = 4.0f * static_cast<float>(M_PI);
constexpr float kVelocityLimitRadS = 50.0f;
constexpr float kTorqueLimitNm = 60.0f;
constexpr float kKpLimit = 5000.0f;
constexpr float kKdLimit = 100.0f;

using CanBus = FlexCAN_T4<CAN3, RX_SIZE_256, TX_SIZE_16>;
using Motor = RoboStride<CAN3, RX_SIZE_256, TX_SIZE_16>;

inline CanBus bus;
inline Motor motors[CAN3_MOTOR_COUNT] = {
    Motor(&bus, kMasterId, 0x01, static_cast<int>(ActuatorType::ROBSTRIDE_03), 0.0f),
    Motor(&bus, kMasterId, 0x02, static_cast<int>(ActuatorType::ROBSTRIDE_03), 0.0f),
    Motor(&bus, kMasterId, 0x03, static_cast<int>(ActuatorType::ROBSTRIDE_03), 0.0f),
    Motor(&bus, kMasterId, 0x04, static_cast<int>(ActuatorType::ROBSTRIDE_03), 0.0f),
    Motor(&bus, kMasterId, 0x05, static_cast<int>(ActuatorType::ROBSTRIDE_03), 0.0f),
    Motor(&bus, kMasterId, 0x06, static_cast<int>(ActuatorType::ROBSTRIDE_03), 0.0f),
    Motor(&bus, kMasterId, 0x07, static_cast<int>(ActuatorType::ROBSTRIDE_03), 0.0f),
    Motor(&bus, kMasterId, 0x08, static_cast<int>(ActuatorType::ROBSTRIDE_03), 0.0f),
    Motor(&bus, kMasterId, 0x09, static_cast<int>(ActuatorType::ROBSTRIDE_03), 0.0f),
    Motor(&bus, kMasterId, 0x0A, static_cast<int>(ActuatorType::ROBSTRIDE_03), 0.0f),
    Motor(&bus, kMasterId, 0x0B, static_cast<int>(ActuatorType::ROBSTRIDE_03), 0.0f),
    Motor(&bus, kMasterId, 0x0C, static_cast<int>(ActuatorType::ROBSTRIDE_03), 0.0f),
};

inline Can3MitCommand commands[CAN3_MOTOR_COUNT] = {};
inline bool initialized[CAN3_MOTOR_COUNT] = {};
inline bool feedback_valid[CAN3_MOTOR_COUNT] = {};
inline bool effective_enabled[CAN3_MOTOR_COUNT] = {};
inline uint32_t last_rx_us[CAN3_MOTOR_COUNT] = {};
inline uint32_t last_stop_us[CAN3_MOTOR_COUNT] = {};
inline uint32_t last_slot_us = 0;
inline uint32_t last_command_ms = 0;
inline uint32_t accepted_command_sequence = 0;
inline size_t next_motor_index = 0;
inline bool has_command = false;
inline bool emergency_stop = false;

inline bool command_alive() {
  return has_command && !emergency_stop &&
         static_cast<uint32_t>(millis() - last_command_ms) <= CAN3_COMMAND_TIMEOUT_MS;
}

inline void receive_callback(const CAN_message_t &message) {
  for (size_t i = 0; i < CAN3_MOTOR_COUNT; ++i) {
    if (motors[i].setCanFrame(message)) {
      feedback_valid[i] = true;
      last_rx_us[i] = micros();
      return;
    }
  }
}

inline bool validate_commands(const Can3MitCommand *new_commands, size_t count) {
  if (new_commands == nullptr || count != CAN3_MOTOR_COUNT) {
    return false;
  }
  bool seen[CAN3_MOTOR_COUNT] = {};
  for (size_t i = 0; i < count; ++i) {
    const auto &command = new_commands[i];
    if (command.motor_id < CAN3_FIRST_MOTOR_ID ||
        command.motor_id >= CAN3_FIRST_MOTOR_ID + CAN3_MOTOR_COUNT ||
        command.control_mode > CAN3_CONTROL_MIT ||
        !std::isfinite(command.position_rad) ||
        !std::isfinite(command.velocity_rad_s) ||
        !std::isfinite(command.kp) ||
        !std::isfinite(command.kd) ||
        !std::isfinite(command.torque_nm)) {
      return false;
    }
    const size_t index = command.motor_id - CAN3_FIRST_MOTOR_ID;
    if (seen[index]) {
      return false;
    }
    seen[index] = true;
  }
  return true;
}

}  // namespace can3_detail

inline void can3_init() {
  using namespace can3_detail;
  bus.begin();
  bus.setBaudRate(kCanBaudRate);
  bus.setMaxMB(16);
  bus.enableFIFO();

  for (size_t i = 0; i < CAN3_MOTOR_COUNT; ++i) {
    commands[i].motor_id = static_cast<uint8_t>(CAN3_FIRST_MOTOR_ID + i);
    delay(50);
    initialized[i] = motors[i].init(
        kVelocityLimitRadS, 100.0f, move_control_mode, false, false);
    motors[i].stopMotorNonBlocking(false);
  }

  bus.onReceive(receive_callback);
  bus.enableFIFOInterrupt();
}

inline bool can3_apply_commands(const Can3MitCommand *new_commands,
                                size_t count,
                                uint32_t sequence,
                                bool request_emergency_stop) {
  using namespace can3_detail;
  if (!validate_commands(new_commands, count)) {
    return false;
  }

  for (size_t i = 0; i < count; ++i) {
    const auto &source = new_commands[i];
    const size_t index = source.motor_id - CAN3_FIRST_MOTOR_ID;
    commands[index] = source;
    commands[index].position_rad = constrain(
        source.position_rad, -kPositionLimitRad, kPositionLimitRad);
    commands[index].velocity_rad_s = constrain(
        source.velocity_rad_s, -kVelocityLimitRadS, kVelocityLimitRadS);
    commands[index].kp = constrain(source.kp, 0.0f, kKpLimit);
    commands[index].kd = constrain(source.kd, 0.0f, kKdLimit);
    commands[index].torque_nm = constrain(
        source.torque_nm, -kTorqueLimitNm, kTorqueLimitNm);
  }

  accepted_command_sequence = sequence;
  emergency_stop = request_emergency_stop;
  last_command_ms = millis();
  has_command = true;
  return true;
}

inline void can3_stop_all() {
  using namespace can3_detail;
  emergency_stop = true;
  for (size_t i = 0; i < CAN3_MOTOR_COUNT; ++i) {
    commands[i].control_mode = CAN3_CONTROL_DISABLED;
    commands[i].flags = 0;
  }
}

inline void can3_loop() {
  using namespace can3_detail;
  const uint32_t now_us = micros();
  if (static_cast<uint32_t>(now_us - last_slot_us) < kMotorSlotIntervalUs) {
    return;
  }
  last_slot_us = now_us;

  const size_t index = next_motor_index;
  next_motor_index = (next_motor_index + 1U) % CAN3_MOTOR_COUNT;
  const auto &command = commands[index];
  const bool connected = feedback_valid[index] &&
                         static_cast<uint32_t>(now_us - last_rx_us[index]) <=
                             kConnectionTimeoutUs;
  const bool enable = command_alive() && initialized[index] && connected &&
                      command.control_mode == CAN3_CONTROL_MIT &&
                      (command.flags & CAN3_COMMAND_ENABLE) != 0U;

  if (enable != effective_enabled[index]) {
    effective_enabled[index] = enable;
    if (enable) {
      motors[index].enableMotorNonBlocking();
    } else {
      motors[index].stopMotorNonBlocking(
          (command.flags & CAN3_COMMAND_CLEAR_FAULT) != 0U);
      last_stop_us[index] = now_us;
    }
    return;
  }

  if (enable) {
    motors[index].sendMitCommand(
        command.position_rad,
        command.velocity_rad_s,
        command.kp,
        command.kd,
        command.torque_nm);
    return;
  }

  if (static_cast<uint32_t>(now_us - last_stop_us[index]) >= kDisabledRefreshUs) {
    motors[index].stopMotorNonBlocking(
        (command.flags & CAN3_COMMAND_CLEAR_FAULT) != 0U);
    last_stop_us[index] = now_us;
  }
}

inline bool can3_command_alive() {
  return can3_detail::command_alive();
}

inline size_t can3_connected_count() {
  size_t count = 0;
  const uint32_t now_us = micros();
  for (size_t i = 0; i < CAN3_MOTOR_COUNT; ++i) {
    count += can3_detail::feedback_valid[i] &&
                     static_cast<uint32_t>(now_us - can3_detail::last_rx_us[i]) <=
                         can3_detail::kConnectionTimeoutUs
                 ? 1U
                 : 0U;
  }
  return count;
}

inline bool can3_get_telemetry(size_t index, Can3MotorTelemetry &telemetry) {
  using namespace can3_detail;
  if (index >= CAN3_MOTOR_COUNT) {
    return false;
  }

  const uint32_t now_us = micros();
  const bool connected = feedback_valid[index] &&
                         static_cast<uint32_t>(now_us - last_rx_us[index]) <=
                             kConnectionTimeoutUs;
  const auto &motor = motors[index];
  const auto &command = commands[index];
  telemetry = {};
  telemetry.motor_id = static_cast<uint8_t>(CAN3_FIRST_MOTOR_ID + index);
  telemetry.status = motor.statusPattern();
  telemetry.fault_code = motor.faultCode();
  telemetry.flags =
      (initialized[index] ? CAN3_STATE_INITIALIZED : 0U) |
      (connected ? CAN3_STATE_CONNECTED : 0U) |
      (feedback_valid[index] ? CAN3_STATE_FEEDBACK_VALID : 0U) |
      (effective_enabled[index] ? CAN3_STATE_ENABLE_REQUESTED : 0U) |
      (command_alive() ? CAN3_STATE_COMMAND_ALIVE : 0U) |
      (motor.faultCode() != 0U ? CAN3_STATE_FAULT : 0U);
  telemetry.last_rx_age_us = feedback_valid[index]
                                 ? static_cast<uint32_t>(now_us - last_rx_us[index])
                                 : UINT32_MAX;
  if (feedback_valid[index]) {
    telemetry.position_rad = motor.feedback.position_rad;
    telemetry.velocity_rad_s = motor.feedback.velocity_rad_s;
    telemetry.torque_nm = motor.feedback.torque_nm;
    telemetry.temp_mos_c = motor.feedback.temp_mos;
  }
  telemetry.temp_rotor_c = NAN;
  telemetry.bus_voltage_v = NAN;
  telemetry.iq_current_a = NAN;
  telemetry.rotation_count = INT32_MIN;
  telemetry.command_position_rad = command.position_rad;
  telemetry.command_velocity_rad_s = command.velocity_rad_s;
  telemetry.command_kp = command.kp;
  telemetry.command_kd = command.kd;
  telemetry.command_torque_nm = command.torque_nm;
  telemetry.command_sequence = accepted_command_sequence;
  return true;
}
