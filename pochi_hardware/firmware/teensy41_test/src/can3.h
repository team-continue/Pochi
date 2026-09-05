#pragma once

#include <Arduino.h>
#include <FlexCAN_T4.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

#include "robostride.h"

constexpr uint8_t CAN3_FIRST_MOTOR_ID = 0;
constexpr size_t CAN3_MOTOR_COUNT = 12;
constexpr uint32_t CAN3_COMMAND_TIMEOUT_MS = 50;

// Load-side absolute encoder positions at the robot's reference pose, indexed
// directly by CAN ID 0..11. Edit this one array when recalibrating the pose.
// Joint order is foot-to-body for each leg:
// FL 0,1,2 / RL 3,4,5 / RR 6,7,8 / FR 9,10,11.
constexpr float CAN3_INITIAL_POSITION_RAD[CAN3_MOTOR_COUNT] = {
    2.9789050f, 5.4445298f, 1.0083539f,
    3.2805416f, 0.9617540f, 0.1628750f,
    4.7653043f, 0.0347014f, 6.1671015f,
    2.9602377f, 2.2337207f, 3.1243120f,
};

// Final MIT target filter in the Teensy's joint coordinate system, indexed by
// CAN ID. These limits are enforced even if a host sends a wider target.
constexpr float CAN3_PI = 3.14159265358979323846f;
constexpr float CAN3_MIN_POSITION_RAD[CAN3_MOTOR_COUNT] = {
    -CAN3_PI, -3.0f * CAN3_PI / 4.0f, -3.0f * CAN3_PI / 4.0f,
    -CAN3_PI, -3.0f * CAN3_PI / 4.0f, -3.0f * CAN3_PI / 4.0f,
    -CAN3_PI, -3.0f * CAN3_PI / 4.0f, -3.0f * CAN3_PI / 4.0f,
    -CAN3_PI, -3.0f * CAN3_PI / 4.0f, -3.0f * CAN3_PI / 4.0f,
};
constexpr float CAN3_MAX_POSITION_RAD[CAN3_MOTOR_COUNT] = {
    CAN3_PI, 3.0f * CAN3_PI / 4.0f, 3.0f * CAN3_PI / 4.0f,
    CAN3_PI, 3.0f * CAN3_PI / 4.0f, 3.0f * CAN3_PI / 4.0f,
    CAN3_PI, 3.0f * CAN3_PI / 4.0f, 3.0f * CAN3_PI / 4.0f,
    CAN3_PI, 3.0f * CAN3_PI / 4.0f, 3.0f * CAN3_PI / 4.0f,
};

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
  CAN3_STATE_ENABLE_PENDING = 1U << 7,
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
constexpr uint32_t kMotorSlotIntervalUs = 1000;
constexpr uint32_t kConnectionTimeoutUs = 250000;
constexpr uint32_t kReadyStableUs = 500000;
constexpr uint32_t kInitializationRetryUs = 1000000;
constexpr uint32_t kDisabledRefreshUs = 500000;
constexpr uint32_t kEnableMotorIntervalUs = 100000;
constexpr uint32_t kEnableRetryUs = 10000;
constexpr uint32_t kEnableGoalPreloadUs = 10000;
constexpr uint32_t kPostEnableHoldUs = 100000;
constexpr uint8_t kEnableMaxAttempts = 20;
constexpr uint32_t kEnableSequenceTimeoutUs = 25000000;
constexpr uint16_t kMechanicalPositionIndex = 0x7019;
constexpr uint8_t kMotorModeReset = 0;
constexpr uint8_t kMotorModeRun = 2;
constexpr uint8_t kMotorModeUnknown = 0xFF;
constexpr uint16_t kAllMotorMask =
    static_cast<uint16_t>((1U << CAN3_MOTOR_COUNT) - 1U);
constexpr float kVelocityLimitRadS = 50.0f;
constexpr float kTorqueLimitNm = 60.0f;
constexpr float kKpLimit = 5000.0f;
constexpr float kKdLimit = 100.0f;

using CanBus = FlexCAN_T4<CAN3, RX_SIZE_256, TX_SIZE_16>;
using Motor = RoboStride<CAN3, RX_SIZE_256, TX_SIZE_16>;

inline CanBus bus;
inline Motor motors[CAN3_MOTOR_COUNT] = {
    Motor(&bus, kMasterId, 0x00, static_cast<int>(ActuatorType::ROBSTRIDE_03), CAN3_INITIAL_POSITION_RAD[0]),
    Motor(&bus, kMasterId, 0x01, static_cast<int>(ActuatorType::ROBSTRIDE_03), CAN3_INITIAL_POSITION_RAD[1]),
    Motor(&bus, kMasterId, 0x02, static_cast<int>(ActuatorType::ROBSTRIDE_03), CAN3_INITIAL_POSITION_RAD[2]),
    Motor(&bus, kMasterId, 0x03, static_cast<int>(ActuatorType::ROBSTRIDE_03), CAN3_INITIAL_POSITION_RAD[3]),
    Motor(&bus, kMasterId, 0x04, static_cast<int>(ActuatorType::ROBSTRIDE_03), CAN3_INITIAL_POSITION_RAD[4]),
    Motor(&bus, kMasterId, 0x05, static_cast<int>(ActuatorType::ROBSTRIDE_03), CAN3_INITIAL_POSITION_RAD[5]),
    Motor(&bus, kMasterId, 0x06, static_cast<int>(ActuatorType::ROBSTRIDE_03), CAN3_INITIAL_POSITION_RAD[6]),
    Motor(&bus, kMasterId, 0x07, static_cast<int>(ActuatorType::ROBSTRIDE_03), CAN3_INITIAL_POSITION_RAD[7]),
    Motor(&bus, kMasterId, 0x08, static_cast<int>(ActuatorType::ROBSTRIDE_03), CAN3_INITIAL_POSITION_RAD[8]),
    Motor(&bus, kMasterId, 0x09, static_cast<int>(ActuatorType::ROBSTRIDE_03), CAN3_INITIAL_POSITION_RAD[9]),
    Motor(&bus, kMasterId, 0x0A, static_cast<int>(ActuatorType::ROBSTRIDE_03), CAN3_INITIAL_POSITION_RAD[10]),
    Motor(&bus, kMasterId, 0x0B, static_cast<int>(ActuatorType::ROBSTRIDE_03), CAN3_INITIAL_POSITION_RAD[11]),
};

inline Can3MitCommand commands[CAN3_MOTOR_COUNT] = {};
inline bool initialized[CAN3_MOTOR_COUNT] = {};
inline bool encoder_valid[CAN3_MOTOR_COUNT] = {};
inline bool motion_feedback_valid[CAN3_MOTOR_COUNT] = {};
inline bool effective_enabled[CAN3_MOTOR_COUNT] = {};
inline bool enable_confirmed[CAN3_MOTOR_COUNT] = {};
inline float encoder_position_rad[CAN3_MOTOR_COUNT] = {};
inline uint32_t last_rx_us[CAN3_MOTOR_COUNT] = {};
inline uint32_t last_stop_us[CAN3_MOTOR_COUNT] = {};
inline uint32_t enable_first_attempt_us[CAN3_MOTOR_COUNT] = {};
inline uint32_t last_enable_attempt_us[CAN3_MOTOR_COUNT] = {};
inline uint32_t enable_confirmed_us[CAN3_MOTOR_COUNT] = {};
inline uint8_t enable_attempt_count[CAN3_MOTOR_COUNT] = {};
inline bool enable_goal_preloaded[CAN3_MOTOR_COUNT] = {};
inline uint32_t enable_goal_preload_us[CAN3_MOTOR_COUNT] = {};
inline float enable_hold_position_rad[CAN3_MOTOR_COUNT] = {};
inline uint8_t reported_mode_status[CAN3_MOTOR_COUNT] = {};
inline uint32_t last_slot_us = 0;
inline uint32_t last_command_ms = 0;
inline uint32_t arm_granted_us = 0;
inline uint32_t last_enable_confirmed_us = 0;
inline uint32_t accepted_command_sequence = 0;
inline size_t next_motor_index = 0;
inline size_t enable_motor_index = CAN3_MOTOR_COUNT;
inline uint16_t armed_enable_mask = 0;
inline uint16_t previous_enable_mask = 0;
inline bool has_command = false;
inline bool emergency_stop = false;
inline bool rearm_required = true;
inline bool zero_mask_seen_since_latch = false;
inline bool arm_granted = false;
inline bool ready = false;
inline bool was_host_communication_alive = false;
inline uint32_t all_connected_since_us = 0;
inline uint32_t initialization_pass_finished_us = 0;
inline size_t initialization_motor_index = 0;
inline bool initialization_stop_step = true;
inline bool initialization_write_ok = false;
inline bool initialization_pass_complete = false;
inline bool run_mode_lost = false;

inline bool host_communication_alive() {
  return has_command &&
         static_cast<uint32_t>(millis() - last_command_ms) <= CAN3_COMMAND_TIMEOUT_MS;
}

inline bool command_alive() {
  return host_communication_alive() && !emergency_stop;
}

inline uint16_t requested_enable_mask(
    const Can3MitCommand *candidate_commands) {
  uint16_t mask = 0;
  for (size_t i = 0; i < CAN3_MOTOR_COUNT; ++i) {
    if (candidate_commands[i].control_mode == CAN3_CONTROL_MIT &&
        (candidate_commands[i].flags & CAN3_COMMAND_ENABLE) != 0U) {
      mask |= static_cast<uint16_t>(1U << i);
    }
  }
  return mask;
}

inline bool motors_connected(uint16_t mask, uint32_t now_us) {
  for (size_t i = 0; i < CAN3_MOTOR_COUNT; ++i) {
    if ((mask & static_cast<uint16_t>(1U << i)) != 0U &&
        (!initialized[i] || !encoder_valid[i] ||
        static_cast<uint32_t>(now_us - last_rx_us[i]) > kConnectionTimeoutUs ||
         motors[i].faultCode() != 0U)) {
      return false;
    }
  }
  return true;
}

inline bool all_motors_connected(uint32_t now_us) {
  return motors_connected(kAllMotorMask, now_us);
}

inline bool positions_safe(uint16_t mask) {
  for (size_t i = 0; i < CAN3_MOTOR_COUNT; ++i) {
    if ((mask & static_cast<uint16_t>(1U << i)) != 0U &&
        (!std::isfinite(encoder_position_rad[i]) ||
        encoder_position_rad[i] < CAN3_MIN_POSITION_RAD[i] ||
         encoder_position_rad[i] > CAN3_MAX_POSITION_RAD[i])) {
      return false;
    }
  }
  return true;
}

inline bool all_enables_confirmed(uint16_t mask) {
  if (mask == 0U) {
    return false;
  }
  for (size_t i = 0; i < CAN3_MOTOR_COUNT; ++i) {
    if ((mask & static_cast<uint16_t>(1U << i)) != 0U &&
        (!enable_confirmed[i] ||
         reported_mode_status[i] != kMotorModeRun)) {
      return false;
    }
  }
  return true;
}

inline size_t first_unconfirmed_motor(uint16_t mask) {
  for (size_t i = 0; i < CAN3_MOTOR_COUNT; ++i) {
    if ((mask & static_cast<uint16_t>(1U << i)) != 0U &&
        !enable_confirmed[i]) {
      return i;
    }
  }
  return CAN3_MOTOR_COUNT;
}

inline void clear_motor_enable_tracking(size_t index) {
  effective_enabled[index] = false;
  enable_confirmed[index] = false;
  enable_first_attempt_us[index] = 0;
  last_enable_attempt_us[index] = 0;
  enable_confirmed_us[index] = 0;
  enable_attempt_count[index] = 0;
  enable_goal_preloaded[index] = false;
  enable_goal_preload_us[index] = 0;
  enable_hold_position_rad[index] = 0.0f;
}

inline void clear_enable_tracking() {
  arm_granted_us = 0;
  last_enable_confirmed_us = 0;
  enable_motor_index = CAN3_MOTOR_COUNT;
  armed_enable_mask = 0;
  run_mode_lost = false;
  for (size_t i = 0; i < CAN3_MOTOR_COUNT; ++i) {
    clear_motor_enable_tracking(i);
  }
}

inline void begin_initialization(uint32_t now_us) {
  ready = false;
  arm_granted = false;
  rearm_required = true;
  zero_mask_seen_since_latch = false;
  all_connected_since_us = 0;
  initialization_motor_index = 0;
  initialization_stop_step = true;
  initialization_write_ok = false;
  initialization_pass_complete = false;
  initialization_pass_finished_us = now_us;
  clear_enable_tracking();

  // Stop every actuator as soon as a global communication fault is detected.
  // The run-mode setup itself is then paced by can3_loop so USB and IMU work
  // continue without a blocking initialization sequence.
  for (size_t i = 0; i < CAN3_MOTOR_COUNT; ++i) {
    initialized[i] = false;
    reported_mode_status[i] = kMotorModeUnknown;
    motors[i].stopMotorNonBlocking(true);
    last_stop_us[i] = now_us;
  }
}

inline bool run_initialization_step(uint32_t now_us) {
  if (initialization_pass_complete) {
    return false;
  }

  const size_t index = initialization_motor_index;
  if (initialization_stop_step) {
    // Clear latched actuator faults during the torque-off initialization pass.
    // Active faults remain reported and still prevent READY.
    initialization_write_ok = motors[index].stopMotorNonBlocking(true);
    last_stop_us[index] = now_us;
    initialization_stop_step = false;
    return true;
  }

  const bool mode_written =
      motors[index].setRunModeNonBlocking(move_control_mode);
  initialized[index] = initialization_write_ok && mode_written;
  initialization_stop_step = true;
  initialization_write_ok = false;
  ++initialization_motor_index;
  if (initialization_motor_index == CAN3_MOTOR_COUNT) {
    initialization_motor_index = 0;
    initialization_pass_complete = true;
    initialization_pass_finished_us = now_us;
  }
  return true;
}

inline void update_safety_state(uint32_t now_us) {
  const bool host_alive = host_communication_alive();
  if (was_host_communication_alive && !host_alive) {
    begin_initialization(now_us);
    previous_enable_mask = requested_enable_mask(commands);
  }
  was_host_communication_alive = host_alive;

  if (arm_granted &&
      (run_mode_lost ||
       (!all_enables_confirmed(armed_enable_mask) && arm_granted_us != 0U &&
        static_cast<uint32_t>(now_us - arm_granted_us) >=
            kEnableSequenceTimeoutUs))) {
    begin_initialization(now_us);
    previous_enable_mask = requested_enable_mask(commands);
    return;
  }

  if (arm_granted && enable_motor_index < CAN3_MOTOR_COUNT) {
    const size_t index = enable_motor_index;
    if (!enable_confirmed[index] &&
        enable_attempt_count[index] >= kEnableMaxAttempts &&
        static_cast<uint32_t>(now_us - last_enable_attempt_us[index]) >=
            kEnableRetryUs) {
      begin_initialization(now_us);
      previous_enable_mask = requested_enable_mask(commands);
      return;
    }
  }

  const bool all_connected = all_motors_connected(now_us);
  const uint16_t monitored_mask =
      armed_enable_mask == 0U ? kAllMotorMask : armed_enable_mask;
  if (ready && !motors_connected(monitored_mask, now_us)) {
    begin_initialization(now_us);
    previous_enable_mask = requested_enable_mask(commands);
    return;
  }

  if (!ready) {
    if (initialization_pass_complete && all_connected) {
      if (all_connected_since_us == 0U) {
        all_connected_since_us = now_us;
      } else if (static_cast<uint32_t>(now_us - all_connected_since_us) >=
                 kReadyStableUs) {
        ready = true;
      }
    } else {
      all_connected_since_us = 0;
    }

    if (!ready && initialization_pass_complete && !all_connected &&
        static_cast<uint32_t>(now_us - initialization_pass_finished_us) >=
            kInitializationRetryUs) {
      initialization_motor_index = 0;
      initialization_stop_step = true;
      initialization_write_ok = false;
      initialization_pass_complete = false;
      initialization_pass_finished_us = now_us;
      for (size_t i = 0; i < CAN3_MOTOR_COUNT; ++i) {
        initialized[i] = false;
      }
    }
  }
}

inline void receive_callback(const CAN_message_t &message) {
  if (!message.flags.extended || message.len != 8) {
    return;
  }

  const uint8_t communication_type =
      static_cast<uint8_t>((message.id >> 24) & 0x1F);
  const uint16_t parameter_index =
      static_cast<uint16_t>(message.buf[0]) |
      (static_cast<uint16_t>(message.buf[1]) << 8);

  for (size_t i = 0; i < CAN3_MOTOR_COUNT; ++i) {
    auto &motor = motors[i];
    if (!motor.setCanFrame(message)) {
      continue;
    }

    float position_rad = NAN;
    const uint32_t receive_us = micros();
    if (communication_type == Communication_Type_MotorRequest) {
      position_rad = motor.feedback.position_rad;
      motion_feedback_valid[i] = true;
      reported_mode_status[i] = motor.statusPattern();
      if (arm_granted &&
          (armed_enable_mask & static_cast<uint16_t>(1U << i)) != 0U &&
          enable_first_attempt_us[i] != 0U) {
        if (reported_mode_status[i] == kMotorModeRun) {
          if (!enable_confirmed[i]) {
            enable_confirmed_us[i] = receive_us;
            last_enable_confirmed_us = receive_us;
          }
          enable_confirmed[i] = true;
          effective_enabled[i] = true;
        } else if (enable_confirmed[i]) {
          enable_confirmed[i] = false;
          effective_enabled[i] = false;
          run_mode_lost = true;
        }
      }
    } else if (communication_type == Communication_Type_GetSingleParameter &&
               parameter_index == kMechanicalPositionIndex) {
      position_rad = motor.rawToJointPosition(motor.drw.mechPos.data);
    }

    if (std::isfinite(position_rad)) {
      encoder_position_rad[i] = position_rad;
      encoder_valid[i] = true;
      last_rx_us[i] = receive_us;
    }
    return;
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
  bus.onReceive(receive_callback);
  bus.enableFIFOInterrupt();
  delay(100);

  // Boot is always torque-off. Initialization is completed non-blockingly in
  // can3_loop, and a fresh OFF -> ON transition is required afterward.
  for (size_t i = 0; i < CAN3_MOTOR_COUNT; ++i) {
    commands[i].motor_id = static_cast<uint8_t>(CAN3_FIRST_MOTOR_ID + i);
    last_stop_us[i] = micros();
  }
  begin_initialization(micros());
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
        source.position_rad,
        CAN3_MIN_POSITION_RAD[index],
        CAN3_MAX_POSITION_RAD[index]);
    commands[index].velocity_rad_s = constrain(
        source.velocity_rad_s, -kVelocityLimitRadS, kVelocityLimitRadS);
    commands[index].kp = constrain(source.kp, 0.0f, kKpLimit);
    commands[index].kd = constrain(source.kd, 0.0f, kKdLimit);
    commands[index].torque_nm = constrain(
        source.torque_nm, -kTorqueLimitNm, kTorqueLimitNm);
  }

  accepted_command_sequence = sequence;
  const bool emergency_stop_started = request_emergency_stop && !emergency_stop;
  emergency_stop = request_emergency_stop;
  last_command_ms = millis();
  has_command = true;

  const uint32_t now_us = micros();
  const uint16_t enable_mask = requested_enable_mask(commands);
  if (emergency_stop_started) {
    begin_initialization(now_us);
  }
  if (request_emergency_stop) {
    previous_enable_mask = enable_mask;
    return true;
  }

  if (enable_mask == 0U) {
    for (size_t i = 0; i < CAN3_MOTOR_COUNT; ++i) {
      if ((armed_enable_mask & static_cast<uint16_t>(1U << i)) != 0U) {
        motors[i].stopMotorNonBlocking(
            (commands[i].flags & CAN3_COMMAND_CLEAR_FAULT) != 0U);
        last_stop_us[i] = now_us;
        reported_mode_status[i] = kMotorModeReset;
      }
      clear_motor_enable_tracking(i);
    }
    armed_enable_mask = 0;
    arm_granted = false;
    rearm_required = true;
    zero_mask_seen_since_latch = true;
    arm_granted_us = 0;
    last_enable_confirmed_us = 0;
    enable_motor_index = CAN3_MOTOR_COUNT;
    run_mode_lost = false;
  } else if (rearm_required) {
    if (previous_enable_mask == 0U && zero_mask_seen_since_latch && ready &&
        motors_connected(enable_mask, now_us) && positions_safe(enable_mask)) {
      clear_enable_tracking();
      armed_enable_mask = enable_mask;
      arm_granted = true;
      rearm_required = false;
      zero_mask_seen_since_latch = false;
      arm_granted_us = now_us;
      enable_motor_index = first_unconfirmed_motor(armed_enable_mask);
    }
  } else {
    const uint16_t removed_mask =
        static_cast<uint16_t>(armed_enable_mask & ~enable_mask);
    for (size_t i = 0; i < CAN3_MOTOR_COUNT; ++i) {
      if ((removed_mask & static_cast<uint16_t>(1U << i)) == 0U) {
        continue;
      }
      motors[i].stopMotorNonBlocking(
          (commands[i].flags & CAN3_COMMAND_CLEAR_FAULT) != 0U);
      last_stop_us[i] = now_us;
      reported_mode_status[i] = kMotorModeReset;
      clear_motor_enable_tracking(i);
    }
    armed_enable_mask =
        static_cast<uint16_t>(armed_enable_mask & enable_mask);

    const uint16_t added_mask =
        static_cast<uint16_t>(enable_mask & ~armed_enable_mask);
    if (added_mask != 0U) {
      if (!motors_connected(added_mask, now_us) ||
          !positions_safe(added_mask)) {
        begin_initialization(now_us);
        previous_enable_mask = enable_mask;
        return true;
      }
      for (size_t i = 0; i < CAN3_MOTOR_COUNT; ++i) {
        if ((added_mask & static_cast<uint16_t>(1U << i)) != 0U) {
          clear_motor_enable_tracking(i);
        }
      }
      armed_enable_mask =
          static_cast<uint16_t>(armed_enable_mask | added_mask);
      arm_granted_us = now_us;
      run_mode_lost = false;
    }

    arm_granted = armed_enable_mask != 0U;
    if (!arm_granted) {
      rearm_required = true;
      zero_mask_seen_since_latch = true;
      arm_granted_us = 0;
      last_enable_confirmed_us = 0;
    }
    if (enable_motor_index >= CAN3_MOTOR_COUNT ||
        (armed_enable_mask &
         static_cast<uint16_t>(1U << enable_motor_index)) == 0U ||
        enable_confirmed[enable_motor_index]) {
      enable_motor_index = first_unconfirmed_motor(armed_enable_mask);
    }
  }
  previous_enable_mask = enable_mask;
  return true;
}

inline void can3_stop_all() {
  using namespace can3_detail;
  emergency_stop = true;
  for (size_t i = 0; i < CAN3_MOTOR_COUNT; ++i) {
    commands[i].control_mode = CAN3_CONTROL_DISABLED;
    commands[i].flags = 0;
  }
  previous_enable_mask = 0;
  begin_initialization(micros());
  zero_mask_seen_since_latch = true;
}

inline void can3_loop() {
  using namespace can3_detail;
  const uint32_t now_us = micros();
  if (static_cast<uint32_t>(now_us - last_slot_us) < kMotorSlotIntervalUs) {
    return;
  }
  last_slot_us = now_us;

  update_safety_state(now_us);
  if (run_initialization_step(now_us)) {
    return;
  }

  if (arm_granted &&
      (enable_motor_index >= CAN3_MOTOR_COUNT ||
       enable_confirmed[enable_motor_index] ||
       (armed_enable_mask &
        static_cast<uint16_t>(1U << enable_motor_index)) == 0U)) {
    enable_motor_index = first_unconfirmed_motor(armed_enable_mask);
  }

  const size_t index = next_motor_index;
  next_motor_index = (next_motor_index + 1U) % CAN3_MOTOR_COUNT;
  const auto &command = commands[index];
  const bool connected = encoder_valid[index] &&
                         static_cast<uint32_t>(now_us - last_rx_us[index]) <=
                             kConnectionTimeoutUs;
  const bool position_in_range =
      encoder_position_rad[index] >= CAN3_MIN_POSITION_RAD[index] &&
      encoder_position_rad[index] <= CAN3_MAX_POSITION_RAD[index];
  const bool selected =
      (armed_enable_mask & static_cast<uint16_t>(1U << index)) != 0U;
  const bool spacing_elapsed =
      last_enable_confirmed_us == 0U ||
      static_cast<uint32_t>(now_us - last_enable_confirmed_us) >=
          kEnableMotorIntervalUs;
  const bool scheduled =
      enable_confirmed[index] ||
      (index == enable_motor_index && spacing_elapsed);
  const bool enable = arm_granted && ready && command_alive() && selected &&
                      scheduled && initialized[index] && connected &&
                      position_in_range &&
                      command.control_mode == CAN3_CONTROL_MIT &&
                      (command.flags & CAN3_COMMAND_ENABLE) != 0U;

  if (!enable) {
    if (effective_enabled[index] ||
        (!selected && enable_first_attempt_us[index] != 0U)) {
      motors[index].stopMotorNonBlocking(
          (command.flags & CAN3_COMMAND_CLEAR_FAULT) != 0U);
      last_stop_us[index] = now_us;
      clear_motor_enable_tracking(index);
      reported_mode_status[index] = kMotorModeReset;
      return;
    }
    if (!selected &&
        static_cast<uint32_t>(now_us - last_stop_us[index]) >=
            kDisabledRefreshUs) {
      motors[index].stopMotorNonBlocking(
          (command.flags & CAN3_COMMAND_CLEAR_FAULT) != 0U);
      last_stop_us[index] = now_us;
      return;
    }

    // Mechanical position stays observable while torque is off or while a
    // selected motor waits for the preceding 100 ms enable interval.
    motors[index].requestParameterNonBlocking(kMechanicalPositionIndex);
    return;
  }

  if (!enable_confirmed[index]) {
    if (!enable_goal_preloaded[index]) {
      enable_hold_position_rad[index] = encoder_position_rad[index];
      if (motors[index].sendMitCommand(
              enable_hold_position_rad[index],
              0.0f,
              command.kp,
              command.kd,
              0.0f)) {
        enable_goal_preloaded[index] = true;
        enable_goal_preload_us[index] = now_us;
      }
      return;
    }

    if (static_cast<uint32_t>(now_us - enable_goal_preload_us[index]) <
        kEnableGoalPreloadUs) {
      return;
    }

    const bool retry_due =
        enable_first_attempt_us[index] == 0U ||
        static_cast<uint32_t>(now_us - last_enable_attempt_us[index]) >=
            kEnableRetryUs;
    if (retry_due && enable_attempt_count[index] < kEnableMaxAttempts &&
        motors[index].enableMotorNonBlocking()) {
      if (enable_first_attempt_us[index] == 0U) {
        enable_first_attempt_us[index] = now_us;
      }
      last_enable_attempt_us[index] = now_us;
      ++enable_attempt_count[index];
      // Queue the same live-pose goal immediately after Enable so the motor
      // never spends a control cycle on a target retained from an earlier run.
      motors[index].sendMitCommand(
          enable_hold_position_rad[index],
          0.0f,
          command.kp,
          command.kd,
          0.0f);
      return;
    }

    motors[index].requestParameterNonBlocking(kMechanicalPositionIndex);
    return;
  }

  const bool keep_startup_pose =
      static_cast<uint32_t>(now_us - enable_confirmed_us[index]) <
      kPostEnableHoldUs;
  motors[index].sendMitCommand(
      keep_startup_pose ? enable_hold_position_rad[index]
                        : command.position_rad,
      command.velocity_rad_s,
      command.kp,
      command.kd,
      command.torque_nm);
}

inline bool can3_command_alive() {
  return can3_detail::command_alive();
}

inline bool can3_host_communication_alive() {
  return can3_detail::host_communication_alive();
}

inline bool can3_initializing() {
  return !can3_detail::ready;
}

inline bool can3_ready() {
  return can3_detail::ready;
}

inline bool can3_rearm_required() {
  return can3_detail::rearm_required;
}

inline bool can3_torque_active() {
  using namespace can3_detail;
  if (!arm_granted || !ready) {
    return false;
  }
  for (size_t i = 0; i < CAN3_MOTOR_COUNT; ++i) {
    if ((armed_enable_mask & static_cast<uint16_t>(1U << i)) != 0U &&
        effective_enabled[i] && enable_confirmed[i] &&
        reported_mode_status[i] == kMotorModeRun) {
      return true;
    }
  }
  return false;
}

inline size_t can3_connected_count() {
  size_t count = 0;
  const uint32_t now_us = micros();
  for (size_t i = 0; i < CAN3_MOTOR_COUNT; ++i) {
    count += can3_detail::encoder_valid[i] &&
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
  const bool connected = encoder_valid[index] &&
                         static_cast<uint32_t>(now_us - last_rx_us[index]) <=
                             kConnectionTimeoutUs;
  const auto &motor = motors[index];
  const auto &command = commands[index];
  telemetry = {};
  telemetry.motor_id = static_cast<uint8_t>(CAN3_FIRST_MOTOR_ID + index);
  telemetry.status = reported_mode_status[index];
  telemetry.fault_code = motor.faultCode();
  telemetry.flags =
      (initialized[index] ? CAN3_STATE_INITIALIZED : 0U) |
      (connected ? CAN3_STATE_CONNECTED : 0U) |
      (encoder_valid[index] ? CAN3_STATE_FEEDBACK_VALID : 0U) |
      (effective_enabled[index] ? CAN3_STATE_ENABLE_REQUESTED : 0U) |
      (command_alive() ? CAN3_STATE_COMMAND_ALIVE : 0U) |
      (motor.faultCode() != 0U ? CAN3_STATE_FAULT : 0U) |
      (arm_granted &&
               (armed_enable_mask & static_cast<uint16_t>(1U << index)) != 0U &&
               !enable_confirmed[index]
           ? CAN3_STATE_ENABLE_PENDING
           : 0U);
  telemetry.last_rx_age_us = encoder_valid[index]
                                 ? static_cast<uint32_t>(now_us - last_rx_us[index])
                                 : UINT32_MAX;
  if (encoder_valid[index]) {
    telemetry.position_rad = encoder_position_rad[index];
  }
  if (effective_enabled[index] && motion_feedback_valid[index]) {
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
