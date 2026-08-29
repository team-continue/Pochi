#pragma once

#include <Arduino.h>

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "can3.h"
#include "imu.h"

constexpr uint8_t USB_PROTOCOL_VERSION = 1;
constexpr uint8_t USB_MESSAGE_COMMAND = 1;
constexpr uint8_t USB_MESSAGE_STATE = 2;
constexpr uint16_t USB_FLAG_EMERGENCY_STOP = 1U << 0;
constexpr uint16_t USB_STATE_COMMAND_ALIVE = 1U << 0;
constexpr uint16_t USB_STATE_ANY_FAULT = 1U << 1;
constexpr uint16_t USB_STATE_ALL_INITIALIZED = 1U << 2;

constexpr size_t USB_HEADER_BYTES = 24;
constexpr size_t USB_COMMAND_RECORD_BYTES = 24;
constexpr size_t USB_MOTOR_STATE_BYTES = 64;
constexpr size_t USB_IMU_STATE_BYTES = 64;
constexpr size_t USB_CRC_BYTES = 4;
constexpr size_t USB_COMMAND_PAYLOAD_BYTES =
    USB_COMMAND_RECORD_BYTES * CAN3_MOTOR_COUNT;
constexpr size_t USB_STATE_PAYLOAD_BYTES =
    USB_MOTOR_STATE_BYTES * CAN3_MOTOR_COUNT + USB_IMU_STATE_BYTES;
constexpr size_t USB_COMMAND_PACKET_BYTES =
    USB_HEADER_BYTES + USB_COMMAND_PAYLOAD_BYTES + USB_CRC_BYTES;
constexpr size_t USB_STATE_PACKET_BYTES =
    USB_HEADER_BYTES + USB_STATE_PAYLOAD_BYTES + USB_CRC_BYTES;
constexpr size_t USB_MAX_DECODED_BYTES = USB_STATE_PACKET_BYTES;
constexpr size_t USB_MAX_ENCODED_BYTES =
    USB_MAX_DECODED_BYTES + USB_MAX_DECODED_BYTES / 254U + 2U;
constexpr uint32_t USB_STATE_INTERVAL_US = 5000;

static_assert(USB_COMMAND_PACKET_BYTES == 316, "Unexpected command packet size");
static_assert(USB_STATE_PACKET_BYTES == 860, "Unexpected state packet size");

namespace usb_detail {

inline uint8_t receive_encoded[USB_MAX_ENCODED_BYTES] = {};
inline uint8_t receive_decoded[USB_COMMAND_PACKET_BYTES] = {};
inline uint8_t transmit_decoded[USB_STATE_PACKET_BYTES] = {};
inline uint8_t transmit_encoded[USB_MAX_ENCODED_BYTES] = {};
inline size_t receive_length = 0;
inline bool receive_overflow = false;
inline bool have_command_sequence = false;
inline uint32_t last_command_sequence = 0;
inline uint32_t state_sequence = 0;
inline uint32_t last_state_us = 0;
inline uint32_t last_micros32 = 0;
inline uint64_t micros_high = 0;

inline void write_u16(uint8_t *buffer, size_t offset, uint16_t value) {
  buffer[offset] = static_cast<uint8_t>(value);
  buffer[offset + 1] = static_cast<uint8_t>(value >> 8);
}

inline void write_u32(uint8_t *buffer, size_t offset, uint32_t value) {
  for (uint8_t i = 0; i < 4; ++i) {
    buffer[offset + i] = static_cast<uint8_t>(value >> (i * 8U));
  }
}

inline void write_u64(uint8_t *buffer, size_t offset, uint64_t value) {
  for (uint8_t i = 0; i < 8; ++i) {
    buffer[offset + i] = static_cast<uint8_t>(value >> (i * 8U));
  }
}

inline void write_float(uint8_t *buffer, size_t offset, float value) {
  uint32_t bits = 0;
  static_assert(sizeof(bits) == sizeof(value), "float32 required");
  std::memcpy(&bits, &value, sizeof(bits));
  write_u32(buffer, offset, bits);
}

inline uint16_t read_u16(const uint8_t *buffer, size_t offset) {
  return static_cast<uint16_t>(buffer[offset]) |
         static_cast<uint16_t>(buffer[offset + 1]) << 8;
}

inline uint32_t read_u32(const uint8_t *buffer, size_t offset) {
  uint32_t value = 0;
  for (uint8_t i = 0; i < 4; ++i) {
    value |= static_cast<uint32_t>(buffer[offset + i]) << (i * 8U);
  }
  return value;
}

inline float read_float(const uint8_t *buffer, size_t offset) {
  const uint32_t bits = read_u32(buffer, offset);
  float value = 0.0f;
  std::memcpy(&value, &bits, sizeof(value));
  return value;
}

inline uint32_t crc32(const uint8_t *data, size_t length) {
  uint32_t crc = 0xFFFFFFFFU;
  for (size_t i = 0; i < length; ++i) {
    crc ^= data[i];
    for (uint8_t bit = 0; bit < 8; ++bit) {
      crc = (crc >> 1U) ^ (0xEDB88320U & (0U - (crc & 1U)));
    }
  }
  return ~crc;
}

inline size_t cobs_encode(const uint8_t *input,
                          size_t length,
                          uint8_t *output,
                          size_t capacity) {
  if (input == nullptr || output == nullptr || capacity == 0U) {
    return 0;
  }
  size_t read_index = 0;
  size_t write_index = 1;
  size_t code_index = 0;
  uint8_t code = 1;
  while (read_index < length) {
    if (input[read_index] == 0U) {
      if (code_index >= capacity) {
        return 0;
      }
      output[code_index] = code;
      code = 1;
      code_index = write_index++;
    } else {
      if (write_index >= capacity) {
        return 0;
      }
      output[write_index++] = input[read_index];
      if (++code == 0xFFU) {
        if (code_index >= capacity) {
          return 0;
        }
        output[code_index] = code;
        code = 1;
        code_index = write_index++;
      }
    }
    ++read_index;
  }
  if (code_index >= capacity) {
    return 0;
  }
  output[code_index] = code;
  return write_index;
}

inline size_t cobs_decode(const uint8_t *input,
                          size_t length,
                          uint8_t *output,
                          size_t capacity) {
  size_t read_index = 0;
  size_t write_index = 0;
  while (read_index < length) {
    const uint8_t code = input[read_index++];
    if (code == 0U || read_index + code - 1U > length) {
      return 0;
    }
    for (uint8_t i = 1; i < code; ++i) {
      if (write_index >= capacity) {
        return 0;
      }
      output[write_index++] = input[read_index++];
    }
    if (code != 0xFFU && read_index < length) {
      if (write_index >= capacity) {
        return 0;
      }
      output[write_index++] = 0U;
    }
  }
  return write_index;
}

inline uint64_t micros64() {
  const uint32_t now = micros();
  if (now < last_micros32) {
    micros_high += 1ULL << 32U;
  }
  last_micros32 = now;
  return micros_high | now;
}

inline void write_header(uint8_t type,
                         uint16_t flags,
                         uint32_t sequence,
                         uint64_t timestamp_us,
                         uint16_t payload_bytes) {
  transmit_decoded[0] = 'P';
  transmit_decoded[1] = 'C';
  transmit_decoded[2] = 'H';
  transmit_decoded[3] = 'I';
  transmit_decoded[4] = USB_PROTOCOL_VERSION;
  transmit_decoded[5] = type;
  write_u16(transmit_decoded, 6, flags);
  write_u32(transmit_decoded, 8, sequence);
  write_u64(transmit_decoded, 12, timestamp_us);
  write_u16(transmit_decoded, 20, payload_bytes);
  transmit_decoded[22] = static_cast<uint8_t>(CAN3_MOTOR_COUNT);
  transmit_decoded[23] = 0;
}

inline bool sequence_is_newer(uint32_t sequence) {
  return !have_command_sequence || !can3_command_alive() ||
         static_cast<int32_t>(sequence - last_command_sequence) > 0;
}

inline bool parse_command(const uint8_t *packet, size_t length) {
  if (length != USB_COMMAND_PACKET_BYTES ||
      packet[0] != 'P' || packet[1] != 'C' || packet[2] != 'H' || packet[3] != 'I' ||
      packet[4] != USB_PROTOCOL_VERSION || packet[5] != USB_MESSAGE_COMMAND ||
      read_u16(packet, 20) != USB_COMMAND_PAYLOAD_BYTES ||
      packet[22] != CAN3_MOTOR_COUNT) {
    return false;
  }
  const uint32_t expected_crc = read_u32(packet, length - USB_CRC_BYTES);
  if (crc32(packet, length - USB_CRC_BYTES) != expected_crc) {
    return false;
  }
  const uint32_t sequence = read_u32(packet, 8);
  if (!sequence_is_newer(sequence)) {
    return false;
  }

  Can3MitCommand commands[CAN3_MOTOR_COUNT] = {};
  size_t offset = USB_HEADER_BYTES;
  for (size_t i = 0; i < CAN3_MOTOR_COUNT; ++i) {
    commands[i].motor_id = packet[offset];
    commands[i].control_mode = packet[offset + 1];
    commands[i].flags = read_u16(packet, offset + 2);
    commands[i].position_rad = read_float(packet, offset + 4);
    commands[i].velocity_rad_s = read_float(packet, offset + 8);
    commands[i].kp = read_float(packet, offset + 12);
    commands[i].kd = read_float(packet, offset + 16);
    commands[i].torque_nm = read_float(packet, offset + 20);
    offset += USB_COMMAND_RECORD_BYTES;
  }

  const bool emergency_stop = (read_u16(packet, 6) & USB_FLAG_EMERGENCY_STOP) != 0U;
  if (!can3_apply_commands(
          commands, CAN3_MOTOR_COUNT, sequence, emergency_stop)) {
    return false;
  }
  last_command_sequence = sequence;
  have_command_sequence = true;
  return true;
}

inline void write_motor_state(size_t &offset, const Can3MotorTelemetry &state) {
  transmit_decoded[offset] = state.motor_id;
  transmit_decoded[offset + 1] = state.status;
  transmit_decoded[offset + 2] = state.fault_code;
  transmit_decoded[offset + 3] = state.flags;
  write_u32(transmit_decoded, offset + 4, state.last_rx_age_us);
  write_float(transmit_decoded, offset + 8, state.position_rad);
  write_float(transmit_decoded, offset + 12, state.velocity_rad_s);
  write_float(transmit_decoded, offset + 16, state.torque_nm);
  write_float(transmit_decoded, offset + 20, state.temp_mos_c);
  write_float(transmit_decoded, offset + 24, state.temp_rotor_c);
  write_float(transmit_decoded, offset + 28, state.bus_voltage_v);
  write_float(transmit_decoded, offset + 32, state.iq_current_a);
  write_u32(transmit_decoded, offset + 36,
            static_cast<uint32_t>(state.rotation_count));
  write_float(transmit_decoded, offset + 40, state.command_position_rad);
  write_float(transmit_decoded, offset + 44, state.command_velocity_rad_s);
  write_float(transmit_decoded, offset + 48, state.command_kp);
  write_float(transmit_decoded, offset + 52, state.command_kd);
  write_float(transmit_decoded, offset + 56, state.command_torque_nm);
  write_u32(transmit_decoded, offset + 60, state.command_sequence);
  offset += USB_MOTOR_STATE_BYTES;
}

inline void write_imu_state(size_t &offset, const ImuTelemetry &state) {
  write_u32(transmit_decoded, offset, state.flags);
  write_u32(transmit_decoded, offset + 4, state.last_rx_age_us);
  write_float(transmit_decoded, offset + 8, state.quaternion_w);
  write_float(transmit_decoded, offset + 12, state.quaternion_x);
  write_float(transmit_decoded, offset + 16, state.quaternion_y);
  write_float(transmit_decoded, offset + 20, state.quaternion_z);
  write_float(transmit_decoded, offset + 24, state.acceleration_x);
  write_float(transmit_decoded, offset + 28, state.acceleration_y);
  write_float(transmit_decoded, offset + 32, state.acceleration_z);
  write_float(transmit_decoded, offset + 36, state.angular_velocity_x);
  write_float(transmit_decoded, offset + 40, state.angular_velocity_y);
  write_float(transmit_decoded, offset + 44, state.angular_velocity_z);
  write_float(transmit_decoded, offset + 48, state.temperature_c);
  write_u32(transmit_decoded, offset + 52, state.sample_counter);
  write_u32(transmit_decoded, offset + 56, state.accuracy);
  write_u32(transmit_decoded, offset + 60, state.reserved);
  offset += USB_IMU_STATE_BYTES;
}

inline void send_state() {
  uint16_t state_flags = can3_command_alive() ? USB_STATE_COMMAND_ALIVE : 0U;
  bool any_fault = false;
  bool all_initialized = true;
  Can3MotorTelemetry motor_states[CAN3_MOTOR_COUNT] = {};
  for (size_t i = 0; i < CAN3_MOTOR_COUNT; ++i) {
    can3_get_telemetry(i, motor_states[i]);
    any_fault |= (motor_states[i].flags & CAN3_STATE_FAULT) != 0U;
    all_initialized &= (motor_states[i].flags & CAN3_STATE_INITIALIZED) != 0U;
  }
  state_flags |= any_fault ? USB_STATE_ANY_FAULT : 0U;
  state_flags |= all_initialized ? USB_STATE_ALL_INITIALIZED : 0U;

  write_header(USB_MESSAGE_STATE,
               state_flags,
               state_sequence++,
               micros64(),
               USB_STATE_PAYLOAD_BYTES);
  size_t offset = USB_HEADER_BYTES;
  for (const auto &state : motor_states) {
    write_motor_state(offset, state);
  }
  write_imu_state(offset, imu_get_telemetry());
  if (offset != USB_HEADER_BYTES + USB_STATE_PAYLOAD_BYTES) {
    return;
  }
  write_u32(transmit_decoded,
            offset,
            crc32(transmit_decoded, offset));
  offset += USB_CRC_BYTES;

  const size_t encoded_length = cobs_encode(
      transmit_decoded,
      offset,
      transmit_encoded,
      sizeof(transmit_encoded));
  if (encoded_length == 0U || !Serial ||
      Serial.availableForWrite() < static_cast<int>(encoded_length + 1U)) {
    return;
  }
  Serial.write(transmit_encoded, encoded_length);
  Serial.write(static_cast<uint8_t>(0));
}

inline void receive_byte(uint8_t value) {
  if (value == 0U) {
    if (!receive_overflow && receive_length > 0U) {
      const size_t decoded_length = cobs_decode(
          receive_encoded,
          receive_length,
          receive_decoded,
          sizeof(receive_decoded));
      if (decoded_length > 0U) {
        parse_command(receive_decoded, decoded_length);
      }
    }
    receive_length = 0;
    receive_overflow = false;
    return;
  }
  if (receive_overflow) {
    return;
  }
  if (receive_length >= sizeof(receive_encoded)) {
    receive_length = 0;
    receive_overflow = true;
    return;
  }
  receive_encoded[receive_length++] = value;
}

}  // namespace usb_detail

inline void usb_init() {
  Serial.begin(115200);
  usb_detail::receive_length = 0;
  usb_detail::receive_overflow = false;
  usb_detail::last_state_us = micros();
  Serial.write(static_cast<uint8_t>(0));
}

inline void usb_loop() {
  while (Serial.available() > 0) {
    usb_detail::receive_byte(static_cast<uint8_t>(Serial.read()));
  }

  const uint32_t now_us = micros();
  if (static_cast<uint32_t>(now_us - usb_detail::last_state_us) >=
      USB_STATE_INTERVAL_US) {
    usb_detail::last_state_us = now_us;
    usb_detail::send_state();
  }
}
