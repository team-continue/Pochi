from __future__ import annotations

import dataclasses
import math
import struct
import time
import zlib
from collections.abc import Iterable

MAGIC = b"PCHI"
VERSION = 1
MESSAGE_COMMAND = 1
MESSAGE_STATE = 2
MOTOR_COUNT = 12

FLAG_EMERGENCY_STOP = 1 << 0
STATE_COMMAND_ALIVE = 1 << 0
STATE_ANY_FAULT = 1 << 1
STATE_ALL_INITIALIZED = 1 << 2
STATE_CAN_INITIALIZING = 1 << 3
STATE_CAN_READY = 1 << 4
STATE_REARM_REQUIRED = 1 << 5
STATE_TORQUE_ACTIVE = 1 << 6

CONTROL_DISABLED = 0
CONTROL_MIT = 1
COMMAND_ENABLE = 1 << 0
COMMAND_CLEAR_FAULT = 1 << 1

HEADER = struct.Struct("<4sBBHIQHBB")
COMMAND_RECORD = struct.Struct("<BBH5f")
MOTOR_STATE_RECORD = struct.Struct("<BBBBI7fi5fI")
IMU_STATE_RECORD = struct.Struct("<II4f3f3ffIII")
CRC = struct.Struct("<I")

COMMAND_PAYLOAD_BYTES = COMMAND_RECORD.size * MOTOR_COUNT
STATE_PAYLOAD_BYTES = MOTOR_STATE_RECORD.size * MOTOR_COUNT + IMU_STATE_RECORD.size
COMMAND_PACKET_BYTES = HEADER.size + COMMAND_PAYLOAD_BYTES + CRC.size
STATE_PACKET_BYTES = HEADER.size + STATE_PAYLOAD_BYTES + CRC.size

assert HEADER.size == 24
assert COMMAND_RECORD.size == 24
assert MOTOR_STATE_RECORD.size == 64
assert IMU_STATE_RECORD.size == 64
assert COMMAND_PACKET_BYTES == 316
assert STATE_PACKET_BYTES == 860


class ProtocolError(ValueError):
    pass


@dataclasses.dataclass(slots=True)
class Header:
    message_type: int
    flags: int
    sequence: int
    timestamp_us: int
    payload_bytes: int
    motor_count: int


@dataclasses.dataclass(slots=True)
class MotorCommand:
    motor_id: int
    control_mode: int = CONTROL_DISABLED
    flags: int = 0
    position_rad: float = 0.0
    velocity_rad_s: float = 0.0
    kp: float = 0.0
    kd: float = 0.0
    torque_nm: float = 0.0


@dataclasses.dataclass(slots=True)
class MotorState:
    motor_id: int
    status: int
    fault_code: int
    flags: int
    last_rx_age_us: int
    position_rad: float
    velocity_rad_s: float
    torque_nm: float
    temp_mos_c: float
    temp_rotor_c: float
    bus_voltage_v: float
    iq_current_a: float
    rotation_count: int
    command_position_rad: float
    command_velocity_rad_s: float
    command_kp: float
    command_kd: float
    command_torque_nm: float
    command_sequence: int


@dataclasses.dataclass(slots=True)
class ImuState:
    flags: int = 0
    last_rx_age_us: int = 0xFFFFFFFF
    quaternion_w: float = math.nan
    quaternion_x: float = math.nan
    quaternion_y: float = math.nan
    quaternion_z: float = math.nan
    acceleration_x: float = math.nan
    acceleration_y: float = math.nan
    acceleration_z: float = math.nan
    angular_velocity_x: float = math.nan
    angular_velocity_y: float = math.nan
    angular_velocity_z: float = math.nan
    temperature_c: float = math.nan
    sample_counter: int = 0
    accuracy: int = 0
    reserved: int = 0


@dataclasses.dataclass(slots=True)
class CommandPacket:
    header: Header
    motors: list[MotorCommand]


@dataclasses.dataclass(slots=True)
class StatePacket:
    header: Header
    motors: list[MotorState]
    imu: ImuState


def disabled_commands() -> list[MotorCommand]:
    return [MotorCommand(motor_id=motor_id) for motor_id in range(MOTOR_COUNT)]


def _pack_header(message_type: int, flags: int, sequence: int, payload_bytes: int) -> bytes:
    return HEADER.pack(
        MAGIC,
        VERSION,
        message_type,
        flags,
        sequence & 0xFFFFFFFF,
        time.monotonic_ns() // 1_000,
        payload_bytes,
        MOTOR_COUNT,
        0,
    )


def _append_crc(data: bytes) -> bytes:
    return data + CRC.pack(zlib.crc32(data) & 0xFFFFFFFF)


def _validate_packet(packet: bytes, expected_type: int, expected_payload: int) -> Header:
    expected_size = HEADER.size + expected_payload + CRC.size
    if len(packet) != expected_size:
        raise ProtocolError(f"packet size {len(packet)} != {expected_size}")
    values = HEADER.unpack_from(packet)
    magic, version, message_type, flags, sequence, timestamp_us, payload_bytes, motor_count, _ = values
    if magic != MAGIC or version != VERSION:
        raise ProtocolError("protocol magic or version mismatch")
    if message_type != expected_type:
        raise ProtocolError(f"unexpected message type {message_type}")
    if payload_bytes != expected_payload or motor_count != MOTOR_COUNT:
        raise ProtocolError("payload shape mismatch")
    expected_crc = CRC.unpack_from(packet, len(packet) - CRC.size)[0]
    actual_crc = zlib.crc32(packet[:-CRC.size]) & 0xFFFFFFFF
    if actual_crc != expected_crc:
        raise ProtocolError("CRC mismatch")
    return Header(message_type, flags, sequence, timestamp_us, payload_bytes, motor_count)


def encode_command(
    commands: Iterable[MotorCommand],
    sequence: int,
    *,
    emergency_stop: bool = False,
) -> bytes:
    commands = list(commands)
    if len(commands) != MOTOR_COUNT or {command.motor_id for command in commands} != set(range(MOTOR_COUNT)):
        raise ProtocolError("a command must contain motor IDs 0 through 11 exactly once")
    payload = bytearray()
    for command in commands:
        values = (
            command.position_rad,
            command.velocity_rad_s,
            command.kp,
            command.kd,
            command.torque_nm,
        )
        if command.control_mode not in (CONTROL_DISABLED, CONTROL_MIT) or not all(map(math.isfinite, values)):
            raise ProtocolError(f"invalid command for motor {command.motor_id}")
        payload += COMMAND_RECORD.pack(
            command.motor_id,
            command.control_mode,
            command.flags,
            *values,
        )
    flags = FLAG_EMERGENCY_STOP if emergency_stop else 0
    return _append_crc(_pack_header(MESSAGE_COMMAND, flags, sequence, len(payload)) + payload)


def decode_command(packet: bytes) -> CommandPacket:
    header = _validate_packet(packet, MESSAGE_COMMAND, COMMAND_PAYLOAD_BYTES)
    motors: list[MotorCommand] = []
    offset = HEADER.size
    for _ in range(MOTOR_COUNT):
        motors.append(MotorCommand(*COMMAND_RECORD.unpack_from(packet, offset)))
        offset += COMMAND_RECORD.size
    if {motor.motor_id for motor in motors} != set(range(MOTOR_COUNT)):
        raise ProtocolError("command motor IDs are incomplete or duplicated")
    return CommandPacket(header, motors)


def encode_state(state: StatePacket) -> bytes:
    if len(state.motors) != MOTOR_COUNT:
        raise ProtocolError("state must contain 12 motors")
    payload = bytearray()
    for motor in state.motors:
        payload += MOTOR_STATE_RECORD.pack(*dataclasses.astuple(motor))
    payload += IMU_STATE_RECORD.pack(*dataclasses.astuple(state.imu))
    header = HEADER.pack(
        MAGIC,
        VERSION,
        MESSAGE_STATE,
        state.header.flags,
        state.header.sequence & 0xFFFFFFFF,
        state.header.timestamp_us,
        len(payload),
        MOTOR_COUNT,
        0,
    )
    return _append_crc(header + payload)


def decode_state(packet: bytes) -> StatePacket:
    header = _validate_packet(packet, MESSAGE_STATE, STATE_PAYLOAD_BYTES)
    motors: list[MotorState] = []
    offset = HEADER.size
    for _ in range(MOTOR_COUNT):
        motors.append(MotorState(*MOTOR_STATE_RECORD.unpack_from(packet, offset)))
        offset += MOTOR_STATE_RECORD.size
    imu = ImuState(*IMU_STATE_RECORD.unpack_from(packet, offset))
    return StatePacket(header, motors, imu)


def cobs_encode(data: bytes) -> bytes:
    output = bytearray([0])
    code_index = 0
    code = 1
    for value in data:
        if value == 0:
            output[code_index] = code
            code_index = len(output)
            output.append(0)
            code = 1
            continue
        output.append(value)
        code += 1
        if code == 0xFF:
            output[code_index] = code
            code_index = len(output)
            output.append(0)
            code = 1
    output[code_index] = code
    return bytes(output)


def cobs_decode(data: bytes) -> bytes:
    output = bytearray()
    index = 0
    while index < len(data):
        code = data[index]
        if code == 0:
            raise ProtocolError("zero byte inside COBS frame")
        index += 1
        end = index + code - 1
        if end > len(data):
            raise ProtocolError("truncated COBS frame")
        output.extend(data[index:end])
        index = end
        if code != 0xFF and index < len(data):
            output.append(0)
    return bytes(output)


def encode_usb_frame(packet: bytes) -> bytes:
    return cobs_encode(packet) + b"\x00"


class CobsStreamDecoder:
    def __init__(self, maximum_encoded_bytes: int = 1024) -> None:
        self._buffer = bytearray()
        self._maximum = maximum_encoded_bytes
        self._overflow = False

    def feed(self, data: bytes) -> list[bytes]:
        frames: list[bytes] = []
        for value in data:
            if value == 0:
                if self._buffer and not self._overflow:
                    try:
                        frames.append(cobs_decode(bytes(self._buffer)))
                    except ProtocolError:
                        pass
                self._buffer.clear()
                self._overflow = False
            elif not self._overflow:
                self._buffer.append(value)
                if len(self._buffer) > self._maximum:
                    self._buffer.clear()
                    self._overflow = True
        return frames
