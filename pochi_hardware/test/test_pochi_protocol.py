from __future__ import annotations

import dataclasses
import math
import socket

import pytest

from pochi_client.protocol import (
    COMMAND_ENABLE,
    COMMAND_PACKET_BYTES,
    CONTROL_MIT,
    FLAG_EMERGENCY_STOP,
    HEADER,
    IMU_STATE_RECORD,
    MOTOR_STATE_RECORD,
    STATE_PACKET_BYTES,
    STATE_CAN_INITIALIZING,
    STATE_CAN_READY,
    STATE_REARM_REQUIRED,
    STATE_TORQUE_ACTIVE,
    CobsStreamDecoder,
    Header,
    ImuState,
    MotorState,
    ProtocolError,
    StatePacket,
    cobs_decode,
    cobs_encode,
    decode_command,
    decode_state,
    disabled_commands,
    encode_command,
    encode_state,
    encode_usb_frame,
)


def make_state() -> StatePacket:
    motors = [
        MotorState(
            motor_id=motor_id,
            status=2,
            fault_code=0,
            flags=0x27,
            last_rx_age_us=100 + motor_id,
            position_rad=motor_id * 0.1,
            velocity_rad_s=motor_id * 0.2,
            torque_nm=motor_id * 0.3,
            temp_mos_c=35.0,
            temp_rotor_c=math.nan,
            bus_voltage_v=math.nan,
            iq_current_a=math.nan,
            rotation_count=-(2**31),
            command_position_rad=0.0,
            command_velocity_rad_s=0.0,
            command_kp=0.0,
            command_kd=0.0,
            command_torque_nm=0.0,
            command_sequence=41,
        )
        for motor_id in range(12)
    ]
    imu = ImuState(flags=7, last_rx_age_us=50, quaternion_w=1.0,
                   quaternion_x=0.0, quaternion_y=0.0, quaternion_z=0.0)
    return StatePacket(
        Header(message_type=2, flags=5, sequence=42, timestamp_us=123456,
               payload_bytes=MOTOR_STATE_RECORD.size * 12 + IMU_STATE_RECORD.size,
               motor_count=12),
        motors,
        imu,
    )


def test_fixed_packet_sizes_and_command_round_trip() -> None:
    commands = disabled_commands()
    commands[0] = dataclasses.replace(
        commands[0],
        control_mode=CONTROL_MIT,
        flags=COMMAND_ENABLE,
        position_rad=1.25,
        velocity_rad_s=-2.5,
        kp=30.0,
        kd=0.8,
        torque_nm=3.0,
    )
    packet = encode_command(commands, 123, emergency_stop=True)
    decoded = decode_command(packet)
    assert len(packet) == COMMAND_PACKET_BYTES == 316
    assert decoded.header.sequence == 123
    assert decoded.header.flags == FLAG_EMERGENCY_STOP
    assert decoded.motors[0].motor_id == commands[0].motor_id
    assert decoded.motors[0].control_mode == CONTROL_MIT
    assert decoded.motors[0].position_rad == pytest.approx(1.25)
    assert decoded.motors[0].velocity_rad_s == pytest.approx(-2.5)
    assert decoded.motors[0].kp == pytest.approx(30.0)
    assert decoded.motors[0].kd == pytest.approx(0.8)
    assert decoded.motors[0].torque_nm == pytest.approx(3.0)


def test_state_round_trip_includes_all_motors_and_imu() -> None:
    state = make_state()
    state.header.flags = (
        STATE_CAN_INITIALIZING
        | STATE_CAN_READY
        | STATE_REARM_REQUIRED
        | STATE_TORQUE_ACTIVE
    )
    packet = encode_state(state)
    decoded = decode_state(packet)
    assert len(packet) == STATE_PACKET_BYTES == 860
    assert [motor.motor_id for motor in decoded.motors] == list(range(12))
    assert decoded.motors[11].position_rad == pytest.approx(1.1)
    assert decoded.imu.quaternion_w == pytest.approx(1.0)
    assert decoded.header.flags == state.header.flags


def test_cobs_and_stream_chunking() -> None:
    packet = encode_state(make_state())
    encoded = cobs_encode(packet)
    assert b"\x00" not in encoded
    assert cobs_decode(encoded) == packet

    stream = CobsStreamDecoder()
    framed = encode_usb_frame(packet)
    assert stream.feed(b"startup text\x00" + framed[:111]) == []
    assert stream.feed(framed[111:]) == [packet]


def test_crc_corruption_is_rejected() -> None:
    packet = bytearray(encode_command(disabled_commands(), 1))
    packet[HEADER.size + 4] ^= 0x20
    with pytest.raises(ProtocolError, match="CRC"):
        decode_command(bytes(packet))


def test_udp_datagram_preserves_raw_packet() -> None:
    receiver = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        receiver.bind(("127.0.0.1", 0))
        receiver.settimeout(1.0)
        packet = encode_state(make_state())
        sender.sendto(packet, receiver.getsockname())
        received, _ = receiver.recvfrom(2048)
        assert decode_state(received).header.sequence == 42
    finally:
        sender.close()
        receiver.close()
