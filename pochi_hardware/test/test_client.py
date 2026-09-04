from __future__ import annotations

import socket
import threading
import time

import pytest

from pochi_client import (
    COMMAND_ENABLE,
    CONTROL_MIT,
    Header,
    ImuState,
    JOINTS,
    JOINT_BY_NAME,
    MotorState,
    PochiClient,
    StatePacket,
    decode_command,
    encode_state,
)
from pochi_client.protocol import IMU_STATE_RECORD, MOTOR_STATE_RECORD


def free_udp_port() -> int:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def fake_state(sequence: int, command_sequence: int) -> StatePacket:
    motors = [
        MotorState(
            motor_id=motor_id,
            status=2,
            fault_code=0,
            flags=0x27,
            last_rx_age_us=100,
            position_rad=motor_id * 0.01,
            velocity_rad_s=0.0,
            torque_nm=0.0,
            temp_mos_c=30.0,
            temp_rotor_c=float("nan"),
            bus_voltage_v=float("nan"),
            iq_current_a=float("nan"),
            rotation_count=-(2**31),
            command_position_rad=0.0,
            command_velocity_rad_s=0.0,
            command_kp=0.0,
            command_kd=0.0,
            command_torque_nm=0.0,
            command_sequence=command_sequence,
        )
        for motor_id in range(12)
    ]
    return StatePacket(
        Header(2, 1, sequence, int(time.monotonic() * 1e6),
               MOTOR_STATE_RECORD.size * 12 + IMU_STATE_RECORD.size, 12),
        motors,
        ImuState(flags=7, quaternion_w=1.0, quaternion_x=0.0,
                 quaternion_y=0.0, quaternion_z=0.0),
    )


def test_joint_ids_follow_foot_to_body_wiring_order() -> None:
    expected = {
        "front_left_calf": 0,
        "front_left_thigh": 1,
        "front_left_hip": 2,
        "rear_left_calf": 3,
        "rear_left_thigh": 4,
        "rear_left_hip": 5,
        "rear_right_calf": 6,
        "rear_right_thigh": 7,
        "rear_right_hip": 8,
        "front_right_calf": 9,
        "front_right_thigh": 10,
        "front_right_hip": 11,
    }
    assert {name: JOINT_BY_NAME[name].motor_id for name in expected} == expected
    assert [joint.motor_id for joint in JOINTS] == list(range(12))


def test_client_sends_atomic_commands_and_receives_state() -> None:
    command_port = free_udp_port()
    state_port = free_udp_port()
    stop = threading.Event()
    seen_commands = []

    def fake_bridge() -> None:
        command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        command_socket.bind(("127.0.0.1", command_port))
        command_socket.settimeout(0.05)
        state_sequence = 0
        try:
            while not stop.is_set():
                try:
                    packet, _ = command_socket.recvfrom(2048)
                except socket.timeout:
                    continue
                command = decode_command(packet)
                seen_commands.append(command)
                state_socket.sendto(
                    encode_state(fake_state(state_sequence, command.header.sequence)),
                    ("127.0.0.1", state_port),
                )
                state_sequence += 1
        finally:
            command_socket.close()
            state_socket.close()

    thread = threading.Thread(target=fake_bridge, daemon=True)
    thread.start()
    client = PochiClient(
        command_address=("127.0.0.1", command_port),
        state_bind=("127.0.0.1", state_port),
        command_hz=100.0,
    )
    try:
        client.start()
        assert client.wait_for_state(1.0) is not None
        client.set_mit("front_left_hip", position_rad=0.3, velocity_rad_s=0.0,
                       kp=20.0, kd=0.5, torque_nm=0.0)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            if any(command.motors[2].flags & COMMAND_ENABLE for command in seen_commands):
                break
            time.sleep(0.005)
        enabled = next(command for command in seen_commands if command.motors[2].flags & COMMAND_ENABLE)
        assert len(enabled.motors) == 12
        assert enabled.motors[2].control_mode == CONTROL_MIT
        assert enabled.motors[2].position_rad == pytest.approx(0.3)
        assert client.connected
        assert client.stats().received_packets > 0
    finally:
        client.close()
        stop.set()
        thread.join(timeout=1.0)
