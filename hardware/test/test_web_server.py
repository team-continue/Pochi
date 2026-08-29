from __future__ import annotations

import json
import math
import time

import pytest

from hardware.web.server import MOTOR_CONNECTED, WebControlService
from pochi_client import ClientStats, Header, ImuState, MotorState, StatePacket
from pochi_client.protocol import IMU_STATE_RECORD, MOTOR_STATE_RECORD


class FakeClient:
    def __init__(self, *, all_connected: bool = True) -> None:
        flags = MOTOR_CONNECTED if all_connected else 0
        motors = [
            MotorState(
                motor_id=motor_id,
                status=2,
                fault_code=0,
                flags=flags,
                last_rx_age_us=200,
                position_rad=motor_id * 0.01,
                velocity_rad_s=0.1,
                torque_nm=0.2,
                temp_mos_c=31.0,
                temp_rotor_c=32.0,
                bus_voltage_v=39.5,
                iq_current_a=0.4,
                rotation_count=0,
                command_position_rad=0.0,
                command_velocity_rad_s=0.0,
                command_kp=0.0,
                command_kd=0.0,
                command_torque_nm=0.0,
                command_sequence=5,
            )
            for motor_id in range(1, 13)
        ]
        self.state = StatePacket(
            Header(
                2,
                0,
                1,
                int(time.monotonic() * 1e6),
                MOTOR_STATE_RECORD.size * 12 + IMU_STATE_RECORD.size,
                12,
            ),
            motors,
            ImuState(flags=1, quaternion_w=1.0, quaternion_x=0.0, quaternion_y=0.0, quaternion_z=0.0),
        )
        self.connected = True
        self.commands: dict[int, dict[str, float | bool]] = {}
        self.disabled = False
        self.stopped = False

    def latest_state(self) -> StatePacket:
        return self.state

    def stats(self) -> ClientStats:
        return ClientStats(received_packets=10, state_hz=200.0, rtt_ms=2.5)

    def state_age(self) -> float:
        return 0.002

    def clear_emergency_stop(self) -> None:
        self.stopped = False

    def emergency_stop(self) -> None:
        self.stopped = True

    def disable_all(self) -> None:
        self.disabled = True

    def set_mit(self, motor_id: int, **values: float | bool) -> None:
        self.commands[motor_id] = values


def test_enable_uses_live_pose_then_target_updates_one_joint() -> None:
    client = FakeClient()
    service = WebControlService(client)  # type: ignore[arg-type]
    service.enable_all()

    assert service.torque_enabled
    assert len(client.commands) == 12
    assert client.commands[1]["position_rad"] == pytest.approx(0.01)

    service.set_target(1, math.radians(25.0))
    assert client.commands[1]["position_rad"] == pytest.approx(math.radians(25.0))
    assert json.dumps(service.snapshot(), allow_nan=False)


def test_torque_enable_requires_feedback_from_every_motor() -> None:
    client = FakeClient(all_connected=False)
    service = WebControlService(client)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unavailable IDs"):
        service.enable_all()
    assert not service.torque_enabled


def test_last_browser_disconnect_disables_every_motor() -> None:
    client = FakeClient()
    service = WebControlService(client)  # type: ignore[arg-type]
    service.register_browser()
    service.enable_all()
    service.unregister_browser()

    assert client.disabled
    assert not service.torque_enabled
