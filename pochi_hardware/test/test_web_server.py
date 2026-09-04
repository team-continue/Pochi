from __future__ import annotations

import json
import math
import time

import pytest

from pochi_hardware.web.server import MOTOR_CONNECTED, WebControlService
from pochi_client import (
    STATE_CAN_INITIALIZING,
    STATE_CAN_READY,
    STATE_REARM_REQUIRED,
    STATE_TORQUE_ACTIVE,
    ClientStats,
    Header,
    ImuState,
    MotorState,
    StatePacket,
)
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
            for motor_id in range(12)
        ]
        self.state = StatePacket(
            Header(
                2,
                STATE_CAN_READY | STATE_REARM_REQUIRED,
                1,
                int(time.monotonic() * 1e6),
                MOTOR_STATE_RECORD.size * 12 + IMU_STATE_RECORD.size,
                12,
            ),
            motors,
            ImuState(flags=3, quaternion_w=1.0, quaternion_x=0.0, quaternion_y=0.0, quaternion_z=0.0),
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

    def disable(self, motor_id: int) -> None:
        self.commands.pop(motor_id, None)

    def set_mit(self, motor_id: int, **values: float | bool) -> None:
        self.commands[motor_id] = values

    def set_all_mit(self, commands: list[object]) -> None:
        for command in commands:
            motor_id = getattr(command, "motor_id")
            self.commands[motor_id] = {
                "position_rad": getattr(command, "position_rad"),
                "velocity_rad_s": getattr(command, "velocity_rad_s"),
                "kp": getattr(command, "kp"),
                "kd": getattr(command, "kd"),
                "torque_nm": getattr(command, "torque_nm"),
                "enable": True,
            }


def test_enable_uses_live_pose_then_target_updates_one_joint() -> None:
    client = FakeClient()
    service = WebControlService(client)  # type: ignore[arg-type]
    service.enable_all()

    assert service.torque_enabled
    assert len(client.commands) == 12
    assert client.commands[0]["position_rad"] == pytest.approx(0.0)

    service.set_target(0, math.radians(25.0))
    assert client.commands[0]["position_rad"] == pytest.approx(math.radians(25.0))
    service.set_target(1, math.radians(25.0))
    assert client.commands[1]["position_rad"] == pytest.approx(math.radians(25.0))
    snapshot = service.snapshot()
    assert snapshot["motors"][1]["positionRad"] == pytest.approx(0.01)
    assert snapshot["motors"][1]["velocityRadS"] == pytest.approx(0.1)
    assert snapshot["motors"][1]["torqueNm"] == pytest.approx(0.2)
    assert json.dumps(snapshot, allow_nan=False)


def test_torque_enable_requires_feedback_from_every_motor() -> None:
    client = FakeClient(all_connected=False)
    service = WebControlService(client)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="unavailable IDs"):
        service.enable_all()
    assert not service.torque_enabled


def test_individual_motor_torque_only_enables_selected_id() -> None:
    client = FakeClient()
    service = WebControlService(client)  # type: ignore[arg-type]

    service.set_motor_enabled(8, True)
    snapshot = service.snapshot()

    assert service.torque_enabled
    assert client.commands[8]["position_rad"] == pytest.approx(0.08)
    assert snapshot["requestedTorqueCount"] == 1
    assert snapshot["motors"][8]["torqueRequested"] is True
    assert snapshot["motors"][7]["torqueRequested"] is False

    service.set_motor_enabled(8, False)
    assert not service.torque_enabled
    assert 8 not in client.commands


def test_pending_motors_track_live_pose_during_staggered_enable() -> None:
    client = FakeClient()
    service = WebControlService(client)  # type: ignore[arg-type]
    service.enable_all()

    client.state.header.flags = STATE_CAN_READY
    client.state.motors[0].status = 2
    client.state.motors[1].status = 0
    client.state.motors[0].position_rad = 0.25
    client.state.motors[1].position_rad = 0.35
    service.snapshot()

    # A motor already in Run keeps the target captured when it was enabled.
    assert client.commands[0]["position_rad"] == pytest.approx(0.0)
    # A motor still waiting for its turn follows the latest passive pose.
    assert client.commands[1]["position_rad"] == pytest.approx(0.35)


def test_torque_enable_requires_firmware_ready() -> None:
    client = FakeClient()
    client.state.header.flags = STATE_CAN_INITIALIZING | STATE_REARM_REQUIRED
    service = WebControlService(client)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="still initializing"):
        service.enable_all()
    assert not service.torque_enabled


def test_torque_enable_rejects_a_motor_outside_joint_limits() -> None:
    client = FakeClient()
    client.state.motors[0].position_rad = 3.0
    service = WebControlService(client)  # type: ignore[arg-type]

    assert service.unavailable_motor_ids() == [0]
    with pytest.raises(ValueError, match="unavailable IDs: 0"):
        service.enable_all()
    assert not service.torque_enabled


def test_gui_target_limits_match_teensy_joint_limits() -> None:
    client = FakeClient()
    service = WebControlService(client)  # type: ignore[arg-type]
    service.enable_all()

    service.set_target(0, math.radians(200.0))
    service.set_target(1, math.radians(120.0))
    service.set_target(2, math.radians(-60.0))

    assert client.commands[0]["position_rad"] == pytest.approx(3.0 * math.pi / 4.0)
    assert client.commands[1]["position_rad"] == pytest.approx(math.pi / 2.0)
    assert client.commands[2]["position_rad"] == pytest.approx(math.radians(-40.0))


def test_global_mit_gains_update_enabled_motors() -> None:
    client = FakeClient()
    service = WebControlService(client)  # type: ignore[arg-type]
    service.set_motor_enabled(5, True)

    service.handle_message({"type": "gains", "kp": 40.0, "kd": 1.0})

    assert client.commands[5]["kp"] == pytest.approx(40.0)
    assert client.commands[5]["kd"] == pytest.approx(1.0)
    assert service.snapshot()["gains"] == {"kp": 40.0, "kd": 1.0}


def test_zero_all_targets_updates_every_target_and_enabled_motor() -> None:
    client = FakeClient()
    service = WebControlService(client)  # type: ignore[arg-type]
    service.set_motor_enabled(5, True)
    service.set_target(5, 0.4)
    service.set_target(6, 0.3)

    service.handle_message({"type": "zeroTargets"})

    snapshot = service.snapshot()
    assert all(motor["targetRad"] == 0.0 for motor in snapshot["motors"])
    assert client.commands[5]["position_rad"] == 0.0
    assert 6 not in client.commands


@pytest.mark.parametrize(
    ("kp", "kd"),
    [(-1.0, 1.0), (5001.0, 1.0), (40.0, -0.1), (40.0, 101.0)],
)
def test_global_mit_gains_reject_out_of_range_values(kp: float, kd: float) -> None:
    client = FakeClient()
    service = WebControlService(client)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="gains"):
        service.handle_message({"type": "gains", "kp": kp, "kd": kd})


def test_last_browser_disconnect_disables_every_motor() -> None:
    client = FakeClient()
    service = WebControlService(client)  # type: ignore[arg-type]
    service.register_browser()
    service.enable_all()
    service.unregister_browser()

    assert client.disabled
    assert not service.torque_enabled


def test_firmware_rearm_latch_forces_web_request_off_after_active() -> None:
    client = FakeClient()
    service = WebControlService(client)  # type: ignore[arg-type]
    service.enable_all()

    client.state.header.flags = STATE_CAN_READY | STATE_TORQUE_ACTIVE
    assert service.snapshot()["torqueEnabled"] is True
    assert service.torque_enabled

    client.disabled = False
    client.state.header.flags = STATE_CAN_INITIALIZING | STATE_REARM_REQUIRED
    snapshot = service.snapshot()
    assert snapshot["torqueEnabled"] is False
    assert snapshot["torqueRequested"] is False
    assert snapshot["safetyState"] == "INITIALIZING"
    assert client.disabled
    assert not service.torque_enabled
