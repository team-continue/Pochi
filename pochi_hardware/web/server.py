#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import contextlib
import math
import threading
from collections.abc import AsyncIterator

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from pochi_client import COMMAND_ENABLE, CONTROL_MIT, JOINTS, JOINT_BY_ID, MotorCommand, PochiClient

MOTOR_INITIALIZED = 1 << 0
MOTOR_CONNECTED = 1 << 1
MOTOR_FEEDBACK_VALID = 1 << 2
MOTOR_ENABLE_REQUESTED = 1 << 3
MOTOR_FAULT = 1 << 6
IMU_INITIALIZED = 1 << 0
IMU_SAMPLE_VALID = 1 << 1

JOINT_LIMITS_RAD = {
    "hip": (-math.pi / 6.0, math.pi / 2.0),
    "thigh": (-math.pi / 2.0, math.pi / 2.0),
    "calf": (-3.0 * math.pi / 4.0, 3.0 * math.pi / 4.0),
}


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _motor_state_text(flags: int, fault_code: int) -> str:
    if flags & MOTOR_FAULT:
        return f"FAULT {fault_code}"
    if flags & MOTOR_ENABLE_REQUESTED:
        return "MIT ENABLED"
    if flags & MOTOR_CONNECTED:
        return "ENCODER LIVE"
    if flags & MOTOR_FEEDBACK_VALID:
        return "STALE"
    if flags & MOTOR_INITIALIZED:
        return "NO RESPONSE"
    return "NOT INITIALIZED"


def _quaternion_to_rpy_deg(w: float, x: float, y: float, z: float) -> tuple[float | None, ...]:
    if not all(math.isfinite(value) for value in (w, x, y, z)):
        return None, None, None
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return tuple(math.degrees(value) for value in (roll, pitch, yaw))


class WebControlService:
    """Own the browser control state and keep torque off until explicit opt-in."""

    def __init__(self, client: PochiClient, *, kp: float = 20.0, kd: float = 0.5) -> None:
        self.client = client
        self.kp = kp
        self.kd = kd
        self.torque_enabled = False
        self.emergency_stop = False
        self.browser_count = 0
        self._targets = {joint.motor_id: 0.0 for joint in JOINTS}
        self._lock = threading.Lock()

    def register_browser(self) -> None:
        with self._lock:
            self.browser_count += 1

    def unregister_browser(self) -> None:
        with self._lock:
            self.browser_count = max(0, self.browser_count - 1)
            should_disable = self.browser_count == 0
        if should_disable:
            self.disable_all()

    def unavailable_motor_ids(self) -> list[int]:
        state = self.client.latest_state()
        if state is None:
            return [joint.motor_id for joint in JOINTS]
        by_id = {motor.motor_id: motor for motor in state.motors}
        return [
            joint.motor_id
            for joint in JOINTS
            if (motor := by_id.get(joint.motor_id)) is None
            or not math.isfinite(motor.position_rad)
            or not (motor.flags & MOTOR_CONNECTED)
            or bool(motor.flags & MOTOR_FAULT)
            or not (
                JOINT_LIMITS_RAD[joint.joint][0]
                <= motor.position_rad
                <= JOINT_LIMITS_RAD[joint.joint][1]
            )
        ]

    def enable_all(self) -> None:
        unavailable = self.unavailable_motor_ids()
        if unavailable:
            joined = ", ".join(map(str, unavailable))
            raise ValueError(f"valid feedback is required from all motors; unavailable IDs: {joined}")

        state = self.client.latest_state()
        assert state is not None
        by_id = {motor.motor_id: motor for motor in state.motors}
        commands: list[MotorCommand] = []
        targets: dict[int, float] = {}
        for joint in JOINTS:
            motor = by_id[joint.motor_id]
            targets[joint.motor_id] = motor.position_rad
            commands.append(
                MotorCommand(
                    motor_id=joint.motor_id,
                    control_mode=CONTROL_MIT,
                    flags=COMMAND_ENABLE,
                    position_rad=motor.position_rad,
                    velocity_rad_s=0.0,
                    kp=self.kp,
                    kd=self.kd,
                    torque_nm=0.0,
                )
            )

        with self._lock:
            self.client.clear_emergency_stop()
            self.client.set_all_mit(commands)
            self._targets = targets
            self.emergency_stop = False
            self.torque_enabled = True

    def disable_all(self) -> None:
        with self._lock:
            self.client.disable_all()
            self.torque_enabled = False

    def set_target(self, motor_id: int, display_position_rad: float) -> None:
        if isinstance(motor_id, bool) or motor_id not in JOINT_BY_ID:
            raise ValueError("motorId must be an integer from 0 through 11")
        if not math.isfinite(display_position_rad):
            raise ValueError("positionRad must be finite")
        joint = JOINT_BY_ID[motor_id]
        low, high = JOINT_LIMITS_RAD[joint.joint]
        clamped = max(low, min(high, display_position_rad))
        with self._lock:
            self._targets[motor_id] = clamped
            if self.torque_enabled:
                self.client.set_mit(
                    motor_id,
                    position_rad=clamped,
                    velocity_rad_s=0.0,
                    kp=self.kp,
                    kd=self.kd,
                    torque_nm=0.0,
                    enable=True,
                )

    def emergency_stop_all(self) -> None:
        with self._lock:
            self.client.emergency_stop()
            self.emergency_stop = True
            self.torque_enabled = False

    def handle_message(self, message: dict[str, object]) -> None:
        message_type = message.get("type")
        if message_type == "torque":
            enabled = message.get("enabled")
            if not isinstance(enabled, bool):
                raise ValueError("torque.enabled must be a boolean")
            self.enable_all() if enabled else self.disable_all()
            return
        if message_type == "target":
            motor_id = message.get("motorId")
            position = message.get("positionRad")
            if isinstance(motor_id, bool) or not isinstance(motor_id, int):
                raise ValueError("target.motorId must be an integer")
            if isinstance(position, bool) or not isinstance(position, (int, float)):
                raise ValueError("target.positionRad must be a number")
            self.set_target(motor_id, float(position))
            return
        if message_type == "emergencyStop":
            self.emergency_stop_all()
            return
        raise ValueError(f"unknown message type: {message_type!r}")

    def snapshot(self) -> dict[str, object]:
        state = self.client.latest_state()
        stats = self.client.stats()
        unavailable = self.unavailable_motor_ids()
        by_id = {motor.motor_id: motor for motor in state.motors} if state else {}
        motors: list[dict[str, object]] = []
        connected_count = 0
        with self._lock:
            targets = dict(self._targets)
            torque_enabled = self.torque_enabled
            emergency_stop = self.emergency_stop

        for joint in JOINTS:
            motor = by_id.get(joint.motor_id)
            if motor is None:
                motors.append(
                    {
                        "id": joint.motor_id,
                        "name": joint.name,
                        "label": joint.label,
                        "leg": joint.leg,
                        "joint": joint.joint,
                        "positionRad": None,
                        "positionDeg": None,
                        "velocityRadS": None,
                        "torqueNm": None,
                        "targetRad": targets[joint.motor_id],
                        "tempMosC": None,
                        "busVoltageV": None,
                        "ageMs": None,
                        "flags": 0,
                        "faultCode": 0,
                        "state": "NO DATA",
                    }
                )
                continue

            position = motor.position_rad
            position_value = _finite(position)
            connected = bool(motor.flags & MOTOR_CONNECTED) and position_value is not None
            connected_count += int(connected)
            motors.append(
                {
                    "id": joint.motor_id,
                    "name": joint.name,
                    "label": joint.label,
                    "leg": joint.leg,
                    "joint": joint.joint,
                    "positionRad": position_value,
                    "positionDeg": None if position_value is None else math.degrees(position_value),
                    "velocityRadS": _finite(motor.velocity_rad_s),
                    "torqueNm": _finite(motor.torque_nm),
                    "targetRad": targets[joint.motor_id],
                    "tempMosC": _finite(motor.temp_mos_c),
                    "busVoltageV": _finite(motor.bus_voltage_v),
                    "ageMs": None
                    if motor.last_rx_age_us == 0xFFFFFFFF
                    else motor.last_rx_age_us / 1000.0,
                    "flags": motor.flags,
                    "faultCode": motor.fault_code,
                    "state": _motor_state_text(motor.flags, motor.fault_code),
                }
            )

        imu_payload: dict[str, object] = {
            "connected": False,
            "rollDeg": None,
            "pitchDeg": None,
            "yawDeg": None,
            "quaternionW": None,
            "quaternionX": None,
            "quaternionY": None,
            "quaternionZ": None,
            "ageMs": None,
            "sampleCounter": 0,
            "accuracy": 0,
        }
        if state is not None:
            imu = state.imu
            quaternion = (
                imu.quaternion_w,
                imu.quaternion_x,
                imu.quaternion_y,
                imu.quaternion_z,
            )
            roll, pitch, yaw = _quaternion_to_rpy_deg(*quaternion)
            imu_payload = {
                "connected": bool(imu.flags & IMU_INITIALIZED)
                and bool(imu.flags & IMU_SAMPLE_VALID)
                and roll is not None,
                "rollDeg": roll,
                "pitchDeg": pitch,
                "yawDeg": yaw,
                "quaternionW": _finite(quaternion[0]),
                "quaternionX": _finite(quaternion[1]),
                "quaternionY": _finite(quaternion[2]),
                "quaternionZ": _finite(quaternion[3]),
                "ageMs": None
                if imu.last_rx_age_us == 0xFFFFFFFF
                else imu.last_rx_age_us / 1000.0,
                "sampleCounter": imu.sample_counter,
                "accuracy": imu.accuracy,
            }

        state_age = self.client.state_age()
        return {
            "type": "state",
            "connected": self.client.connected,
            "connectedCount": connected_count,
            "expectedCount": len(JOINTS),
            "canEnableTorque": not unavailable and self.client.connected,
            "unavailableMotorIds": unavailable,
            "torqueEnabled": torque_enabled,
            "emergencyStop": emergency_stop,
            "stateAgeMs": state_age * 1000.0 if math.isfinite(state_age) else None,
            "motors": motors,
            "imu": imu_payload,
            "stats": {
                "stateHz": stats.state_hz,
                "rttMs": stats.rtt_ms,
                "receivedPackets": stats.received_packets,
                "droppedPackets": stats.dropped_packets,
                "invalidPackets": stats.invalid_packets,
                "lastError": stats.last_error,
            },
        }


def create_app(client: PochiClient | None = None) -> FastAPI:
    hardware_client = client or PochiClient()
    service = WebControlService(hardware_client)

    @contextlib.asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        hardware_client.disable_all()
        hardware_client.start()
        try:
            yield
        finally:
            service.disable_all()
            hardware_client.close()

    app = FastAPI(title="Pochi Web Control", lifespan=lifespan)
    app.state.control_service = service

    @app.get("/health")
    async def health() -> JSONResponse:
        snapshot = service.snapshot()
        imu = snapshot["imu"]
        assert isinstance(imu, dict)
        return JSONResponse(
            {
                "ok": True,
                "hardwareConnected": snapshot["connected"],
                "motorsConnected": snapshot["connectedCount"],
                "imuConnected": imu["connected"],
                "torqueEnabled": snapshot["torqueEnabled"],
            }
        )

    @app.websocket("/ws")
    async def websocket_endpoint(websocket: WebSocket) -> None:
        await websocket.accept()
        service.register_browser()
        try:
            while True:
                try:
                    message = await asyncio.wait_for(websocket.receive_json(), timeout=1.0 / 30.0)
                    if not isinstance(message, dict):
                        raise ValueError("WebSocket messages must be JSON objects")
                    service.handle_message(message)
                except TimeoutError:
                    pass
                except ValueError as exc:
                    await websocket.send_json({"type": "error", "message": str(exc)})
                await websocket.send_json(service.snapshot())
        except WebSocketDisconnect:
            pass
        finally:
            service.unregister_browser()

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser(description="Pochi UDP-to-WebSocket control server")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--command-host", default="127.0.0.1")
    parser.add_argument("--command-port", type=int, default=15000)
    parser.add_argument("--state-host", default="0.0.0.0")
    parser.add_argument("--state-port", type=int, default=15001)
    args = parser.parse_args()
    configured_client = PochiClient(
        command_address=(args.command_host, args.command_port),
        state_bind=(args.state_host, args.state_port),
    )
    uvicorn.run(create_app(configured_client), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
