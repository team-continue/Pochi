"""Preview the scripted stand-up controller through the real hardware web UI.

Physics-level tracking of ``StandUpController`` is already covered by
``play_standup.py`` (MuJoCo + viser). What that does *not* exercise is the
translation from pochi_rl's simulation joint order (``FL_hip_roll``,
``FL_hip_pitch``, ``FL_knee``, ``FR_...``, ...) into the hardware's CAN-ID
commands -- nothing in the repo ties those two naming/sign conventions
together yet, and that mapping is exactly what has to be right before this
manoeuvre ever runs on a real motor.

This script runs the controller open-loop (perfect tracking assumed: the only
feedback it uses is a small droop correction, and that closed-loop behaviour
was already checked in sim) and publishes its 12 targets as fake ``StatePacket``
UDP frames -- the identical wire format ``teensy_udp_bridge.py`` produces from
a real Teensy. Point an ordinary ``pochi_hardware.web.server`` instance at
this stream and the unmodified production web UI renders it, so the CAN-ID
mapping and joint signs can be eyeballed leg by leg before touching hardware.

No Teensy, no CAN bus, no serial port, and no PochiClient *command* socket are
touched by this script.

Usage (two terminals, from the repo root):

  # 1. This script -- computes the manoeuvre, sends fake state only.
  cd pochi_rl
  uv run python scripts/sim2sim_standup_bridge.py

  # 2. A *second* web server instance, on ports the real bridge does not use,
  #    so nothing a browser does here can ever reach the real robot.
  cd ..
  uv run python -m pochi_hardware.web.server --port 8766 \
      --command-port 25000 --state-port 25001

Then open http://127.0.0.1:8766 (or tunnel it, same as the real UI). Torque
buttons and target sliders in that tab do nothing useful -- this is real
hardware-facing server code pointed at a fake data source purely to check
that positions land on the right joint, in the right direction.

Only the *names* are translated here: `MOTOR_SIGN` in
`pochi_rl.robot.pochi_constants` makes the model's joints turn the way the real
motors do, so an angle out of the controller is already the number the Teensy
takes for that motor and the values pass through untouched. What this script
still proves is that the naming half lines up -- that a target meant for the
front-left knee reaches CAN ID 0 and not some other joint.

Watch, during "approach": the splayed hip rolls should visibly close in toward
the body, and all four legs should bend the same way.
"""

from __future__ import annotations

import argparse
import math
import socket
import sys
import time
from pathlib import Path

import numpy as np

# pochi_client is pure stdlib (no external deps), so it is safe to reach into
# pochi_hardware's tree from pochi_rl's venv rather than declaring a real
# cross-package dependency for what is a one-off debug bridge.
sys.path.insert(
  0, str(Path(__file__).resolve().parents[2] / "pochi_hardware" / "client")
)

from pochi_client.joint_layout import JOINT_BY_NAME  # noqa: E402
from pochi_client.protocol import (  # noqa: E402
  MOTOR_COUNT,
  STATE_ALL_INITIALIZED,
  STATE_CAN_READY,
  STATE_COMMAND_ALIVE,
  STATE_TORQUE_ACTIVE,
  Header,
  ImuState,
  MotorState,
  StatePacket,
  encode_state,
)

from pochi_rl.control.standup import (
  COLLAPSED_JOINT_POS,
  StandUpConfig,
  StandUpController,
)
from pochi_rl.robot import JOINT_NAMES
from pochi_rl.task_spec import POCHI_TASK_SPEC

_LEG_TO_HARDWARE = {
  "FL": "front_left",
  "FR": "front_right",
  "RL": "rear_left",
  "RR": "rear_right",
}
_KIND_TO_HARDWARE = {"hip_roll": "hip", "hip_pitch": "thigh", "knee": "calf"}

# Mirrors can3.h's CAN3_MIN/MAX_POSITION_RAD (and web/server.py's
# JOINT_LIMITS_RAD), repeated here so an out-of-range mapped target is flagged
# in this preview instead of only being discovered when firmware clamps it.
_HARDWARE_LIMITS_RAD = {
  "hip": (math.radians(-40.0), math.pi / 2.0),
  "thigh": (-math.pi / 2.0, math.pi / 2.0),
  "calf": (-3.0 * math.pi / 4.0, 3.0 * math.pi / 4.0),
}

# can3.h CAN3_STATE_* bits, mirrored so the fake telemetry renders the same
# way real telemetry would ("MIT RUN" per joint, no faults).
_MOTOR_INITIALIZED = 1 << 0
_MOTOR_CONNECTED = 1 << 1
_MOTOR_FEEDBACK_VALID = 1 << 2
_MOTOR_ENABLE_REQUESTED = 1 << 3
_MOTOR_MODE_RUN = 2

# imu.h IMU_STATE_* bits.
_IMU_INITIALIZED = 1 << 0
_IMU_SAMPLE_VALID = 1 << 1
_IMU_QUAT6 = 1 << 2


def _joint_mapping() -> tuple[list[int], list[tuple[float, float]]]:
  """pochi_rl JOINT_NAMES index -> (hardware CAN id, hardware limit)."""
  motor_ids: list[int] = []
  limits: list[tuple[float, float]] = []
  for name in JOINT_NAMES:
    leg, kind = name.split("_", 1)
    hw_name = f"{_LEG_TO_HARDWARE[leg]}_{_KIND_TO_HARDWARE[kind]}"
    motor_ids.append(JOINT_BY_NAME[hw_name].motor_id)
    limits.append(_HARDWARE_LIMITS_RAD[_KIND_TO_HARDWARE[kind]])
  return motor_ids, limits


def _endpoint(value: str) -> tuple[str, int]:
  host, separator, port = value.rpartition(":")
  if not separator:
    raise argparse.ArgumentTypeError("endpoint must be HOST:PORT")
  return host, int(port)


def parse_args() -> argparse.Namespace:
  cfg = StandUpConfig()
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument(
    "--state-dest",
    type=_endpoint,
    default=("127.0.0.1", 25001),
    help="host:port to send fake state to -- must NOT be the real bridge's "
    "port (default 15001); pair with a second web.server --state-port",
  )
  p.add_argument(
    "--hold-s", type=float, default=3.0, help="pause once standing before looping"
  )
  p.add_argument("--settle-duration", type=float, default=cfg.settle_duration_s)
  p.add_argument("--approach-duration", type=float, default=cfg.approach_duration_s)
  p.add_argument("--rise-duration", type=float, default=cfg.rise_duration_s)
  return p.parse_args()


def main() -> int:
  args = parse_args()
  motor_ids, limits = _joint_mapping()

  cfg = StandUpConfig(
    settle_duration_s=args.settle_duration,
    approach_duration_s=args.approach_duration,
    rise_duration_s=args.rise_duration,
  )
  controller = StandUpController(cfg)
  # The manoeuvre starts wherever the robot lands after motors are cut from
  # the crouch and it sinks onto its belly -- COLLAPSED_JOINT_POS, not the
  # crouch pose itself (see mujoco_driver.StandUpSim.reset). Starting from
  # the crouch pose instead makes "settle"/"approach" a no-op, since that
  # pose already equals the approach phase's own target.
  start = np.array([COLLAPSED_JOINT_POS[name] for name in JOINT_NAMES])
  controller.reset(start)
  q = start.copy()

  control_dt = POCHI_TASK_SPEC.control.sim_dt * POCHI_TASK_SPEC.control.decimation
  ticks_per_cycle = int(round((cfg.total_duration_s + args.hold_s) / control_dt))

  sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
  header_flags = (
    STATE_CAN_READY | STATE_TORQUE_ACTIVE | STATE_COMMAND_ALIVE | STATE_ALL_INITIALIZED
  )

  print(
    f"Publishing fake Teensy state to {args.state_dest[0]}:{args.state_dest[1]} "
    f"at {1.0 / control_dt:.0f} Hz. Ctrl-C to stop."
  )
  print("pochi_rl joint -> hardware CAN id:")
  for name, motor_id in zip(JOINT_NAMES, motor_ids, strict=True):
    leg, kind = name.split("_", 1)
    print(
      f"  {name:14s} -> CAN {motor_id:2d}  "
      f"({_LEG_TO_HARDWARE[leg]}_{_KIND_TO_HARDWARE[kind]})"
    )

  sequence = 0
  tick = 0
  warned: set[int] = set()
  next_tick = time.monotonic()
  last_status = 0.0

  try:
    while True:
      targets = controller.act(q, control_dt)
      q = targets  # perfect tracking: no independent plant, so state == command

      motor_states: list[MotorState | None] = [None] * MOTOR_COUNT
      for i, motor_id in enumerate(motor_ids):
        position = float(targets[i])
        low, high = limits[i]
        if not (low <= position <= high) and motor_id not in warned:
          warned.add(motor_id)
          print(
            f"WARNING: CAN {motor_id} ({JOINT_NAMES[i]}) target "
            f"{math.degrees(position):.1f} deg is outside the firmware limit "
            f"[{math.degrees(low):.1f}, {math.degrees(high):.1f}] deg -- "
            "would be clamped on real hardware",
            file=sys.stderr,
          )
        motor_states[motor_id] = MotorState(
          motor_id=motor_id,
          status=_MOTOR_MODE_RUN,
          fault_code=0,
          flags=_MOTOR_INITIALIZED
          | _MOTOR_CONNECTED
          | _MOTOR_FEEDBACK_VALID
          | _MOTOR_ENABLE_REQUESTED,
          last_rx_age_us=0,
          position_rad=position,
          velocity_rad_s=0.0,
          torque_nm=0.0,
          temp_mos_c=25.0,
          temp_rotor_c=25.0,
          bus_voltage_v=48.0,
          iq_current_a=0.0,
          rotation_count=0,
          command_position_rad=position,
          command_velocity_rad_s=0.0,
          command_kp=0.0,
          command_kd=0.0,
          command_torque_nm=0.0,
          command_sequence=sequence,
        )

      imu = ImuState(
        flags=_IMU_INITIALIZED | _IMU_SAMPLE_VALID | _IMU_QUAT6,
        last_rx_age_us=0,
        quaternion_w=1.0,
        quaternion_x=0.0,
        quaternion_y=0.0,
        quaternion_z=0.0,
        acceleration_x=0.0,
        acceleration_y=0.0,
        acceleration_z=9.81,
        angular_velocity_x=0.0,
        angular_velocity_y=0.0,
        angular_velocity_z=0.0,
        temperature_c=25.0,
        sample_counter=sequence,
        accuracy=3,
      )
      header = Header(
        message_type=2,
        flags=header_flags,
        sequence=sequence,
        timestamp_us=int(time.time() * 1_000_000),
        payload_bytes=0,
        motor_count=MOTOR_COUNT,
      )
      packet = encode_state(StatePacket(header, motor_states, imu))  # type: ignore[arg-type]
      sock.sendto(packet, args.state_dest)

      now = time.monotonic()
      if now - last_status >= 0.5:
        last_status = now
        print(
          f"  t={controller._t:5.2f}s  phase={controller.phase:8s}  "
          f"target_height={controller.target_base_height:.3f} m"
        )

      sequence += 1
      tick += 1
      if tick >= ticks_per_cycle:
        controller.reset(start)
        q = start.copy()
        tick = 0

      next_tick += control_dt
      sleep_for = next_tick - time.monotonic()
      if sleep_for > 0:
        time.sleep(sleep_for)
      else:
        next_tick = time.monotonic()
  except KeyboardInterrupt:
    pass
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
