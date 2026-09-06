"""Run the learned Pochi-Deliberate-Walk-v0 policy on the real robot, through
the existing web control server.

This is the walking counterpart to ``run_standup_hardware.py`` and copies its
safety architecture wholesale: commands go over the *existing*
``pochi_hardware.web.server`` WebSocket (target clamping, per-motor fault
reporting, torque-off-on-disconnect, the operator's browser stays live with
its E-STOP), and the same staged ``--mode`` progression applies.

Three things make this riskier than the stand-up script and shape everything
below:

1. This checkpoint was trained against the *full-torque, full-speed* RS02
   actuator model (``POCHI_ROBOT_CFG`` -- no speed limiting, unlike the
   stand-up policy's ``POCHI_SLOW_ROBOT_CFG``), so nothing caps how hard or
   fast the motors can be driven except the trained behaviour itself. It has
   never touched real motors.
2. The policy observes ``base_lin_vel`` (the base's own translational
   velocity), which has no direct hardware sensor -- the IMU only reports
   orientation, angular velocity and acceleration. This estimates it instead
   with ``pochi_rl.control.state_estimator.BodyVelocityEstimator``, a Kalman
   filter fusing the IMU with leg odometry from the joint encoders under a
   no-slip assumption. Its contact detection is a torque threshold (no F/T
   sensors exist) and its accuracy has not been checked against anything but
   simulated data -- see ``pochi_rl/tests/test_state_estimator.py``.
3. The checkpoint does not speak the robot's joint coordinates.  It was
   trained before 9c1f1d3, back when the MJCF used the canonical leg frame;
   since that commit a joint angle in the model *is* the angle the Teensy
   reports, which negates six of the twelve.  ``--policy-joint-frame`` picks
   which, and it defaults to the older frame because that is what every
   checkpoint in ``pochi_rl/checkpoints/`` was trained in.  See
   ``_CANONICAL_SIGN`` below.

Modes, in the order they should be run on a robot that has never done this:

``dryrun``   (default) Connect, read state once, run the estimator and the
             policy once, print the resulting observation and joint targets.
             Sends nothing.
``hold``     Torque on, hold the pose the robot is already in. Identical to
             ``run_standup_hardware.py``'s -- proves gains/arming only.
``walk``     The policy loop, at 50 Hz. Requires the robot already standing
             at (or near) the default stance -- this checkpoint has no
             get-up behaviour, that is ``run_standup_hardware.py``'s job.
             Velocity commands come from the keyboard, live (see the key
             legend printed at startup); releasing back to a stop is always
             the safe option.
``standup-walk``
             Stands up (the same ``StandUpController``
             ``run_standup_hardware.py --mode standup`` uses) and, without
             ever dropping torque, hands off straight into the ``walk``
             policy loop on the same connection. For testing on the floor
             when a second operator to lift the robot for the held-up stage
             below isn't available -- it does not replace what that stage is
             for, so start with the command at zero and watch a few seconds
             before touching a forward-speed key.

Run ``walk`` for the first time with the robot lifted and fully supported by
an operator, feet off the ground -- the script cannot detect this itself.
Only move to a floor attempt once that looks sane: legs cycling in the
RL -> FL -> RR -> FR order, no fault, no runaway tracking error.

Usage (motors powered, from the repo root, web server already running):

  uv run python pochi_hardware/src/run_walk_hardware.py                 # dryrun
  uv run python pochi_hardware/src/run_walk_hardware.py --mode hold --yes
  uv run python pochi_hardware/src/run_walk_hardware.py --mode walk --yes
  uv run python pochi_hardware/src/run_walk_hardware.py --mode standup-walk --yes

Keyboard, in ``walk`` mode (matches the sim playback viewer's own GUI
bounds/step sizes, from ``pochi_rl.mjlab.deliberate_walk``'s
``DeliberateVelocityCommand.create_gui``):

  [i]/[k]  forward speed +/- 0.01 m/s   (trained range 0.00 to 0.14 m/s;
                                         -0.14 to 0.14 with --omni, for a
                                         checkpoint trained to walk backward too)
  [u]/[o]  sideways speed +/- 0.005 m/s (trained range -0.025 to 0.025 m/s)
  [j]/[l]  turn rate -/+ 0.01 rad/s     (trained range -0.15 to 0.15 rad/s)
  [g]      "walk slowly" preset: 0.10 m/s forward, no turn (the validated demo)
  [space]  stop (velocity command back to 0, 0, 0 -- the policy keeps running)
  [d]      toggle damping mode: kp -> 0, kd -> --damping-kd (2.0 by default).
           The policy is paused and the watchdog's tracking-error check is
           skipped while this is on -- the joints go slack except for a
           viscous damper, deliberately different from [q]/Ctrl-C, which
           drops torque (and thus all resistance) entirely. Toggling it back
           off resumes the policy from wherever the robot settled.
  [q]      quit (Ctrl-C also drops torque)
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

# The controller/estimator/policy loader live in pochi_rl, a separate uv
# project; this script runs in the hardware venv, so reach into its source
# tree rather than declaring a cross-project dependency -- same as
# run_standup_hardware.py and sim2sim_standup_bridge.py.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pochi_rl" / "src"))

import numpy as np
import torch
from pochi_rl.control.standup import StandUpConfig, StandUpController
from pochi_rl.control.state_estimator import (
    BodyVelocityEstimator,
    StateEstimatorConfig,
    gravity_direction_body,
)
from pochi_rl.policy_io import load_actor
from pochi_rl.robot import DEFAULT_JOINT_POS, JOINT_NAMES, MOTOR_SIGN
from pochi_rl.task_spec import POCHI_TASK_SPEC
from run_standup_hardware import (
    Link,
    _raw_terminal,
    arm,
    check_ready,
    disarm,
    joint_mapping,
    measured_positions,
    preflight_trajectory,
    send_targets,
    watchdog,
)

# --- Pochi-Deliberate-Walk-v0's own numbers ----------------------------------
# Mirrors pochi_rl.mjlab.deliberate_walk.DeliberateWalkSpec /
# DeliberateVelocityCommand.create_gui exactly (see
# origin/feature/deliberate-walk, not merged into this branch -- this script
# only needs the few constants below, not the training-side env code, which
# needs mjlab). Keep these in step if that spec ever changes.
GAIT_PERIOD_S = 2.8
COMMAND_THRESHOLD = 0.025
COMMAND_RANGES = {
    "vx": (0.0, 0.14),
    "vy": (-0.025, 0.025),
    "wz": (-0.15, 0.15),
}
COMMAND_STEPS = {"vx": 0.01, "vy": 0.005, "wz": 0.01}
WALK_SLOWLY_COMMAND = (0.10, 0.0, 0.0)

GAIT_PHASE_DIM = 2
OBS_DIM = POCHI_TASK_SPEC.observations.policy_dim + GAIT_PHASE_DIM

# Walking is more dynamic than the stand-up ramp; expect to retune this at the
# held-up stage rather than trust it as-is.
DEFAULT_MAX_ERROR_DEG = 25.0
# How far any joint may sit from the default stance before "walk" refuses to
# start -- this checkpoint has no get-up behaviour of its own.
STANCE_TOLERANCE_DEG = 15.0

_DEFAULT_JOINT_POS = np.array([DEFAULT_JOINT_POS[name] for name in JOINT_NAMES])

# --- The checkpoint's joint frame is not the robot's --------------------------
# Every checkpoint that exists today was trained before 9c1f1d3, the commit
# that made a joint angle in the MJCF *be* the angle the Teensy reports for
# that motor: it negated the joint axis of the six joints whose RS02 is bolted
# in backwards (`MOTOR_SIGN == -1`) and negated `DEFAULT_JOINT_POS` to match.
# Before it the model used the canonical leg frame -- all four legs about
# +x/+y, with the front/rear stance mirror spelled out in the defaults instead.
# `Pochi-Deliberate-Walk-v0` lives on origin/feature/deliberate-walk, which
# branches from 32d6ea3, i.e. from before that change.
#
# The two frames describe the same physical poses and differ by exactly
# `MOTOR_SIGN`, which is -1 on six joints:
#
#     FL_hip_roll  FL_hip_pitch  FL_knee  FR_hip_roll  RL_hip_pitch  RL_knee
#
#     q_canonical = MOTOR_SIGN * q_motor       (joint velocities the same way)
#     q_motor     = MOTOR_SIGN * q_canonical
#
# The two DEFAULT_JOINT_POS transform the same way, so the observation's
# *relative* joint position needs one multiply and no re-referencing:
#
#     (q - q0)_canonical = MOTOR_SIGN * (q - q0)_motor
#
# and an action, being a delta on the default stance, converts the same way on
# its way back out.
#
# Two things deliberately do *not* get the sign:
#
#   * `last_action`, which goes back into the observation exactly as the policy
#     emitted it -- it is the network's own output, not a joint angle.
#   * the state estimator, which takes motor-frame angles and applies
#     `MOTOR_SIGN` itself before the leg kinematics, and returns a body
#     velocity, which has no joint frame at all.
#
# Running the wrong frame is not a subtle degradation -- six of the twelve
# joints move the wrong way -- but it does not necessarily look violent, and
# walk_log2.csv is what it looks like: an upright robot with clean joint
# tracking, moving its legs, and an estimated forward velocity of zero.
_CANONICAL_SIGN = np.array([MOTOR_SIGN[name] for name in JOINT_NAMES])
_IDENTITY_SIGN = np.ones(len(JOINT_NAMES))


def frame_sign(policy_joint_frame: str) -> np.ndarray:
    """Motor frame -> the checkpoint's joint frame, per joint.  See above."""
    return _CANONICAL_SIGN if policy_joint_frame == "canonical" else _IDENTITY_SIGN


def joint_targets(action: np.ndarray, sign: np.ndarray) -> np.ndarray:
    """Motor-frame position targets from a policy action.  The action is a
    delta on the default stance in the *policy's* frame, so it is the delta
    that gets the sign, not the sum."""
    return _DEFAULT_JOINT_POS + POCHI_TASK_SPEC.control.action_scale * (sign * action)


# What each slice of the 50-dim observation is, for the log's sidecar -- an
# offline replay needs this to take the vector apart again.
OBS_LAYOUT = (
    ("base_lin_vel", 3),
    ("base_ang_vel", 3),
    ("projected_gravity", 3),
    ("velocity_commands", 3),
    ("joint_pos_rel", 12),
    ("joint_vel", 12),
    ("last_action", 12),
    ("gait_phase", GAIT_PHASE_DIM),
)


def imu_reading(snapshot: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(quaternion_wxyz, gyro_body, accel_body) from a Link snapshot.

    Raises if the IMU isn't reporting -- there is no safe fallback for any of
    these three.
    """
    imu = snapshot["imu"]
    if not imu.get("connected"):
        raise RuntimeError("IMU not connected -- cannot estimate base velocity")

    def _field(name: str) -> float:
        value = imu[name]
        if value is None:
            raise RuntimeError(
                f"IMU field '{name}' is missing (firmware isn't reporting it -- "
                "check pochi_hardware/firmware/teensy41_test/src/imu.h enables "
                "raw accel/gyro DMP sensors, not just GAME_ROTATION_VECTOR)"
            )
        return float(value)

    quaternion = np.array(
        [
            _field("quaternionW"),
            _field("quaternionX"),
            _field("quaternionY"),
            _field("quaternionZ"),
        ]
    )
    gyro = np.array(
        [
            _field("angularVelocityX"),
            _field("angularVelocityY"),
            _field("angularVelocityZ"),
        ]
    )
    accel = np.array(
        [_field("accelerationX"), _field("accelerationY"), _field("accelerationZ")]
    )
    if not (
        np.all(np.isfinite(quaternion))
        and np.all(np.isfinite(gyro))
        and np.all(np.isfinite(accel))
    ):
        raise RuntimeError("IMU reports non-finite values")
    return quaternion, gyro, accel


def measured_velocities(snapshot: dict, motor_ids: list[int]) -> np.ndarray:
    by_id = {motor["id"]: motor for motor in snapshot["motors"]}
    out = np.zeros(len(motor_ids))
    for i, motor_id in enumerate(motor_ids):
        velocity = by_id[motor_id]["velocityRadS"]
        out[i] = 0.0 if velocity is None else float(velocity)
    return out


def measured_torques(snapshot: dict, motor_ids: list[int]) -> np.ndarray:
    by_id = {motor["id"]: motor for motor in snapshot["motors"]}
    out = np.zeros(len(motor_ids))
    for i, motor_id in enumerate(motor_ids):
        torque = by_id[motor_id]["torqueNm"]
        out[i] = 0.0 if torque is None else float(torque)
    return out


def gait_phase(elapsed_s: float, command: np.ndarray) -> np.ndarray:
    """(sin, cos) of the deliberate-walk gait clock -- see module docstring."""
    moving = (math.hypot(command[0], command[1]) + abs(command[2])) > COMMAND_THRESHOLD
    if not moving:
        return np.zeros(2)
    angle = 2.0 * math.pi * elapsed_s / GAIT_PERIOD_S
    return np.array([math.sin(angle), math.cos(angle)])


def build_observation(
    *,
    base_lin_vel: np.ndarray,
    base_ang_vel: np.ndarray,
    gravity_body: np.ndarray,
    command: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    last_action: np.ndarray,
    elapsed_s: float,
    sign: np.ndarray,
) -> np.ndarray:
    """The 50-dim actor input, in exactly the order the training env built it:
    base_lin_vel, base_ang_vel, projected_gravity, velocity_commands,
    joint_pos_rel, joint_vel_rel, last_action, gait_phase.

    ``joint_pos``/``joint_vel`` arrive in motor coordinates; ``sign`` carries
    them into the frame the checkpoint was trained in (see ``frame_sign``)."""
    joint_pos_rel = sign * (joint_pos - _DEFAULT_JOINT_POS)
    phase = gait_phase(elapsed_s, command)
    return np.concatenate(
        [
            base_lin_vel,
            base_ang_vel,
            gravity_body,
            command,
            joint_pos_rel,
            sign * joint_vel,
            last_action,
            phase,
        ]
    ).astype(np.float32)


class Teleop:
    """Live (vx, vy, wz) command, nudged by keypresses. See the module
    docstring for the key legend; bounds/steps match the sim playback GUI."""

    def __init__(self, *, vx_range: tuple[float, float] = COMMAND_RANGES["vx"]) -> None:
        self.command = np.zeros(3)
        # Only vx varies by checkpoint (forward-only vs. --omni's symmetric
        # range) -- vy and wz are already symmetric and shared by all of them.
        self.ranges = {**COMMAND_RANGES, "vx": vx_range}

    def handle_key(self, key: str) -> None:
        if key == "i":
            self._nudge("vx", +COMMAND_STEPS["vx"])
        elif key == "k":
            self._nudge("vx", -COMMAND_STEPS["vx"])
        elif key == "o":
            self._nudge("vy", +COMMAND_STEPS["vy"])
        elif key == "u":
            self._nudge("vy", -COMMAND_STEPS["vy"])
        elif key == "l":
            self._nudge("wz", +COMMAND_STEPS["wz"])
        elif key == "j":
            self._nudge("wz", -COMMAND_STEPS["wz"])
        elif key == "g":
            self.command = np.array(WALK_SLOWLY_COMMAND)
        elif key == " ":
            self.command = np.zeros(3)

    def _nudge(self, axis: str, delta: float) -> None:
        index = {"vx": 0, "vy": 1, "wz": 2}[axis]
        low, high = self.ranges[axis]
        self.command[index] = min(max(self.command[index] + delta, low), high)


async def run_dryrun(link: Link, args: argparse.Namespace) -> int:
    motor_ids, _ = joint_mapping()
    snapshot = await link.wait_for_snapshot()

    problems = check_ready(snapshot, motor_ids)
    if problems:
        print("not ready:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    actor = load_actor(args.checkpoint)
    estimator = BodyVelocityEstimator(
        StateEstimatorConfig(contact_torque_threshold_nm=args.contact_torque_threshold)
    )

    positions = measured_positions(snapshot, motor_ids)
    velocities = measured_velocities(snapshot, motor_ids)
    torques = measured_torques(snapshot, motor_ids)
    quaternion, gyro, accel = imu_reading(snapshot)

    v_body = estimator.step(
        accel_body=accel,
        gyro_body=gyro,
        quaternion_wxyz=quaternion,
        joint_pos=positions,
        joint_vel=velocities,
        joint_torque=torques,
        dt=1.0 / POCHI_TASK_SPEC.control.policy_hz,
    )
    g_body = gravity_direction_body(quaternion)
    command = np.array(WALK_SLOWLY_COMMAND)
    last_action = np.zeros(12)
    sign = frame_sign(args.policy_joint_frame)
    obs = build_observation(
        base_lin_vel=v_body,
        base_ang_vel=gyro,
        gravity_body=g_body,
        command=command,
        joint_pos=positions,
        joint_vel=velocities,
        last_action=last_action,
        elapsed_s=0.0,
        sign=sign,
    )
    with torch.inference_mode():
        action = actor(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
    targets = joint_targets(action, sign)

    print(f"checkpoint:             {args.checkpoint}")
    print(
        f"policy joint frame:     {args.policy_joint_frame} "
        f"(sign {' '.join(f'{value:+.0f}' for value in sign)})"
    )
    print(f"estimated base_lin_vel: {v_body}")
    print(f"projected_gravity:      {g_body}")
    print(f"observation ({obs.shape[0]} dims): {obs}")
    print(f"raw action:              {action}")
    print("resulting joint targets (deg), by name:")
    for name, target in zip(JOINT_NAMES, targets, strict=True):
        print(f"  {name:14s} {math.degrees(target):7.1f}")
    print("\ndryrun: nothing was sent. Re-run with --mode hold, then --mode walk.")
    return 0


async def run_hold(link: Link, args: argparse.Namespace) -> int:
    motor_ids, _ = joint_mapping()
    snapshot = await link.wait_for_snapshot()
    problems = check_ready(snapshot, motor_ids)
    if problems:
        print("not ready:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    targets = measured_positions(snapshot, motor_ids)
    try:
        await arm(link, args.kp, args.kd)
    except Exception as exc:  # noqa: BLE001 - already disarmed by arm()
        print(f"could not arm: {exc}", file=sys.stderr)
        return 1
    print(f"holding the measured pose for {args.duration:.1f}s -- Ctrl-C to stop")
    max_error_rad = math.radians(args.max_error_deg)
    deadline = time.monotonic() + args.duration
    try:
        while time.monotonic() < deadline:
            snapshot = link.fresh_snapshot()
            watchdog(snapshot, motor_ids, targets, max_error_rad)
            await asyncio.sleep(0.1)
    except KeyboardInterrupt:
        pass
    except Exception as exc:  # noqa: BLE001
        print(f"\nABORT: {exc}", file=sys.stderr)
        await disarm(link, emergency=True)
        return 1
    print("\ndropping torque")
    await disarm(link)
    return 0


def run_metadata(args: argparse.Namespace, sign: np.ndarray) -> dict:
    """Everything an offline replay of the log needs and cannot recover from
    the rows: which network produced the actions, in which joint frame, under
    which gains, against which stance."""
    checkpoint = Path(args.checkpoint)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    return {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "policy_joint_frame": args.policy_joint_frame,
        "omni": bool(args.omni),
        "vx_range": [-0.14, 0.14] if args.omni else list(COMMAND_RANGES["vx"]),
        "joint_sign": [float(value) for value in sign],
        "joint_names": list(JOINT_NAMES),
        "default_joint_pos_motor_frame": [float(q) for q in _DEFAULT_JOINT_POS],
        "action_scale": float(POCHI_TASK_SPEC.control.action_scale),
        "policy_hz": float(POCHI_TASK_SPEC.control.policy_hz),
        "obs_layout": [{"name": name, "dim": dim} for name, dim in OBS_LAYOUT],
        "kp": float(args.kp),
        "kd": float(args.kd),
        "damping_kd": float(args.damping_kd),
        "max_error_deg": float(args.max_error_deg),
        "contact_torque_threshold": float(args.contact_torque_threshold),
        "gait_period_s": GAIT_PERIOD_S,
        "started_unix": time.time(),
    }


class TickLogger:
    """Per-tick CSV log of the walk loop -- one row per 50 Hz control tick,
    for diagnosing things a 0.5s status line can't show (a single-tick
    tracking-error spike, exactly what tripped the watchdog above)."""

    def __init__(self, path: str, *, metadata: dict) -> None:
        # Everything that is constant for the run goes in a sidecar rather than
        # into every row: which checkpoint, which joint frame, which gains.
        # A log that does not say those cannot be replayed offline, which is
        # the whole point of having it.
        Path(path).with_suffix(".meta.json").write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n"
        )
        # Held open for the loop's lifetime and closed in its `finally`, not a
        # `with` block -- the loop calls `.log()` many times before `.close()`.
        self._file = open(path, "w", newline="")  # noqa: SIM115
        self._writer = csv.writer(self._file)
        self._writer.writerow(
            ["t", "wall_clock", "mode", "cmd_vx", "cmd_vy", "cmd_wz"]
            + ["v_est_x", "v_est_y", "v_est_z"]
            + ["gyro_x", "gyro_y", "gyro_z"]
            + ["gravity_x", "gravity_y", "gravity_z"]
            + ["max_tracking_error_deg"]
            + [f"pos_{name}" for name in JOINT_NAMES]
            + [f"vel_{name}" for name in JOINT_NAMES]
            + [f"torque_{name}" for name in JOINT_NAMES]
            + [f"target_{name}" for name in JOINT_NAMES]
            + [f"action_{name}" for name in JOINT_NAMES]
            + [f"obs_{index}" for index in range(OBS_DIM)]
        )

    def log(
        self,
        *,
        elapsed: float,
        mode: str,
        command: np.ndarray,
        v_body: np.ndarray,
        gyro: np.ndarray,
        gravity_body: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        torques: np.ndarray,
        previous_targets: np.ndarray,
        new_targets: np.ndarray,
        action: np.ndarray | None,
        obs: np.ndarray | None,
    ) -> None:
        """One row per tick, including damping ticks -- a gap in `t` used to be
        the only trace those left, which made a log impossible to line up with
        wall-clock events.  `action`/`obs` are None on those (no policy ran);
        the columns stay, empty."""
        max_error_deg = float(np.degrees(np.abs(positions - previous_targets)).max())
        self._writer.writerow(
            [elapsed, time.time(), mode, *command, *v_body, *gyro, *gravity_body]
            + [max_error_deg]
            + list(positions)
            + list(velocities)
            + list(torques)
            + list(new_targets)
            + (list(action) if action is not None else [""] * len(JOINT_NAMES))
            + (list(obs) if obs is not None else [""] * OBS_DIM)
        )
        # Flushed every tick: the runs worth reading are the ones that end in
        # an abort, and a half-written buffer loses exactly the last seconds.
        self._file.flush()

    def close(self) -> None:
        self._file.close()


async def _set_gains(link: Link, kp: float, kd: float) -> None:
    """Change the MIT-mode gains on already-armed motors, live -- the same
    message ``arm()`` sends before its own enable step, without touching
    torque-enable state. Used to drop into/out of damping mode (kp=0) inside
    the walk loop without a disarm/rearm cycle."""
    await link.send({"type": "gains", "kp": kp, "kd": kd})


async def _walk_loop(
    link: Link,
    args: argparse.Namespace,
    motor_ids: list[int],
    initial_targets: np.ndarray,
) -> int:
    """The policy loop itself, at 50 Hz. Shared by ``run_walk`` (robot already
    standing when this is called) and ``run_standup_then_walk`` (robot has
    just finished standing up, on the *same* armed connection -- torque is
    never dropped between the two)."""
    actor = load_actor(args.checkpoint)
    estimator = BodyVelocityEstimator(
        StateEstimatorConfig(contact_torque_threshold_nm=args.contact_torque_threshold)
    )
    vx_range = (-0.14, 0.14) if args.omni else COMMAND_RANGES["vx"]
    teleop = Teleop(vx_range=vx_range)
    control_dt = 1.0 / POCHI_TASK_SPEC.control.policy_hz
    max_error_rad = math.radians(args.max_error_deg)
    sign = frame_sign(args.policy_joint_frame)

    print(
        f"policy joint frame: {args.policy_joint_frame} "
        f"(sign {' '.join(f'{value:+.0f}' for value in sign)})"
    )
    print(f"vx range: {vx_range[0]:+.2f} to {vx_range[1]:+.2f} m/s")
    print(
        "[i]/[k] forward +/-   [u]/[o] sideways +/-   [j]/[l] turn +/-\n"
        "[g] walk slowly (0.10 m/s fwd)   [space] stop   [d] damping   [q] quit"
    )
    logger = (
        TickLogger(args.log_file, metadata=run_metadata(args, sign))
        if args.log_file
        else None
    )
    if logger:
        print(f"logging every tick to {args.log_file}")

    loop = asyncio.get_event_loop()
    keys: asyncio.Queue[str] = asyncio.Queue()
    loop.add_reader(sys.stdin.fileno(), lambda: keys.put_nowait(sys.stdin.read(1)))

    last_action = np.zeros(12)
    targets = initial_targets.copy()
    elapsed = 0.0
    next_tick = time.monotonic()
    last_report = 0.0
    damping = False
    try:
        with _raw_terminal():
            while True:
                while not keys.empty():
                    key = keys.get_nowait()
                    if key.lower() == "q" or key == "\x03":
                        raise KeyboardInterrupt
                    if key.lower() == "d":
                        damping = not damping
                        if damping:
                            await _set_gains(link, 0.0, args.damping_kd)
                            print(
                                f"\ndamping mode ON (kp=0, kd={args.damping_kd}) "
                                "-- policy paused, joints free except for damping"
                            )
                        else:
                            snapshot = link.fresh_snapshot()
                            targets = measured_positions(snapshot, motor_ids)
                            last_action = np.zeros(12)
                            await _set_gains(link, args.kp, args.kd)
                            print(
                                f"\ndamping mode OFF -- resuming the policy at "
                                f"kp={args.kp} kd={args.kd} from wherever it settled"
                            )
                        continue
                    teleop.handle_key(key.lower() if key != " " else " ")

                if damping:
                    snapshot = link.fresh_snapshot()
                    positions = measured_positions(snapshot, motor_ids)
                    await send_targets(link, motor_ids, positions)
                    targets = positions
                    if logger:
                        # Damping mode is the fallback the operator reaches
                        # for when something is already wrong, so logging it
                        # must not be able to abort it: a dead IMU here costs
                        # three columns, not the robot's torque.
                        try:
                            quaternion, gyro, _ = imu_reading(snapshot)
                            gravity = gravity_direction_body(quaternion)
                        except RuntimeError:
                            gyro = gravity = np.full(3, np.nan)
                        logger.log(
                            elapsed=elapsed,
                            mode="damping",
                            command=teleop.command,
                            v_body=np.zeros(3),
                            gyro=gyro,
                            gravity_body=gravity,
                            positions=positions,
                            velocities=measured_velocities(snapshot, motor_ids),
                            torques=measured_torques(snapshot, motor_ids),
                            previous_targets=positions,
                            new_targets=positions,
                            action=None,
                            obs=None,
                        )
                    now = time.monotonic()
                    if now - last_report >= 0.5:
                        last_report = now
                        print(f"  t={elapsed:6.2f}s  DAMPING (kp=0)", end="\r")
                    elapsed += control_dt
                    next_tick += control_dt
                    sleep_for = next_tick - time.monotonic()
                    if sleep_for > 0:
                        await asyncio.sleep(sleep_for)
                    else:
                        next_tick = time.monotonic()
                    continue

                snapshot = link.fresh_snapshot()

                positions = measured_positions(snapshot, motor_ids)
                velocities = measured_velocities(snapshot, motor_ids)
                torques = measured_torques(snapshot, motor_ids)
                quaternion, gyro, accel = imu_reading(snapshot)

                v_body = estimator.step(
                    accel_body=accel,
                    gyro_body=gyro,
                    quaternion_wxyz=quaternion,
                    joint_pos=positions,
                    joint_vel=velocities,
                    joint_torque=torques,
                    dt=control_dt,
                )
                g_body = gravity_direction_body(quaternion)
                obs = build_observation(
                    base_lin_vel=v_body,
                    base_ang_vel=gyro,
                    gravity_body=g_body,
                    command=teleop.command,
                    joint_pos=positions,
                    joint_vel=velocities,
                    last_action=last_action,
                    elapsed_s=elapsed,
                    sign=sign,
                )
                with torch.inference_mode():
                    action = (
                        actor(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
                    )
                last_action = action
                previous_targets = targets
                targets = joint_targets(action, sign)

                if logger:
                    logger.log(
                        elapsed=elapsed,
                        mode="walk",
                        command=teleop.command,
                        v_body=v_body,
                        gyro=gyro,
                        gravity_body=g_body,
                        positions=positions,
                        velocities=velocities,
                        torques=torques,
                        previous_targets=previous_targets,
                        new_targets=targets,
                        action=action,
                        obs=obs,
                    )

                # Checked here, after everything above is already computed
                # (and logged) for this tick, rather than at the top of the
                # loop -- the only effect of the ordering is that the tick
                # that trips it still gets logged, for exactly this kind of
                # after-the-fact diagnosis.
                watchdog(snapshot, motor_ids, previous_targets, max_error_rad)
                await send_targets(link, motor_ids, targets)

                now = time.monotonic()
                if now - last_report >= 0.5:
                    last_report = now
                    print(
                        f"  t={elapsed:6.2f}s  cmd=({teleop.command[0]:+.3f},"
                        f"{teleop.command[1]:+.3f},{teleop.command[2]:+.3f})  "
                        f"v_est=({v_body[0]:+.3f},{v_body[1]:+.3f},{v_body[2]:+.3f})",
                        end="\r",
                    )

                elapsed += control_dt
                next_tick += control_dt
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    await asyncio.sleep(sleep_for)
                else:
                    next_tick = time.monotonic()
    except KeyboardInterrupt:
        print("\ninterrupted -- dropping torque")
        await disarm(link)
        return 130
    except Exception as exc:  # noqa: BLE001 - any surprise means stop the robot
        print(f"\nABORT: {exc}", file=sys.stderr)
        await disarm(link, emergency=True)
        return 1
    finally:
        loop.remove_reader(sys.stdin.fileno())
        if logger:
            logger.close()


async def run_walk(link: Link, args: argparse.Namespace) -> int:
    motor_ids, _ = joint_mapping()
    snapshot = await link.wait_for_snapshot()
    problems = check_ready(snapshot, motor_ids)
    if problems:
        print("not ready:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    start = measured_positions(snapshot, motor_ids)
    offset_deg = np.degrees(np.abs(start - _DEFAULT_JOINT_POS))
    if offset_deg.max() > STANCE_TOLERANCE_DEG:
        worst = JOINT_NAMES[int(offset_deg.argmax())]
        print(
            f"refusing to walk: {worst} is {offset_deg.max():.1f} deg from the default "
            f"stance (tolerance {STANCE_TOLERANCE_DEG:.0f} deg). This checkpoint has no "
            "get-up behaviour -- stand the robot up first (run_standup_hardware.py), or "
            "use --mode standup-walk to do both in one connection."
        )
        return 1

    try:
        await arm(link, args.kp, args.kd)
    except Exception as exc:  # noqa: BLE001 - already disarmed by arm()
        print(f"could not arm: {exc}", file=sys.stderr)
        return 1

    return await _walk_loop(link, args, motor_ids, initial_targets=start)


async def run_standup_then_walk(link: Link, args: argparse.Namespace) -> int:
    """Stand up, then hand off straight into the walk policy -- one
    connection, one ``arm()``, torque never dropped in between.

    This exists for testing on the floor without a second operator to lift
    the robot for the held-up stage: it skips straight from a proven,
    slow manoeuvre (``run_standup_hardware.py``'s own ``StandUpController``)
    into the untested walk policy, with no gap where the robot could be
    unpowered mid-transition. It does *not* replace the held-up test's
    purpose -- the first walk targets this sends are still the first ones
    this checkpoint has ever sent to real motors. Start with the command at
    zero (the default) and watch a few seconds of standing-still tracking
    before touching a forward-speed key.
    """
    motor_ids, limits = joint_mapping()
    snapshot = await link.wait_for_snapshot()
    problems = check_ready(snapshot, motor_ids)
    if problems:
        print("not ready:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    start = measured_positions(snapshot, motor_ids)
    standup_cfg = StandUpConfig(
        settle_duration_s=args.settle_duration,
        approach_duration_s=args.approach_duration,
        rise_duration_s=args.rise_duration,
    )
    _, worst = preflight_trajectory(start, standup_cfg, motor_ids, limits)
    if worst and not args.allow_clamp:
        print("targets that the firmware would clamp:")
        for name, (over, value) in worst.items():
            print(
                f"  {name:14s} reaches {math.degrees(value):7.1f} deg, "
                f"{math.degrees(over):.2f} deg outside"
            )
        print(
            "\nrefusing to stand up: see the clamped targets above "
            "(--allow-clamp to override)"
        )
        return 1

    controller = StandUpController(standup_cfg)
    controller.reset(start)
    control_dt = 1.0 / POCHI_TASK_SPEC.control.policy_hz
    max_error_rad = math.radians(args.max_error_deg)

    try:
        await arm(link, args.kp, args.kd)
    except Exception as exc:  # noqa: BLE001 - already disarmed by arm()
        print(f"could not arm: {exc}", file=sys.stderr)
        return 1
    print(
        f"standing up over {standup_cfg.total_duration_s:.2f}s -- Ctrl-C drops torque"
    )

    targets = start.copy()
    elapsed = 0.0
    next_tick = time.monotonic()
    try:
        while elapsed < standup_cfg.total_duration_s + args.hold_s:
            snapshot = link.fresh_snapshot()
            watchdog(snapshot, motor_ids, targets, max_error_rad)
            measured = measured_positions(snapshot, motor_ids)
            targets = controller.act(measured, control_dt)
            await send_targets(link, motor_ids, targets)

            elapsed += control_dt
            next_tick += control_dt
            sleep_for = next_tick - time.monotonic()
            if sleep_for > 0:
                await asyncio.sleep(sleep_for)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        print("\ninterrupted during stand-up -- dropping torque")
        await disarm(link)
        return 130
    except Exception as exc:  # noqa: BLE001 - any surprise means stop the robot
        print(f"\nABORT during stand-up: {exc}", file=sys.stderr)
        await disarm(link, emergency=True)
        return 1

    print(
        "standing -- handing off to the walk policy (same connection, torque held on)"
    )
    return await _walk_loop(link, args, motor_ids, initial_targets=targets)


def parse_args() -> argparse.Namespace:
    standup_cfg = StandUpConfig()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--mode",
        choices=("dryrun", "hold", "walk", "standup-walk"),
        default="dryrun",
    )
    p.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    p.add_argument(
        "--yes", action="store_true", help="required for anything that moves"
    )
    p.add_argument(
        "--checkpoint",
        default=str(
            Path(__file__).resolve().parents[2]
            / "pochi_rl"
            / "checkpoints"
            / "pochi_deliberate_walk_model_950.pt"
        ),
    )
    p.add_argument(
        "--omni",
        action="store_true",
        help="checkpoint was trained with a symmetric vx range "
        "(-0.14 to 0.14 m/s) and can walk backward -- default is the "
        "forward-only range (0.00 to 0.14) every non-omni checkpoint in "
        "pochi_rl/checkpoints/ was trained on. vy/wz ranges are unaffected: "
        "both are already symmetric and shared by every checkpoint.",
    )
    p.add_argument(
        "--policy-joint-frame",
        choices=("canonical", "motor"),
        default="canonical",
        help="which joint frame the checkpoint speaks. 'canonical' for one "
        "trained before 9c1f1d3 -- which is every checkpoint in "
        "pochi_rl/checkpoints/, and the default; 'motor' for one retrained "
        "since, whose joint angles are already the ones the Teensy reports. "
        "Getting this wrong inverts six of the twelve joints and the robot "
        "will not walk (see _CANONICAL_SIGN)",
    )
    # Conservative hardware defaults shared with the Teensy and web UI.
    # Explicit --kp/--kd options can still be used for controlled tuning.
    p.add_argument("--kp", type=float, default=14.0)
    p.add_argument("--kd", type=float, default=0.05)
    p.add_argument("--max-error-deg", type=float, default=DEFAULT_MAX_ERROR_DEG)
    p.add_argument("--contact-torque-threshold", type=float, default=3.0)
    p.add_argument(
        "--damping-kd",
        type=float,
        default=2.0,
        help="walk/standup-walk only: kd used while damping mode ([d]) is on "
        "(kp forced to 0 -- a viscous damper toward zero joint velocity, not "
        "a hard stop and not a full disarm)",
    )
    p.add_argument(
        "--log-file",
        default=None,
        help="walk/standup-walk only: CSV path, one row per 50 Hz tick "
        "(command, estimated velocity, gyro, gravity, per-joint position/"
        "velocity/torque/target/action, the full 50-dim observation, and the "
        "tracking error the watchdog checks). A <name>.meta.json sidecar gets "
        "the checkpoint, its hash, the joint frame and the gains",
    )
    p.add_argument("--duration", type=float, default=10.0, help="hold mode only")
    # standup-walk mode only -- same defaults/meaning as run_standup_hardware.py.
    p.add_argument(
        "--settle-duration", type=float, default=standup_cfg.settle_duration_s
    )
    p.add_argument(
        "--approach-duration", type=float, default=standup_cfg.approach_duration_s
    )
    p.add_argument("--rise-duration", type=float, default=standup_cfg.rise_duration_s)
    p.add_argument("--hold-s", type=float, default=2.0)
    p.add_argument("--allow-clamp", action="store_true")
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    async with Link(args.url) as link:
        if args.mode == "hold":
            return await run_hold(link, args)
        if args.mode == "walk":
            return await run_walk(link, args)
        if args.mode == "standup-walk":
            return await run_standup_then_walk(link, args)
        return await run_dryrun(link, args)


def main() -> int:
    args = parse_args()
    if args.mode != "dryrun" and not args.yes:
        print(f"--mode {args.mode} moves the robot; pass --yes to confirm")
        return 2
    try:
        return asyncio.run(main_async(args))
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
