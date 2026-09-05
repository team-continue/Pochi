"""Run the learned Pochi-Deliberate-Walk-v0 policy on the real robot, through
the existing web control server.

This is the walking counterpart to ``run_standup_hardware.py`` and copies its
safety architecture wholesale: commands go over the *existing*
``pochi_hardware.web.server`` WebSocket (target clamping, per-motor fault
reporting, torque-off-on-disconnect, the operator's browser stays live with
its E-STOP), and the same staged ``--mode`` progression applies.

Two things make this riskier than the stand-up script and shape everything
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

  [i]/[k]  forward speed +/- 0.01 m/s   (trained range 0.00 to 0.14 m/s)
  [u]/[o]  sideways speed +/- 0.005 m/s (trained range -0.025 to 0.025 m/s)
  [j]/[l]  turn rate -/+ 0.01 rad/s     (trained range -0.15 to 0.15 rad/s)
  [g]      "walk slowly" preset: 0.10 m/s forward, no turn (the validated demo)
  [space]  stop (0, 0, 0)
  [q]      quit (Ctrl-C also drops torque)
"""

from __future__ import annotations

import argparse
import asyncio
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
from pochi_rl.robot import DEFAULT_JOINT_POS, JOINT_NAMES
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
) -> np.ndarray:
    """The 50-dim actor input, in exactly the order the training env built it:
    base_lin_vel, base_ang_vel, projected_gravity, velocity_commands,
    joint_pos_rel, joint_vel_rel, last_action, gait_phase."""
    joint_pos_rel = joint_pos - _DEFAULT_JOINT_POS
    phase = gait_phase(elapsed_s, command)
    return np.concatenate(
        [
            base_lin_vel,
            base_ang_vel,
            gravity_body,
            command,
            joint_pos_rel,
            joint_vel,
            last_action,
            phase,
        ]
    ).astype(np.float32)


class Teleop:
    """Live (vx, vy, wz) command, nudged by keypresses. See the module
    docstring for the key legend; bounds/steps match the sim playback GUI."""

    def __init__(self) -> None:
        self.command = np.zeros(3)

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
        low, high = COMMAND_RANGES[axis]
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
    obs = build_observation(
        base_lin_vel=v_body,
        base_ang_vel=gyro,
        gravity_body=g_body,
        command=command,
        joint_pos=positions,
        joint_vel=velocities,
        last_action=last_action,
        elapsed_s=0.0,
    )
    with torch.inference_mode():
        action = actor(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
    targets = _DEFAULT_JOINT_POS + POCHI_TASK_SPEC.control.action_scale * action

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
    teleop = Teleop()
    control_dt = 1.0 / POCHI_TASK_SPEC.control.policy_hz
    max_error_rad = math.radians(args.max_error_deg)

    print(
        "[i]/[k] forward +/-   [u]/[o] sideways +/-   [j]/[l] turn +/-\n"
        "[g] walk slowly (0.10 m/s fwd)   [space] stop   [q] quit"
    )

    loop = asyncio.get_event_loop()
    keys: asyncio.Queue[str] = asyncio.Queue()
    loop.add_reader(sys.stdin.fileno(), lambda: keys.put_nowait(sys.stdin.read(1)))

    last_action = np.zeros(12)
    targets = initial_targets.copy()
    elapsed = 0.0
    next_tick = time.monotonic()
    last_report = 0.0
    try:
        with _raw_terminal():
            while True:
                while not keys.empty():
                    key = keys.get_nowait()
                    if key.lower() == "q" or key == "\x03":
                        raise KeyboardInterrupt
                    teleop.handle_key(key.lower() if key != " " else " ")

                snapshot = link.fresh_snapshot()
                watchdog(snapshot, motor_ids, targets, max_error_rad)

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
                )
                with torch.inference_mode():
                    action = (
                        actor(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
                    )
                last_action = action
                targets = (
                    _DEFAULT_JOINT_POS + POCHI_TASK_SPEC.control.action_scale * action
                )
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
    # Matches POCHI_ACTUATOR's training-time stiffness/damping
    # (pochi_rl.mjlab.entity_cfg) -- the policy was trained assuming joints
    # track this closely, so start here rather than the stand-up script's
    # much softer 40/1.0.
    p.add_argument("--kp", type=float, default=60.0)
    p.add_argument("--kd", type=float, default=1.5)
    p.add_argument("--max-error-deg", type=float, default=DEFAULT_MAX_ERROR_DEG)
    p.add_argument("--contact-torque-threshold", type=float, default=3.0)
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
