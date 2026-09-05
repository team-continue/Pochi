"""Run the scripted stand-up on the real robot, through the web control server.

This is the first thing in the repo that commands the real motors with a
trajectory rather than a slider, so it is deliberately staged: every mode below
is a superset of the one before it, and the default mode sends nothing at all.

Commands go over the *existing* ``pochi_hardware.web.server`` WebSocket rather
than straight down ``PochiClient``.  That costs a message per joint per tick,
which is affordable because the manoeuvre is quasi-static by design (it peaks at
0.43 rad/s in simulation), and it buys the whole safety layer that server
already implements and that a standalone script would have to reimplement:
targets clamped to ``JOINT_LIMITS_RAD``, per-motor fault reporting, torque
dropped automatically if this process dies, and -- the point -- the operator's
browser stays connected to the same server with its TORQUE OFF and E-STOP
buttons live while this runs.  It also means the state UDP port stays with the
web server instead of being fought over.

Modes, in the order they should be run on a robot that has never done this:

``dryrun``   (default) Connect, read state, pre-flight the exact trajectory the
             measured pose would produce, print it.  Sends nothing.
``hold``     Torque on, hold the pose the robot is already in.  Proves the gains
             are sane and that arming does not make the robot jump.
``jog``      Move one named joint a few degrees and back.  Proves sign and
             CAN-ID mapping on the real machine, one motor at a time.
``standup``  The full manoeuvre.
``teleop``   Close to the crouch stance once, then step between down (belly on
             the floor), crouch, and stand on each keypress
             ([u] stand, [c] crouch, [d] lie down, [q] quit).  crouch<->stand
             is the same IK height ramp as standing up, run backwards;
             down<->crouch is the ``approach`` phase's hip-roll join, run
             backwards.  Only adjacent rungs are reachable directly -- from
             stand, [d] is refused until you go through crouch.  [a] toggles
             repeating stand<->crouch on its own (a squat), reversing each
             time the current ramp settles; any other key cancels it.
             Requires an interactive terminal (reads raw keypresses from
             stdin).

The controller is reset from the *measured* joint positions, not from
``COLLAPSED_JOINT_POS``, so the robot is stood up from wherever it actually is.
That also keeps the start pose inside the firmware's limits: the canned
collapsed pose sits 0.1-2.5 deg outside them on eight of the twelve joints and
would be silently clamped.

Aborting -- Ctrl-C, a fault, a stale state, or a joint drifting further from its
target than ``--max-error-deg`` -- disables torque, which drops the robot.  That
is the right failure mode here only because the manoeuvre never lifts it far off
the floor; it is not a safe abort for a robot at full stance height.

Usage (motors powered, from the repo root, web server already running):

  uv run python pochi_hardware/src/run_standup_hardware.py                 # dryrun
  uv run python pochi_hardware/src/run_standup_hardware.py --mode hold --yes
  uv run python pochi_hardware/src/run_standup_hardware.py --mode jog \
      --joint FL_knee --delta-deg 5 --yes
  uv run python pochi_hardware/src/run_standup_hardware.py --mode standup --yes
  uv run python pochi_hardware/src/run_standup_hardware.py --mode teleop --yes
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import math
import sys
import termios
import time
import tty
from pathlib import Path

# The controller lives in pochi_rl, which is a separate uv project; this script
# runs in the hardware venv (websockets, pochi_client), so reach into its source
# tree rather than declaring a cross-project dependency.  The mirror image of
# what pochi_rl/scripts/sim2sim_standup_bridge.py does to reach pochi_client.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "pochi_rl" / "src"))

import numpy as np  # noqa: E402
import websockets  # noqa: E402
from pochi_client.joint_layout import JOINT_BY_NAME  # noqa: E402

from pochi_rl.control.standup import (  # noqa: E402
    COLLAPSED_JOINT_POS,
    PoseSequencer,
    PoseSequencerConfig,
    StandUpConfig,
    StandUpController,
)
from pochi_rl.robot import JOINT_NAMES  # noqa: E402
from pochi_rl.task_spec import POCHI_STANDUP_SPEC, POCHI_TASK_SPEC  # noqa: E402

# Same translation as sim2sim_standup_bridge.py; keep the two in step (they are
# both small enough that a shared home is not worth a cross-package import).
_LEG_TO_HARDWARE = {
    "FL": "front_left",
    "FR": "front_right",
    "RL": "rear_left",
    "RR": "rear_right",
}
_KIND_TO_HARDWARE = {"hip_roll": "hip", "hip_pitch": "thigh", "knee": "calf"}

# Mirrors web/server.py's JOINT_LIMITS_RAD (itself a mirror of can3.h), so the
# pre-flight can say "this would be clamped" before anything moves.
_HARDWARE_LIMITS_RAD = {
    "hip": (-3.0 * math.pi / 4.0, 3.0 * math.pi / 4.0),
    "thigh": (-3.0 * math.pi / 4.0, 3.0 * math.pi / 4.0),
    "calf": (-math.pi, math.pi),
}

# Simulated worst case at kp=40 is 9.6 deg of tracking error, at the hip rolls
# during the rise.  Twice that is a fault, not a heavy leg.
DEFAULT_MAX_ERROR_DEG = 20.0
# The server pushes a snapshot after every message it handles, so state this old
# means the link, the bridge, or the Teensy has stopped.
MAX_STATE_AGE_S = 0.5
# The firmware confirms each enable over CAN, so arming is not instant.
ARM_TIMEOUT_S = 10.0


def joint_mapping() -> tuple[list[int], list[tuple[float, float]]]:
    """pochi_rl JOINT_NAMES index -> (hardware CAN id, firmware limit)."""
    motor_ids: list[int] = []
    limits: list[tuple[float, float]] = []
    for name in JOINT_NAMES:
        leg, kind = name.split("_", 1)
        hw_name = f"{_LEG_TO_HARDWARE[leg]}_{_KIND_TO_HARDWARE[kind]}"
        motor_ids.append(JOINT_BY_NAME[hw_name].motor_id)
        limits.append(_HARDWARE_LIMITS_RAD[_KIND_TO_HARDWARE[kind]])
    return motor_ids, limits


class Link:
    """One WebSocket to the control server, with the latest snapshot cached."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.socket: websockets.ClientConnection | None = None
        self.snapshot: dict | None = None
        self.snapshot_at = 0.0
        self._reader: asyncio.Task | None = None

    async def __aenter__(self) -> Link:
        self.socket = await websockets.connect(self.url, max_size=None)
        self._reader = asyncio.create_task(self._read_loop())
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._reader is not None:
            self._reader.cancel()
        if self.socket is not None:
            await self.socket.close()

    async def _read_loop(self) -> None:
        assert self.socket is not None
        async for raw in self.socket:
            message = json.loads(raw)
            if message.get("type") == "error":
                print(f"server error: {message.get('message')}", file=sys.stderr)
                continue
            self.snapshot = message
            self.snapshot_at = time.monotonic()

    async def send(self, message: dict) -> None:
        assert self.socket is not None
        await self.socket.send(json.dumps(message))

    async def wait_for_snapshot(self, timeout: float = 3.0) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if self.snapshot is not None:
                return self.snapshot
            await asyncio.sleep(0.02)
        raise TimeoutError(f"no snapshot from {self.url} within {timeout:.1f}s")

    def fresh_snapshot(self) -> dict:
        if self.snapshot is None:
            raise RuntimeError("no snapshot yet")
        age = time.monotonic() - self.snapshot_at
        if age > MAX_STATE_AGE_S:
            raise RuntimeError(f"state is {age:.2f}s stale -- link or Teensy is down")
        return self.snapshot


def measured_positions(snapshot: dict, motor_ids: list[int]) -> np.ndarray:
    """Encoder angles in JOINT_NAMES order, straight from the snapshot."""
    by_id = {motor["id"]: motor for motor in snapshot["motors"]}
    out = np.zeros(len(motor_ids))
    for i, motor_id in enumerate(motor_ids):
        position = by_id[motor_id]["positionRad"]
        if position is None:
            raise RuntimeError(f"CAN {motor_id} has no valid position")
        out[i] = float(position)
    return out


def check_ready(snapshot: dict, motor_ids: list[int]) -> list[str]:
    """Everything that would make commanding this robot a bad idea."""
    problems: list[str] = []
    if not snapshot.get("connected"):
        problems.append("no telemetry from the Teensy bridge")
    by_id = {motor["id"]: motor for motor in snapshot["motors"]}
    for motor_id in motor_ids:
        motor = by_id.get(motor_id)
        if motor is None:
            problems.append(f"CAN {motor_id} missing from telemetry")
            continue
        if motor["positionRad"] is None:
            problems.append(f"CAN {motor_id} ({motor['name']}) has no CAN feedback")
            continue
        if motor["faultCode"]:
            problems.append(
                f"CAN {motor_id} ({motor['name']}) reports fault {motor['faultCode']}"
            )
        if not motor["canEnableTorque"]:
            problems.append(
                f"CAN {motor_id} ({motor['name']}) refuses torque: {motor['state']}, "
                f"at {motor['positionDeg']:.1f} deg"
            )
    return problems


def preflight_trajectory(
    start: np.ndarray,
    cfg: StandUpConfig,
    motor_ids: list[int],
    limits: list[tuple[float, float]],
) -> tuple[np.ndarray, dict[str, tuple[float, float]]]:
    """Roll the controller forward open-loop and find anything out of range.

    Open-loop, so this is the trajectory the controller *asks* for; on the robot
    the droop correction will add a little to it during the rise.  That is a
    centimetre-scale offset, not the thing this check is looking for.
    """
    controller = StandUpController(cfg)
    controller.reset(start)
    q = start.copy()
    control_dt = POCHI_TASK_SPEC.control.sim_dt * POCHI_TASK_SPEC.control.decimation
    ticks = int(round(cfg.total_duration_s / control_dt))

    worst: dict[str, tuple[float, float]] = {}
    peak_step = np.zeros(len(motor_ids))
    previous = start.copy()
    for _ in range(ticks):
        q = controller.act(q, control_dt)
        peak_step = np.maximum(peak_step, np.abs(q - previous))
        previous = q.copy()
        for i, name in enumerate(JOINT_NAMES):
            low, high = limits[i]
            over = max(low - q[i], q[i] - high)
            if over > 0 and over > worst.get(name, (0.0, 0.0))[0]:
                worst[name] = (over, q[i])
    return peak_step / control_dt, worst


async def send_targets(link: Link, motor_ids: list[int], targets: np.ndarray) -> None:
    for motor_id, position in zip(motor_ids, targets, strict=True):
        await link.send(
            {"type": "target", "motorId": int(motor_id), "positionRad": float(position)}
        )


def watchdog(
    snapshot: dict,
    motor_ids: list[int],
    targets: np.ndarray,
    max_error_rad: float,
) -> None:
    """Raise if the robot is not doing what it was told."""
    by_id = {motor["id"]: motor for motor in snapshot["motors"]}
    for i, motor_id in enumerate(motor_ids):
        motor = by_id[motor_id]
        if motor["faultCode"]:
            raise RuntimeError(
                f"CAN {motor_id} ({motor['name']}) faulted: {motor['state']}"
            )
        position = motor["positionRad"]
        if position is None:
            raise RuntimeError(f"CAN {motor_id} ({motor['name']}) lost CAN feedback")
        error = abs(float(position) - float(targets[i]))
        if error > max_error_rad:
            raise RuntimeError(
                f"CAN {motor_id} ({motor['name']}) is {math.degrees(error):.1f} deg "
                f"from its target -- stopping"
            )


async def arm(link: Link, kp: float, kd: float) -> None:
    """Set the gains, then enable torque holding the pose the robot is in.

    The server's own enable path seeds every target from the current encoder
    angle before enabling, so this is not a step input.

    Anything that goes wrong after the enable is sent drops torque again before
    raising.  The server only disarms by itself when its *last* client goes
    away, and the operator's browser is expected to be one of them, so a script
    that walks out on a half-armed robot leaves it energised.
    """
    await link.send({"type": "gains", "kp": kp, "kd": kd})
    await asyncio.sleep(0.1)
    await link.send({"type": "torque", "enabled": True})
    deadline = time.monotonic() + ARM_TIMEOUT_S
    active = 0
    try:
        while time.monotonic() < deadline:
            snapshot = link.fresh_snapshot()
            active = int(snapshot["activeTorqueCount"])
            if active == 12:
                print(f"torque on, 12/12 motors, kp={kp:.0f} kd={kd:.1f}")
                return
            await asyncio.sleep(0.05)
    except Exception:
        await disarm(link)
        raise
    await disarm(link)
    raise RuntimeError(f"only {active}/12 motors armed within {ARM_TIMEOUT_S:.0f}s")


async def disarm(link: Link, *, emergency: bool = False) -> None:
    try:
        if emergency:
            await link.send({"type": "emergencyStop"})
        await link.send({"type": "torque", "enabled": False})
        await asyncio.sleep(0.2)
    except Exception as exc:  # noqa: BLE001 - last-ditch, must not mask the cause
        print(f"failed to disarm cleanly: {exc}", file=sys.stderr)


async def run_standup(link: Link, args: argparse.Namespace) -> int:
    motor_ids, limits = joint_mapping()
    snapshot = await link.wait_for_snapshot()

    problems = check_ready(snapshot, motor_ids)
    if problems:
        print("not ready to move:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    start = measured_positions(snapshot, motor_ids)
    cfg = StandUpConfig(
        settle_duration_s=args.settle_duration,
        approach_duration_s=args.approach_duration,
        rise_duration_s=args.rise_duration,
    )
    peak_speed, worst = preflight_trajectory(start, cfg, motor_ids, limits)

    print("measured start pose (deg), by CAN id:")
    for i, name in enumerate(JOINT_NAMES):
        print(f"  CAN {motor_ids[i]:2d}  {name:14s} {math.degrees(start[i]):7.1f}")
    envelope = POCHI_STANDUP_SPEC.safety.motor_speed_limit
    print(
        f"\ntrajectory: {cfg.total_duration_s:.2f}s, "
        f"peak commanded speed {peak_speed.max():.2f} rad/s "
        f"(motor envelope {envelope:.1f} rad/s)"
    )
    if worst:
        print("targets that the firmware would clamp:")
        for name, (over, value) in worst.items():
            print(
                f"  {name:14s} reaches {math.degrees(value):7.1f} deg, "
                f"{math.degrees(over):.2f} deg outside"
            )
    else:
        print("every target stays inside the firmware limits")

    if args.mode == "dryrun":
        print("\ndryrun: nothing was sent.  Re-run with --mode standup --yes to move.")
        return 0
    if worst and not args.allow_clamp:
        print(
            "\nrefusing to move: see the clamped targets above "
            "(--allow-clamp to override)"
        )
        return 1

    max_error_rad = math.radians(args.max_error_deg)
    control_dt = POCHI_TASK_SPEC.control.sim_dt * POCHI_TASK_SPEC.control.decimation
    controller = StandUpController(cfg)
    controller.reset(start)

    try:
        await arm(link, args.kp, args.kd)
    except Exception as exc:  # noqa: BLE001 - already disarmed by arm()
        print(f"could not arm: {exc}", file=sys.stderr)
        return 1
    print(f"standing up over {cfg.total_duration_s:.2f}s -- Ctrl-C drops torque")

    targets = start.copy()
    next_tick = time.monotonic()
    last_report = 0.0
    elapsed = 0.0
    try:
        while elapsed < cfg.total_duration_s + args.hold_s:
            snapshot = link.fresh_snapshot()
            watchdog(snapshot, motor_ids, targets, max_error_rad)
            measured = measured_positions(snapshot, motor_ids)
            targets = controller.act(measured, control_dt)
            await send_targets(link, motor_ids, targets)

            now = time.monotonic()
            if now - last_report >= 0.5:
                last_report = now
                error = np.abs(measured - targets)
                print(
                    f"  t={elapsed:5.2f}s  phase={controller.phase:8s}  "
                    f"target_height={controller.target_base_height:.3f}m  "
                    f"max_err={math.degrees(error.max()):4.1f}deg"
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

    print("standing.  Holding until you disarm; Ctrl-C drops torque.")
    try:
        while True:
            snapshot = link.fresh_snapshot()
            watchdog(snapshot, motor_ids, targets, max_error_rad)
            await send_targets(link, motor_ids, targets)
            await asyncio.sleep(control_dt)
    except KeyboardInterrupt:
        print("\ndropping torque")
        await disarm(link)
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"\nABORT while holding: {exc}", file=sys.stderr)
        await disarm(link, emergency=True)
        return 1


async def run_hold(link: Link, args: argparse.Namespace) -> int:
    motor_ids, _ = joint_mapping()
    snapshot = await link.wait_for_snapshot()
    problems = check_ready(snapshot, motor_ids)
    if problems:
        print("not ready to move:")
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
            measured = measured_positions(snapshot, motor_ids)
            drift = math.degrees(np.abs(measured - targets).max())
            print(f"  max drift {drift:5.2f} deg", end="\r")
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


async def run_jog(link: Link, args: argparse.Namespace) -> int:
    if args.joint not in JOINT_NAMES:
        print(f"--joint must be one of: {', '.join(JOINT_NAMES)}")
        return 2
    motor_ids, limits = joint_mapping()
    index = JOINT_NAMES.index(args.joint)
    motor_id = motor_ids[index]

    snapshot = await link.wait_for_snapshot()
    problems = check_ready(snapshot, motor_ids)
    if problems:
        print("not ready to move:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    start = measured_positions(snapshot, motor_ids)
    origin = float(start[index])
    delta = math.radians(args.delta_deg)
    low, high = limits[index]
    if not (low <= origin + delta <= high):
        print(
            f"{args.joint} at {math.degrees(origin):.1f} deg cannot move "
            f"{args.delta_deg:+.1f} deg without leaving [{math.degrees(low):.0f}, "
            f"{math.degrees(high):.0f}] deg"
        )
        return 1

    print(
        f"jogging {args.joint} (CAN {motor_id}) {args.delta_deg:+.1f} deg "
        f"from {math.degrees(origin):.1f} deg and back, "
        f"over {2 * args.jog_duration:.1f}s"
    )
    try:
        await arm(link, args.kp, args.kd)
    except Exception as exc:  # noqa: BLE001 - already disarmed by arm()
        print(f"could not arm: {exc}", file=sys.stderr)
        return 1
    max_error_rad = math.radians(args.max_error_deg)
    control_dt = POCHI_TASK_SPEC.control.sim_dt * POCHI_TASK_SPEC.control.decimation
    ticks = int(round(args.jog_duration / control_dt))
    try:
        for phase in (1.0, -1.0):
            for k in range(ticks):
                fraction = (k + 1) / ticks
                moved = delta * (fraction if phase > 0 else 1.0 - fraction)
                target = origin + moved
                snapshot = link.fresh_snapshot()
                by_id = {motor["id"]: motor for motor in snapshot["motors"]}
                position = by_id[motor_id]["positionRad"]
                if position is None or by_id[motor_id]["faultCode"]:
                    raise RuntimeError(f"CAN {motor_id} lost feedback or faulted")
                error = abs(float(position) - target)
                if error > max_error_rad:
                    raise RuntimeError(
                        f"CAN {motor_id} is {math.degrees(error):.1f} deg "
                        "from its target -- stopping"
                    )
                await link.send(
                    {"type": "target", "motorId": int(motor_id), "positionRad": target}
                )
                await asyncio.sleep(control_dt)
            landed = origin + (delta if phase > 0 else 0.0)
            print(f"  reached {math.degrees(landed):.1f} deg")
    except KeyboardInterrupt:
        print("\ninterrupted -- dropping torque")
        await disarm(link)
        return 130
    except Exception as exc:  # noqa: BLE001
        print(f"\nABORT: {exc}", file=sys.stderr)
        await disarm(link, emergency=True)
        return 1
    print("dropping torque")
    await disarm(link)
    return 0


@contextlib.contextmanager
def _raw_terminal():
    """Single keypresses, no Enter, no local echo -- restores the tty on exit."""
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    tty.setcbreak(fd)
    try:
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


async def run_teleop(link: Link, args: argparse.Namespace) -> int:
    """Get to the closed stance once, then ramp the base height up or down on
    each keypress -- crouching is just the stand-up rise run backwards, so one
    controller (:class:`TeleopHeightController`) drives both directions.
    """
    motor_ids, limits = joint_mapping()
    snapshot = await link.wait_for_snapshot()

    problems = check_ready(snapshot, motor_ids)
    if problems:
        print("not ready to move:")
        for problem in problems:
            print(f"  - {problem}")
        return 1

    start = measured_positions(snapshot, motor_ids)
    cfg = StandUpConfig(
        settle_duration_s=args.settle_duration,
        approach_duration_s=args.approach_duration,
    )
    control_dt = POCHI_TASK_SPEC.control.sim_dt * POCHI_TASK_SPEC.control.decimation
    max_error_rad = math.radians(args.max_error_deg)
    controller = StandUpController(cfg)
    controller.reset(start)

    try:
        await arm(link, args.kp, args.kd)
    except Exception as exc:  # noqa: BLE001 - already disarmed by arm()
        print(f"could not arm: {exc}", file=sys.stderr)
        return 1

    print(f"closing to the crouch stance over {cfg.approach_end_s:.2f}s")
    targets = start.copy()
    try:
        while controller.phase != "rise":
            snapshot = link.fresh_snapshot()
            watchdog(snapshot, motor_ids, targets, max_error_rad)
            measured = measured_positions(snapshot, motor_ids)
            targets = controller.act(measured, control_dt)
            await send_targets(link, motor_ids, targets)
            await asyncio.sleep(control_dt)
    except KeyboardInterrupt:
        print("\ninterrupted -- dropping torque")
        await disarm(link)
        return 130
    except Exception as exc:  # noqa: BLE001 - any surprise means stop the robot
        print(f"\nABORT: {exc}", file=sys.stderr)
        await disarm(link, emergency=True)
        return 1

    teleop = PoseSequencer(
        PoseSequencerConfig(
            down_joint_pos=COLLAPSED_JOINT_POS,
            crouch_height=cfg.crouch_height,
            stand_height=cfg.stand_height,
            hip_roll=cfg.hip_roll,
            crouch_stand_duration_s=args.stand_crouch_duration,
            down_crouch_duration_s=args.crouch_down_duration,
            droop_gain=cfg.droop_gain,
            droop_limit=cfg.droop_limit,
        ),
        start="crouch",
    )

    print(
        "in the crouch stance.  [u] stand up  [c] crouch  "
        "[d] lie down (belly on the floor)  [a] auto squat (repeat stand<->crouch)  "
        "[q] quit (Ctrl-C also drops torque)"
    )
    loop = asyncio.get_event_loop()
    keys: asyncio.Queue[str] = asyncio.Queue()
    loop.add_reader(sys.stdin.fileno(), lambda: keys.put_nowait(sys.stdin.read(1)))
    auto_squat = False
    auto_squat_hold_s = 0.3
    hold_elapsed = 0.0
    try:
        with _raw_terminal():
            while True:
                while not keys.empty():
                    key = keys.get_nowait()
                    target = {"u": "stand", "c": "crouch", "d": "down"}.get(key.lower())
                    if target is not None:
                        auto_squat = False
                        refusal = teleop.command(target)
                        print(f"\n{refusal}" if refusal else f"\n-> {target}")
                    elif key.lower() == "a":
                        if teleop.state == "down":
                            print("\nreach 'crouch' first before auto squat")
                        else:
                            auto_squat = not auto_squat
                            hold_elapsed = 0.0
                            print(f"\n-> auto squat: {'on' if auto_squat else 'off'}")
                    elif key in ("q", "Q", "\x03"):
                        raise KeyboardInterrupt

                if auto_squat and teleop.settled:
                    hold_elapsed += control_dt
                    if hold_elapsed >= auto_squat_hold_s:
                        teleop.command("crouch" if teleop.state == "stand" else "stand")
                        hold_elapsed = 0.0

                snapshot = link.fresh_snapshot()
                watchdog(snapshot, motor_ids, targets, max_error_rad)
                measured = measured_positions(snapshot, motor_ids)
                targets = teleop.act(measured, control_dt)
                await send_targets(link, motor_ids, targets)
                await asyncio.sleep(control_dt)
    except KeyboardInterrupt:
        print("\ndropping torque")
        await disarm(link)
        return 0
    except Exception as exc:  # noqa: BLE001 - any surprise means stop the robot
        print(f"\nABORT: {exc}", file=sys.stderr)
        await disarm(link, emergency=True)
        return 1
    finally:
        loop.remove_reader(sys.stdin.fileno())


def parse_args() -> argparse.Namespace:
    cfg = StandUpConfig()
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--mode",
        choices=("dryrun", "hold", "jog", "standup", "teleop"),
        default="dryrun",
        help="dryrun (default) sends nothing",
    )
    p.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    p.add_argument(
        "--yes", action="store_true", help="required for anything that moves"
    )
    p.add_argument("--kp", type=float, default=40.0)
    p.add_argument("--kd", type=float, default=1.0)
    p.add_argument("--max-error-deg", type=float, default=DEFAULT_MAX_ERROR_DEG)
    p.add_argument("--allow-clamp", action="store_true")
    p.add_argument("--duration", type=float, default=10.0, help="hold mode only")
    p.add_argument("--joint", default="FL_knee", help="jog mode only")
    p.add_argument("--delta-deg", type=float, default=5.0, help="jog mode only")
    p.add_argument("--jog-duration", type=float, default=2.0, help="jog mode only")
    p.add_argument("--hold-s", type=float, default=2.0, help="standup mode only")
    p.add_argument("--settle-duration", type=float, default=cfg.settle_duration_s)
    p.add_argument("--approach-duration", type=float, default=cfg.approach_duration_s)
    p.add_argument("--rise-duration", type=float, default=cfg.rise_duration_s)
    p.add_argument(
        "--stand-crouch-duration",
        type=float,
        default=cfg.rise_duration_s,
        help="teleop mode only: seconds for a full crouch<->stand ramp",
    )
    p.add_argument(
        "--crouch-down-duration",
        type=float,
        default=cfg.approach_duration_s,
        help="teleop mode only: seconds for a full crouch<->down (belly) ramp",
    )
    return p.parse_args()


async def main_async(args: argparse.Namespace) -> int:
    async with Link(args.url) as link:
        if args.mode == "hold":
            return await run_hold(link, args)
        if args.mode == "jog":
            return await run_jog(link, args)
        if args.mode == "teleop":
            return await run_teleop(link, args)
        return await run_standup(link, args)


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
