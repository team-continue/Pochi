"""Scripted stand-up policy for Pochi.

Standing up is a one-shot, quasi-static manoeuvre with an obvious solution, so
this is written out rather than trained: a Cartesian trajectory for the feet,
solved back into joint targets with the two-link IK in
:mod:`pochi_rl.control.leg_kinematics`.  It has the same interface as a learned
policy -- joint positions in, joint position targets out, at the task spec's
50 Hz -- so it drops into the same PD actuators that
``Pochi-Velocity-Flat-v0`` drives.

The robot starts flat on the floor.  Cutting the motors from the folded crouch
drops it there on its own: the hip rolls splay out to their stops while the
knees stay curled, so it settles into a frog sprawl with the belly down and the
front thighs resting on the ground.  Standing back up is then just undoing
that -- the feet are already beside the body rather than thrown out fore and
aft, so they never have to be dragged in underneath, and nothing but the feet
ever pushes.

Four phases:

``settle``    Hold the pose the robot woke up in, so any arrival transient has
              died out before anything moves.
``approach``  Joint-space interpolation onto the crouch stance, which mostly
              means closing the splayed hip rolls.  This is what lifts the
              belly clear; by the end of it the thighs carry no load and the
              feet carry all of it.
``rise``      The feet stay under the hips and the base height is ramped from
              the crouch to the nominal stance along a smoothstep, so the
              manoeuvre starts and ends at zero velocity.
``hold``      Stay standing.

Only joint encoders are used for feedback -- no base state -- so the same
controller runs on hardware.  With all four feet planted, base height *is*
observable from the encoders, which is what ``droop_gain`` below exploits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from pochi_rl.control import leg_kinematics as lk
from pochi_rl.robot import DEFAULT_JOINT_POS, JOINT_NAMES, LEGS, NOMINAL_BASE_HEIGHT

# Order the controller works in: four legs x (hip_roll, hip_pitch, knee), which
# is JOINT_NAMES.  Cached as index triples so ``act`` stays allocation-light.
_KINDS = ("hip_roll", "hip_pitch", "knee")
_LEG_SLICES = {
  leg: tuple(JOINT_NAMES.index(f"{leg}_{kind}") for kind in _KINDS) for leg in LEGS
}

# Which way each knee folds, taken from the stance in pochi_constants so the two
# stay in sync.  Physically the front legs fold backwards and the rear legs
# forwards; that is no longer readable off the signs, because the stance is
# written in motor coordinates and `MOTOR_SIGN` has already absorbed the
# mounting direction of each module.
#
# Everything below therefore works in motor coordinates while `leg_kinematics`
# solves in the canonical leg frame, and no conversion is needed between them.
# That holds on two properties of `MOTOR_SIGN`, not by accident:
#
#   * hip pitch and knee always share a sign within a leg, and negating both
#     leaves `lk.extension` unchanged, so measured extension is frame-agnostic;
#   * every `lk.inverse` call here asks for a foot straight under the hip
#     (x=0), where the two elbow branches are exact mirror images, so picking
#     the branch by the sign already stored in DEFAULT_JOINT_POS returns angles
#     in that same convention.
#
# If a future MOTOR_SIGN ever splits pitch and knee within one leg, both
# properties break and this needs an explicit canonical<->motor conversion.
_KNEE_SIGN = {leg: math.copysign(1.0, DEFAULT_JOINT_POS[f"{leg}_knee"]) for leg in LEGS}

# Leg extension at the nominal stance, and hence the constant offset between
# "hip-to-foot distance" and "base height": the foot pad sticks out below the
# foot body origin that the kinematics stop at.
_NOMINAL_EXTENSION = lk.extension(
  DEFAULT_JOINT_POS["FL_hip_pitch"], DEFAULT_JOINT_POS["FL_knee"]
)
FOOT_PAD_OFFSET = NOMINAL_BASE_HEIGHT - _NOMINAL_EXTENSION

# Deepest crouch the joint limits allow, with a small margin so the knees are
# not sitting on their stops.  About 0.17 m; Pochi cannot fold flatter, so this
# is what "lying down" looks like for it.  At this height the only thing
# touching the floor is the four feet -- the belly clears it by 0.11 m and the
# knees by 0.12 m.
MIN_CROUCH_HEIGHT = lk.MIN_REACH + FOOT_PAD_OFFSET + 0.01


def _folded_stance() -> dict[str, float]:
  """Joint angles for the deepest crouch, feet straight under the hips."""
  reach = MIN_CROUCH_HEIGHT - FOOT_PAD_OFFSET
  pose: dict[str, float] = {}
  for leg in LEGS:
    hip_pitch, knee = lk.inverse(0.0, -reach, _KNEE_SIGN[leg])
    pose[f"{leg}_hip_roll"] = 0.0
    pose[f"{leg}_hip_pitch"] = hip_pitch
    pose[f"{leg}_knee"] = knee
  return pose


#: The crouch the robot is posed in before the motors are cut; it collapses
#: from here onto its belly, and that collapsed pose is what the manoeuvre
#: actually starts from.
FOLDED_JOINT_POS = _folded_stance()

# Where that collapse lands: hip rolls splayed out to their stops, knees still
# curled, belly on the floor.  Measured by simulating it (see
# ``pochi_rl.control.mujoco_driver.StandUpSim.reset``); ``test_standup`` re-runs
# the collapse and checks these still match, so they cannot drift away from the
# model.  They are written down rather than simulated at import time because the
# RL environment has to pose thousands of robots per reset and cannot afford to
# settle each one.  Angles are clamped to the joint limits: the soft limits let
# the settled pose overshoot them by a fraction of a milliradian.
COLLAPSED_BASE_HEIGHT = 0.0633
COLLAPSED_JOINT_POS = {
  "FL_hip_roll": -0.700,
  "FL_hip_pitch": -1.081,
  "FL_knee": 2.400,
  "FR_hip_roll": 0.700,
  "FR_hip_pitch": 1.080,
  "FR_knee": -2.400,
  "RL_hip_roll": 0.700,
  "RL_hip_pitch": 1.600,
  "RL_knee": -2.400,
  "RR_hip_roll": -0.700,
  "RR_hip_pitch": -1.600,
  "RR_knee": 2.400,
}


def _smoothstep(u: float) -> float:
  """3u^2 - 2u^3 on [0, 1]: zero slope at both ends, so no velocity step."""
  u = min(max(u, 0.0), 1.0)
  return u * u * (3.0 - 2.0 * u)


def reference_height(t: float, settle_s: float, rise_s: float) -> float:
  """Base height the manoeuvre should be at ``t`` seconds in.

  Hold at the collapsed height for ``settle_s``, then smoothstep up to the
  nominal stance over ``rise_s``.  Both the scripted controller and the learned
  task's reward read this, so the shape they are aiming at is defined once.
  """
  return COLLAPSED_BASE_HEIGHT + (NOMINAL_BASE_HEIGHT - COLLAPSED_BASE_HEIGHT) * (
    _smoothstep((t - settle_s) / rise_s)
  )


@dataclass(frozen=True)
class StandUpConfig:
  """Timing and shape of the manoeuvre.  Defaults are deliberately slow."""

  settle_duration_s: float = 0.75
  approach_duration_s: float = 2.5
  rise_duration_s: float = 5.0

  crouch_height: float = MIN_CROUCH_HEIGHT
  stand_height: float = NOMINAL_BASE_HEIGHT
  hip_roll: float = 0.0

  # Integral correction on leg extension, active through the rise and hold.
  # The PD actuators need a steady-state error to hold the robot's weight, so
  # the commanded stance always lands a couple of centimetres above the one the
  # robot settles into; this walks the command out until the *measured*
  # extension matches what was asked for.  [1/s], and the clamp is in metres.
  droop_gain: float = 1.2
  droop_limit: float = 0.06

  @property
  def approach_end_s(self) -> float:
    return self.settle_duration_s + self.approach_duration_s

  @property
  def total_duration_s(self) -> float:
    return self.approach_end_s + self.rise_duration_s


class StandUpController:
  """Stateful stand-up policy.  Call :meth:`reset`, then :meth:`act` at 50 Hz."""

  def __init__(self, cfg: StandUpConfig | None = None) -> None:
    self.cfg = cfg or StandUpConfig()
    self._crouch = self._stance_targets(self.cfg.crouch_height)
    self._start = np.array([FOLDED_JOINT_POS[n] for n in JOINT_NAMES])
    self._t = 0.0
    self._droop = {leg: 0.0 for leg in LEGS}

  # -- construction helpers ----------------------------------------------------

  def _stance_targets(self, base_height: float) -> np.ndarray:
    """Joint targets that stand the base at ``base_height``, feet under hips."""
    targets = np.zeros(len(JOINT_NAMES))
    reach = base_height - FOOT_PAD_OFFSET
    for leg, (i_roll, i_pitch, i_knee) in _LEG_SLICES.items():
      hip_pitch, knee = lk.inverse(0.0, -reach, _KNEE_SIGN[leg])
      targets[i_roll] = self.cfg.hip_roll
      targets[i_pitch] = hip_pitch
      targets[i_knee] = knee
    return targets

  # -- policy interface --------------------------------------------------------

  def reset(self, joint_pos: np.ndarray | None = None) -> None:
    """Restart the manoeuvre from the measured pose ``joint_pos``."""
    self._t = 0.0
    self._droop = {leg: 0.0 for leg in LEGS}
    if joint_pos is not None:
      self._start = np.asarray(joint_pos, dtype=float).copy()

  @property
  def phase(self) -> str:
    cfg = self.cfg
    if self._t < cfg.settle_duration_s:
      return "settle"
    if self._t < cfg.approach_end_s:
      return "approach"
    if self._t < cfg.total_duration_s:
      return "rise"
    return "hold"

  @property
  def target_base_height(self) -> float:
    """Base height the trajectory is asking for right now."""
    cfg = self.cfg
    u = (self._t - cfg.approach_end_s) / cfg.rise_duration_s
    return cfg.crouch_height + (cfg.stand_height - cfg.crouch_height) * _smoothstep(u)

  def act(self, joint_pos: np.ndarray, dt: float) -> np.ndarray:
    """Advance the clock by ``dt``; return 12 targets in JOINT_NAMES order."""
    cfg = self.cfg
    q = np.asarray(joint_pos, dtype=float)

    if self._t < cfg.settle_duration_s:
      self._t += dt
      return self._start.copy()

    if self._t < cfg.approach_end_s:
      s = _smoothstep((self._t - cfg.settle_duration_s) / cfg.approach_duration_s)
      self._t += dt
      return (1.0 - s) * self._start + s * self._crouch

    reach = self.target_base_height - FOOT_PAD_OFFSET
    targets = np.zeros(len(JOINT_NAMES))
    for leg, (i_roll, i_pitch, i_knee) in _LEG_SLICES.items():
      measured = lk.extension(q[i_pitch], q[i_knee])
      self._droop[leg] = float(
        np.clip(
          self._droop[leg] + cfg.droop_gain * (reach - measured) * dt,
          -cfg.droop_limit,
          cfg.droop_limit,
        )
      )
      hip_pitch, knee = lk.inverse(0.0, -(reach + self._droop[leg]), _KNEE_SIGN[leg])
      targets[i_roll] = cfg.hip_roll
      targets[i_pitch] = hip_pitch
      targets[i_knee] = knee

    self._t += dt
    return targets


@dataclass(frozen=True)
class TeleopConfig:
  """Shape of an on-demand base-height ramp, triggered by e.g. a keypress."""

  crouch_height: float = MIN_CROUCH_HEIGHT
  stand_height: float = NOMINAL_BASE_HEIGHT
  hip_roll: float = 0.0
  transition_duration_s: float = 4.0

  # Same droop correction as StandUpConfig; see its docstring.
  droop_gain: float = 1.2
  droop_limit: float = 0.06


class TeleopHeightController:
  """Ramp the base height up or down on command, feet fixed under the hips.

  Standing up and crouching back down are the same move on this robot -- once
  the hip rolls are closed and the feet are under the hips, the whole stance is
  one number, the leg extension, and getting to any height in range is just
  solving :mod:`pochi_rl.control.leg_kinematics` for that number.  So unlike
  :class:`StandUpController`, which only ever runs that ramp upward once, this
  drives it in either direction, as many times as asked.

  Expects to be started from that closed stance (what ``StandUpController``'s
  ``settle``+``approach`` phases produce) -- it has no notion of the folded
  floor pose and will not get there from it.
  """

  def __init__(
    self, cfg: TeleopConfig | None = None, *, start_height: float | None = None
  ) -> None:
    self.cfg = cfg or TeleopConfig()
    height = self.cfg.crouch_height if start_height is None else start_height
    self._from_height = height
    self._to_height = height
    self._t = 0.0
    self._duration = 1e-6
    self._droop = {leg: 0.0 for leg in LEGS}

  def command(self, mode: str) -> None:
    """Start ramping toward ``"stand"`` or ``"crouch"`` from the current height.

    Safe to call again mid-ramp (e.g. the operator changes their mind): it
    restarts from wherever the base actually is, so there is no jump.
    """
    if mode not in ("stand", "crouch"):
      raise ValueError(f"mode must be 'stand' or 'crouch', got {mode!r}")
    target = self.cfg.stand_height if mode == "stand" else self.cfg.crouch_height
    current = self.height
    if target == self._to_height:
      return
    full_range = abs(self.cfg.stand_height - self.cfg.crouch_height)
    self._from_height = current
    self._to_height = target
    self._t = 0.0
    self._duration = max(
      abs(target - current) / full_range * self.cfg.transition_duration_s, 1e-6
    )

  @property
  def height(self) -> float:
    """Base height the ramp is asking for right now."""
    u = _smoothstep(self._t / self._duration)
    return self._from_height + (self._to_height - self._from_height) * u

  @property
  def settled(self) -> bool:
    """Whether the ramp has finished (does not mean the robot has caught up)."""
    return self._t >= self._duration

  def act(self, joint_pos: np.ndarray, dt: float) -> np.ndarray:
    """Advance the clock by ``dt``; return 12 targets in JOINT_NAMES order."""
    q = np.asarray(joint_pos, dtype=float)
    reach = self.height - FOOT_PAD_OFFSET
    targets = np.zeros(len(JOINT_NAMES))
    for leg, (i_roll, i_pitch, i_knee) in _LEG_SLICES.items():
      measured = lk.extension(q[i_pitch], q[i_knee])
      self._droop[leg] = float(
        np.clip(
          self._droop[leg] + self.cfg.droop_gain * (reach - measured) * dt,
          -self.cfg.droop_limit,
          self.cfg.droop_limit,
        )
      )
      hip_pitch, knee = lk.inverse(0.0, -(reach + self._droop[leg]), _KNEE_SIGN[leg])
      targets[i_roll] = self.cfg.hip_roll
      targets[i_pitch] = hip_pitch
      targets[i_knee] = knee

    self._t += dt
    return targets


def _stance_pose(base_height: float, hip_roll: float) -> np.ndarray:
  """Joint targets, feet under the hips, base at ``base_height``, no droop."""
  targets = np.zeros(len(JOINT_NAMES))
  reach = base_height - FOOT_PAD_OFFSET
  for leg, (i_roll, i_pitch, i_knee) in _LEG_SLICES.items():
    hip_pitch, knee = lk.inverse(0.0, -reach, _KNEE_SIGN[leg])
    targets[i_roll] = hip_roll
    targets[i_pitch] = hip_pitch
    targets[i_knee] = knee
  return targets


_POSE_STATES = ("down", "crouch", "stand")


@dataclass(frozen=True)
class PoseSequencerConfig:
  """Shape of the down/crouch/stand ladder and its two ramps."""

  #: Joint targets with the belly on the floor -- typically ``COLLAPSED_JOINT_POS``.
  down_joint_pos: dict[str, float]
  crouch_height: float = MIN_CROUCH_HEIGHT
  stand_height: float = NOMINAL_BASE_HEIGHT
  hip_roll: float = 0.0
  crouch_stand_duration_s: float = 4.0
  down_crouch_duration_s: float = 3.0
  droop_gain: float = 1.2
  droop_limit: float = 0.06


class PoseSequencer:
  """Step between ``"down"`` (belly on the floor), ``"crouch"``, and ``"stand"``.

  ``crouch``<->``stand`` is the Cartesian height ramp of
  :class:`TeleopHeightController`, feet fixed under the hips.  ``down``<->
  ``crouch`` is a joint-space blend instead: getting the belly off the floor
  means splaying the hip rolls open, which is not a height the leg IK alone
  can express.  It is :class:`StandUpController`'s ``approach`` phase, just
  runnable in either direction.

  Only one ramp is ever live.  A command more than one rung away from the
  current one is refused (returns a message) rather than skipping the middle
  rung -- the two mechanisms should never blend into each other -- and
  re-issuing a reachable command mid-ramp restarts it from wherever the robot
  actually is, so there is never a jump.
  """

  def __init__(self, cfg: PoseSequencerConfig, *, start: str = "crouch") -> None:
    if start not in _POSE_STATES:
      raise ValueError(f"start must be one of {_POSE_STATES}, got {start!r}")
    self.cfg = cfg
    self._crouch_pose = _stance_pose(cfg.crouch_height, cfg.hip_roll)
    self._down_pose = np.array([cfg.down_joint_pos[n] for n in JOINT_NAMES])
    self._rung = _POSE_STATES.index(start)
    self._height = self._fresh_height_controller(
      start_height=cfg.stand_height if start == "stand" else cfg.crouch_height
    )
    self._in_joint_ramp = False
    self._joint_from = self._crouch_pose.copy()
    self._joint_to = self._crouch_pose.copy()
    self._joint_t = 0.0
    self._joint_duration = 1e-6
    if start == "down":
      self._last_targets = self._down_pose.copy()
    elif start == "crouch":
      self._last_targets = self._crouch_pose.copy()
    else:
      self._last_targets = _stance_pose(cfg.stand_height, cfg.hip_roll)

  def _fresh_height_controller(self, *, start_height: float) -> TeleopHeightController:
    return TeleopHeightController(
      TeleopConfig(
        crouch_height=self.cfg.crouch_height,
        stand_height=self.cfg.stand_height,
        hip_roll=self.cfg.hip_roll,
        transition_duration_s=self.cfg.crouch_stand_duration_s,
        droop_gain=self.cfg.droop_gain,
        droop_limit=self.cfg.droop_limit,
      ),
      start_height=start_height,
    )

  @property
  def state(self) -> str:
    """The rung being headed toward (not necessarily reached yet)."""
    return _POSE_STATES[self._rung]

  @property
  def settled(self) -> bool:
    """Whether the current ramp has finished (open-loop; not a hardware check)."""
    if self._in_joint_ramp:
      return self._joint_t >= self._joint_duration
    return self._height.settled

  def command(self, state: str) -> str | None:
    """Head toward ``"down"``, ``"crouch"``, or ``"stand"``.

    Returns an explanatory message if ``state`` is more than one rung away
    (nothing is sent in that case -- the caller should tell the operator to
    get to the rung in between first), otherwise ``None``.
    """
    if state not in _POSE_STATES:
      raise ValueError(f"state must be one of {_POSE_STATES}, got {state!r}")
    target_rung = _POSE_STATES.index(state)
    if abs(target_rung - self._rung) > 1:
      via = _POSE_STATES[self._rung + (1 if target_rung > self._rung else -1)]
      return f"reach {via!r} first"
    if target_rung == self._rung:
      return None

    if {self._rung, target_rung} == {0, 1}:
      self._joint_from = self._last_targets.copy()
      self._joint_to = self._down_pose if target_rung == 0 else self._crouch_pose
      self._joint_t = 0.0
      self._joint_duration = self.cfg.down_crouch_duration_s
      self._in_joint_ramp = True
    else:
      if self._in_joint_ramp:
        # The joint ramp always lands exactly on `self._crouch_pose`, droop
        # zero, so a fresh height controller starting there matches with no
        # jump -- the old one's state is stale from however long ago it was
        # last driven.
        self._height = self._fresh_height_controller(start_height=self.cfg.crouch_height)
      self._in_joint_ramp = False
      self._height.command("stand" if target_rung == 2 else "crouch")
    self._rung = target_rung
    return None

  def act(self, joint_pos: np.ndarray, dt: float) -> np.ndarray:
    """Advance the clock by ``dt``; return 12 targets in JOINT_NAMES order."""
    if self._in_joint_ramp:
      u = _smoothstep(self._joint_t / self._joint_duration)
      targets = self._joint_from + (self._joint_to - self._joint_from) * u
      self._joint_t += dt
    else:
      targets = self._height.act(joint_pos, dt)
    self._last_targets = targets
    return targets
