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
# stay in sync: the front legs fold backwards, the rear legs forwards.
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
  "FL_hip_roll": 0.700,
  "FL_hip_pitch": 1.081,
  "FL_knee": -2.400,
  "FR_hip_roll": -0.700,
  "FR_hip_pitch": 1.080,
  "FR_knee": -2.400,
  "RL_hip_roll": 0.700,
  "RL_hip_pitch": -1.600,
  "RL_knee": 2.400,
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
