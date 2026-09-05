"""Planar two-link kinematics for one Pochi leg.

All four legs are geometrically identical: the thigh runs ``THIGH_LENGTH`` down
-z from the hip and the shank ``SHANK_LENGTH`` down -z from the knee.  With hip
roll at zero the leg stays in the x-z plane and reduces to a textbook two-link
arm, which is all the stand-up controller needs.

This module works in the *canonical* leg frame -- hip roll about +x, hip pitch
and knee about +y for every leg -- which is the frame the CAD is measured in.
It is not the frame the motors report: six of the twelve are bolted in facing
the other way, so ``assets/pochi/pochi.xml`` gives those joints a negated axis
(see ``MOTOR_SIGN`` in ``pochi_rl.robot.pochi_constants``).  Callers holding
motor-frame angles are responsible for the sign; ``pochi_rl.control.standup``
documents why, for the stance it solves, the two frames happen to agree.

Positions here are the *foot* relative to the *hip*, in the base frame, and
ignore the constant outboard y offset of the foot (``FOOT_OFFSET_Y``): it is the
same for every pose and so cancels out of the controller.
"""

from __future__ import annotations

import math

import numpy as np

from pochi_rl.robot import (
  FOOT_OFFSET_Y,
  HIP_OFFSET_X,
  HIP_OFFSET_Y,
  SHANK_LENGTH,
  THIGH_LENGTH,
)

L1 = THIGH_LENGTH
L2 = SHANK_LENGTH

_FRONT_LEGS = ("FL", "FR")
_LEFT_LEGS = ("FL", "RL")

# Reach limits of the two-link chain.  The lower bound comes from the knee
# limit in the MJCF (+/-2.4 rad), not from the links folding onto each other.
KNEE_LIMIT = 2.4
MIN_REACH = math.sqrt(L1 * L1 + L2 * L2 + 2 * L1 * L2 * math.cos(KNEE_LIMIT))
MAX_REACH = L1 + L2


def forward(hip_pitch: float, knee: float) -> tuple[float, float]:
  """Foot position (x, z) relative to the hip, in the leg's sagittal plane."""
  x = -(L1 * math.sin(hip_pitch) + L2 * math.sin(hip_pitch + knee))
  z = -(L1 * math.cos(hip_pitch) + L2 * math.cos(hip_pitch + knee))
  return x, z


def extension(hip_pitch: float, knee: float) -> float:
  """Straight-line hip-to-foot distance, i.e. how far the leg is extended."""
  return math.hypot(*forward(hip_pitch, knee))


def inverse(x: float, z: float, knee_sign: float) -> tuple[float, float]:
  """Hip pitch and knee angles that put the foot at (x, z) relative to the hip.

  ``knee_sign`` picks the elbow branch in the canonical frame: -1 folds the knee
  backwards, +1 forwards.  Which of those a given leg wants is a property of how
  its motor is mounted, so callers take the sign from the stance rather than
  writing it down.  Targets outside the reachable annulus are pulled back onto
  it along the same direction rather than raising, so the caller can ask for
  anything.
  """
  r = math.hypot(x, z)
  if r < 1e-9:
    x, z, r = 0.0, -MIN_REACH, MIN_REACH
  clamped = min(max(r, MIN_REACH), MAX_REACH - 1e-4)
  if clamped != r:
    x *= clamped / r
    z *= clamped / r
    r = clamped

  cos_knee = (r * r - L1 * L1 - L2 * L2) / (2 * L1 * L2)
  knee = math.copysign(math.acos(min(max(cos_knee, -1.0), 1.0)), knee_sign)
  hip_pitch = math.atan2(-x, -z) - math.atan2(
    L2 * math.sin(knee), L1 + L2 * math.cos(knee)
  )
  return hip_pitch, knee


def _hip_offset(leg: str) -> tuple[float, float]:
  x = HIP_OFFSET_X if leg in _FRONT_LEGS else -HIP_OFFSET_X
  y = HIP_OFFSET_Y if leg in _LEFT_LEGS else -HIP_OFFSET_Y
  return x, y


def forward3d(
  leg: str, hip_roll: float, hip_pitch: float, knee: float
) -> tuple[float, float, float]:
  """Foot position relative to the base origin, in the base (IMU) frame.

  Unlike :func:`forward`, this is a real 4-leg forward kinematics: it adds the
  hip-roll rotation and the per-leg hip/foot offsets that the planar solve
  drops (they cancel out of the stand-up controller, which only ever needs one
  leg at a time and never varies hip roll). ``hip_roll``/``hip_pitch``/``knee``
  are canonical-frame angles -- a caller holding motor-frame angles has to
  multiply by ``pochi_rl.robot.MOTOR_SIGN`` first, same as everywhere else in
  this module.
  """
  x0, z0 = forward(hip_pitch, knee)
  y0 = FOOT_OFFSET_Y if leg in _LEFT_LEGS else -FOOT_OFFSET_Y
  cos_r, sin_r = math.cos(hip_roll), math.sin(hip_roll)
  y = y0 * cos_r - z0 * sin_r
  z = y0 * sin_r + z0 * cos_r
  hip_x, hip_y = _hip_offset(leg)
  return (x0 + hip_x, y + hip_y, z)


def forward3d_jacobian(
  leg: str, hip_roll: float, hip_pitch: float, knee: float, *, eps: float = 1e-6
) -> np.ndarray:
  """d(foot xyz) / d(hip_roll, hip_pitch, knee), central-difference.

  Used only to turn measured joint velocities into a foot velocity for the
  state estimator's no-slip measurement, so a numerical Jacobian over
  :func:`forward3d` is preferred here over hand-derived analytic partials --
  cheaper to keep correct, and accuracy at this scale does not matter for a
  filter measurement the way it would for a control law.
  """
  angles = np.array([hip_roll, hip_pitch, knee], dtype=float)
  jac = np.zeros((3, 3))
  for i in range(3):
    delta = np.zeros(3)
    delta[i] = eps
    plus = np.array(forward3d(leg, *(angles + delta)))
    minus = np.array(forward3d(leg, *(angles - delta)))
    jac[:, i] = (plus - minus) / (2.0 * eps)
  return jac
