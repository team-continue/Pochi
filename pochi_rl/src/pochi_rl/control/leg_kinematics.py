"""Planar two-link kinematics for one Pochi leg.

All four legs are geometrically identical (see ``assets/pochi/pochi.xml``): the
hip roll axis is +x, the hip pitch and knee axes are +y, the thigh runs
``THIGH_LENGTH`` down -z from the hip and the shank ``SHANK_LENGTH`` down -z
from the knee.  With hip roll at zero the leg therefore stays in the x-z plane
and reduces to a textbook two-link arm, which is all the stand-up controller
needs.

Positions here are the *foot* relative to the *hip*, in the base frame, and
ignore the constant outboard y offset of the foot (``FOOT_OFFSET_Y``): it is the
same for every pose and so cancels out of the controller.
"""

from __future__ import annotations

import math

from pochi_rl.robot import SHANK_LENGTH, THIGH_LENGTH

L1 = THIGH_LENGTH
L2 = SHANK_LENGTH

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

  ``knee_sign`` picks the elbow branch: -1 folds the knee backwards (the sign
  the front legs use in ``DEFAULT_JOINT_POS``), +1 forwards (the rear legs).
  Targets outside the reachable annulus are pulled back onto it along the same
  direction rather than raising, so the caller can ask for anything.
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
