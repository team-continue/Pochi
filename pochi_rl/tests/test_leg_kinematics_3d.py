"""The 3-DOF leg forward kinematics used by the hardware state estimator."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pochi_rl.control import leg_kinematics as lk
from pochi_rl.robot import DEFAULT_JOINT_POS, JOINT_KINDS, LEGS, MOTOR_SIGN
from pochi_rl.robot.pochi_constants import BASE_BODY, FOOT_BODIES, FOOT_OFFSET_Y


def _canonical(leg: str) -> tuple[float, float, float]:
  """DEFAULT_JOINT_POS (motor frame) for one leg, converted to canonical."""
  return tuple(
    MOTOR_SIGN[f"{leg}_{kind}"] * DEFAULT_JOINT_POS[f"{leg}_{kind}"]
    for kind in JOINT_KINDS
  )


def test_forward3d_matches_planar_at_zero_hip_roll() -> None:
  for leg in LEGS:
    _, hip_pitch, knee = _canonical(leg)
    x0, z0 = lk.forward(hip_pitch, knee)
    x, y, z = lk.forward3d(leg, 0.0, hip_pitch, knee)
    hip_x, hip_y = lk._hip_offset(leg)
    assert (x, z) == pytest.approx((x0 + hip_x, z0))
    expected_y = (FOOT_OFFSET_Y if leg in lk._LEFT_LEGS else -FOOT_OFFSET_Y) + hip_y
    assert y == pytest.approx(expected_y)


def test_forward3d_jacobian_hip_roll_column_is_analytic() -> None:
  """At hip_roll = 0, d(y, z)/d(hip_roll) has a closed form: (-z0, y0)."""
  for leg in LEGS:
    _, hip_pitch, knee = _canonical(leg)
    x0, z0 = lk.forward(hip_pitch, knee)
    y0 = FOOT_OFFSET_Y if leg in lk._LEFT_LEGS else -FOOT_OFFSET_Y
    jacobian = lk.forward3d_jacobian(leg, 0.0, hip_pitch, knee)
    assert jacobian[:, 0] == pytest.approx((0.0, -z0, y0), abs=1e-6)


def test_default_stance_is_symmetric_3d() -> None:
  """Same invariant as test_motor_sign.test_default_stance_is_symmetric, but
  through `forward3d` rather than MuJoCo -- if the two ever disagree, one of
  them has a sign or offset wrong."""
  feet = {leg: np.array(lk.forward3d(leg, *_canonical(leg))) for leg in LEGS}

  heights = [foot[2] for foot in feet.values()]
  assert max(heights) - min(heights) < 1e-9, f"feet at different heights: {heights}"

  for left, right in (("FL", "FR"), ("RL", "RR")):
    assert feet[left][1] == pytest.approx(-feet[right][1], abs=1e-9)
    assert feet[left][0] == pytest.approx(feet[right][0], abs=1e-9)
  assert feet["FL"][0] == pytest.approx(-feet["RL"][0], abs=1e-9)


def test_forward3d_matches_mjcf() -> None:
  """Cross-check against the generated model, the same independent oracle
  `test_motor_sign.test_default_stance_is_symmetric` uses."""
  mujoco = pytest.importorskip("mujoco")

  mj = mujoco.MjModel.from_xml_path(str(Path("assets") / "pochi" / "pochi.xml"))
  data = mujoco.MjData(mj)
  data.qpos[0:3] = (0.0, 0.0, 0.0)
  data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
  for leg in LEGS:
    for kind in JOINT_KINDS:
      name = f"{leg}_{kind}"
      joint = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, name)
      data.qpos[mj.jnt_qposadr[joint]] = DEFAULT_JOINT_POS[name]
  mujoco.mj_forward(mj, data)

  base_index = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
  base_pos = data.xpos[base_index].copy()

  for leg, body in zip(LEGS, FOOT_BODIES, strict=True):
    assert body.startswith(leg)
    index = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, body)
    expected = data.xpos[index] - base_pos
    got = np.array(lk.forward3d(leg, *_canonical(leg)))
    assert got == pytest.approx(expected, abs=2e-3), leg
