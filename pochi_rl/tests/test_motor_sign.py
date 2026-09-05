"""The model's joints turn the way the real motors do.

`MOTOR_SIGN` is the one place that records which of the twelve RS02 modules are
bolted in facing the other way.  Getting it wrong is not a crash -- the robot
simply folds the wrong way -- so it is pinned here, along with the two things
that have to keep agreeing with it: the generated MJCF, and the symmetry of the
stance it produces.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pochi_rl.robot import DEFAULT_JOINT_POS, FOOT_BODIES, JOINT_NAMES, MOTOR_SIGN

# Verified on the physical robot by driving one joint at a time.  The hardware
# side of the same fact is VIEWER_REVERSED_MOTOR_IDS in
# pochi_hardware/web/app/page.tsx; the modules sit in a rotationally symmetric
# pattern, so FL matches RR and FR matches RL rather than left mirroring right.
EXPECTED_SIGNS = {
  "FL_hip_roll": -1.0,
  "FL_hip_pitch": -1.0,
  "FL_knee": -1.0,
  "FR_hip_roll": -1.0,
  "FR_hip_pitch": 1.0,
  "FR_knee": 1.0,
  "RL_hip_roll": 1.0,
  "RL_hip_pitch": -1.0,
  "RL_knee": -1.0,
  "RR_hip_roll": 1.0,
  "RR_hip_pitch": 1.0,
  "RR_knee": 1.0,
}


def test_motor_sign_matches_the_robot() -> None:
  assert MOTOR_SIGN == EXPECTED_SIGNS


def test_motor_sign_never_splits_pitch_and_knee() -> None:
  """The precondition `standup` relies on to skip a frame conversion.

  Hip pitch and knee sharing a sign within a leg is what makes leg extension
  and the IK's elbow branch read the same in either frame; see the note on
  `_KNEE_SIGN` in `pochi_rl.control.standup`.
  """
  for leg in ("FL", "FR", "RL", "RR"):
    assert MOTOR_SIGN[f"{leg}_hip_pitch"] == MOTOR_SIGN[f"{leg}_knee"]


def test_mjcf_axes_follow_motor_sign() -> None:
  """The generated model is the table's only other copy; they must not drift."""
  mujoco = pytest.importorskip("mujoco")

  mj = mujoco.MjModel.from_xml_path(str(Path("assets") / "pochi" / "pochi.xml"))
  for name in JOINT_NAMES:
    joint = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, name)
    canonical = (1.0, 0.0, 0.0) if name.endswith("hip_roll") else (0.0, 1.0, 0.0)
    expected = [MOTOR_SIGN[name] * value for value in canonical]
    assert list(mj.jnt_axis[joint]) == pytest.approx(expected), name


def test_default_stance_is_symmetric() -> None:
  """A wrong sign shows up here as a leg standing somewhere the others do not.

  At the default stance the four feet carry a quarter of the weight each, so
  they have to sit at one height, mirrored left to right and front to rear.
  """
  mujoco = pytest.importorskip("mujoco")

  mj = mujoco.MjModel.from_xml_path(str(Path("assets") / "pochi" / "pochi.xml"))
  data = mujoco.MjData(mj)
  data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
  for name in JOINT_NAMES:
    joint = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, name)
    data.qpos[mj.jnt_qposadr[joint]] = DEFAULT_JOINT_POS[name]
  mujoco.mj_forward(mj, data)

  feet = {}
  for body in FOOT_BODIES:
    index = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, body)
    feet[body[:2]] = data.xpos[index].copy()

  heights = [position[2] for position in feet.values()]
  assert max(heights) - min(heights) < 1e-3, f"feet at different heights: {heights}"

  for left, right in (("FL", "FR"), ("RL", "RR")):
    assert feet[left][1] == pytest.approx(-feet[right][1], abs=1e-3)
    assert feet[left][0] == pytest.approx(feet[right][0], abs=1e-3)
  assert feet["FL"][0] == pytest.approx(-feet["RL"][0], abs=1e-3)
