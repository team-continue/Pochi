"""The scripted stand-up policy actually stands the robot up, on its feet."""

from __future__ import annotations

import math

import numpy as np
import pytest

from pochi_rl.control import leg_kinematics as lk
from pochi_rl.control.standup import (
  FOLDED_JOINT_POS,
  FOOT_PAD_OFFSET,
  MIN_CROUCH_HEIGHT,
  StandUpConfig,
  StandUpController,
)
from pochi_rl.robot import (
  DEFAULT_JOINT_POS,
  FOOT_GEOMS,
  JOINT_NAMES,
  LEGS,
  NOMINAL_BASE_HEIGHT,
)

# Joint limits from assets/pochi/pochi.xml.
LIMITS = {"hip_roll": 0.7, "hip_pitch": 1.6, "knee": 2.4}


def _sim(cfg: StandUpConfig | None = None):
  pytest.importorskip("mujoco")
  from pochi_rl.control.mujoco_driver import StandUpSim

  return StandUpSim(cfg)


def test_ik_inverts_fk() -> None:
  for hip_pitch, knee in ((0.8, -1.5), (0.2, -2.0), (-0.4, -1.1), (-0.8, 1.5)):
    x, z = lk.forward(hip_pitch, knee)
    got = lk.inverse(x, z, math.copysign(1.0, knee))
    assert got == pytest.approx((hip_pitch, knee), abs=1e-9)


def test_ik_clamps_unreachable_targets() -> None:
  # Straight down, far beyond the links, and far inside the knee limit.
  for reach in (10.0, 0.0):
    hip_pitch, knee = lk.inverse(0.0, -reach, -1.0)
    assert lk.MIN_REACH - 1e-6 <= lk.extension(hip_pitch, knee) <= lk.MAX_REACH


def test_default_stance_matches_nominal_height() -> None:
  """The IK's height model is the one pochi_constants pins.

  Standing the base at NOMINAL_BASE_HEIGHT with the foot straight under the hip
  has to reproduce DEFAULT_JOINT_POS.  The knee comes back exactly; hip pitch is
  a few milliradians off because the CAD stance puts the foot 1.5 mm ahead of
  the hip rather than dead under it.
  """
  hip_pitch, knee = lk.inverse(0.0, -(NOMINAL_BASE_HEIGHT - FOOT_PAD_OFFSET), -1.0)
  assert hip_pitch == pytest.approx(DEFAULT_JOINT_POS["FL_hip_pitch"], abs=0.01)
  assert knee == pytest.approx(DEFAULT_JOINT_POS["FL_knee"], abs=1e-6)


def test_folded_pose_is_folded_and_legal() -> None:
  """The pose the robot is posed in before going limp is the tightest crouch."""
  for name, value in FOLDED_JOINT_POS.items():
    limit = LIMITS[name.split("_", 1)[1]]
    assert abs(value) <= limit, f"{name} = {value} outside +/-{limit}"
  assert abs(FOLDED_JOINT_POS["FL_knee"]) > 2.3, "the knees should be curled up"
  reach = lk.extension(FOLDED_JOINT_POS["FL_hip_pitch"], FOLDED_JOINT_POS["FL_knee"])
  assert reach + FOOT_PAD_OFFSET == pytest.approx(MIN_CROUCH_HEIGHT, abs=1e-6)
  assert MIN_CROUCH_HEIGHT < 0.55 * NOMINAL_BASE_HEIGHT


def test_targets_stay_inside_joint_limits() -> None:
  cfg = StandUpConfig()
  controller = StandUpController(cfg)
  controller.reset(np.array([FOLDED_JOINT_POS[n] for n in JOINT_NAMES]))
  dt = 0.02
  for _ in range(int((cfg.total_duration_s + 2.0) / dt)):
    targets = controller.act(np.zeros(len(JOINT_NAMES)), dt)
    for name, value in zip(JOINT_NAMES, targets, strict=True):
      limit = LIMITS[name.split("_", 1)[1]]
      assert abs(value) <= limit, f"{name} target {value} outside +/-{limit}"


def test_trajectory_is_continuous() -> None:
  """No step change in the command, including across the phase boundaries."""
  cfg = StandUpConfig()
  controller = StandUpController(cfg)
  start = np.array([FOLDED_JOINT_POS[n] for n in JOINT_NAMES])
  controller.reset(start)
  dt = 0.02
  previous = start
  for _ in range(int((cfg.total_duration_s + 2.0) / dt)):
    # Feed the command back as the measurement: perfect tracking, so the droop
    # integrator contributes nothing and only the trajectory is under test.
    targets = controller.act(previous, dt)
    assert np.max(np.abs(targets - previous)) < 0.02
    previous = targets


# Base height at which the belly collision box touches the floor: the box is
# 0.06 m thick and sits 0.001 m up in the base frame.
BELLY_DOWN_HEIGHT = 0.059


def test_starts_lying_on_the_floor() -> None:
  """Cutting the motors really does put it down, not leave it crouching."""
  sim = _sim()
  assert sim.base_height < BELLY_DOWN_HEIGHT + 0.01
  assert sim.base_height < 0.4 * MIN_CROUCH_HEIGHT + 0.01
  # It got there by splaying the hip rolls out, knees still curled.
  for leg in LEGS:
    roll = sim.joint_pos[JOINT_NAMES.index(f"{leg}_hip_roll")]
    knee = sim.joint_pos[JOINT_NAMES.index(f"{leg}_knee")]
    assert abs(roll) > 0.6, f"{leg} hip roll should be splayed out, got {roll}"
    assert abs(knee) > 2.3, f"{leg} knee should still be curled, got {knee}"
  # And it is lying level, not on its side.
  assert sim.data.xmat[sim._base_id].reshape(3, 3)[2, 2] > 0.99


def test_stands_up() -> None:
  sim = _sim()
  sim.run(sim.controller.cfg.total_duration_s + 4.0)

  assert sim.phase == "hold"
  # Within a centimetre of the stance the rest of the project is tuned around.
  assert sim.base_height == pytest.approx(NOMINAL_BASE_HEIGHT, abs=0.01)
  # Upright: the base z axis still points at the sky.
  assert sim.data.xmat[sim._base_id].reshape(3, 3)[2, 2] > 0.999
  # And parked there, not still moving.
  assert np.max(np.abs(sim.data.qvel)) < 0.01


def test_pushes_off_its_feet_only() -> None:
  """Only the feet ever push.

  Lying down, the belly and the front thighs rest on the floor -- that is what
  lying down means.  What matters is that they are unloaded before the robot
  starts to rise, so the whole lift comes off the feet.  A start sprawled fore
  and aft instead would have to drag the legs in underneath and lever the robot
  up on its knees, which is what this pins against.
  """
  mujoco = pytest.importorskip("mujoco")
  sim = _sim()
  model, data = sim.model, sim.data
  feet = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in FOOT_GEOMS}
  ground = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ground")

  start_height = sim.base_height
  force = np.zeros(6)
  offenders: dict[str, float] = {}
  steps = int((sim.controller.cfg.total_duration_s + 2.0) / model.opt.timestep)
  for _ in range(steps):
    sim.step()
    if sim.base_height < start_height + 0.001:
      continue  # still down; contacts that merely bear the resting pose are fine
    for c in range(data.ncon):
      pair = {int(data.contact.geom1[c]), int(data.contact.geom2[c])}
      culprits = (pair - {ground} if ground in pair else pair) - feet
      if not culprits:
        continue
      mujoco.mj_contactForce(model, data, c, force)
      for g in culprits:
        name = str(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g))
        offenders[name] = max(offenders.get(name, 0.0), abs(float(force[0])))
  loaded = {k: round(v, 3) for k, v in offenders.items() if v > 0.5}
  assert not loaded, f"non-foot contacts carrying load while rising: {loaded}"


def test_feet_take_the_whole_weight_once_it_stands() -> None:
  """By the time it is up, the feet carry all of it."""
  mujoco = pytest.importorskip("mujoco")
  sim = _sim()
  model, data = sim.model, sim.data
  feet = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, g) for g in FOOT_GEOMS}
  weight = float(sum(model.body_mass)) * 9.81

  sim.run(sim.controller.cfg.total_duration_s + 2.0)
  force = np.zeros(6)
  on_feet = 0.0
  for c in range(data.ncon):
    if not ({int(data.contact.geom1[c]), int(data.contact.geom2[c])} & feet):
      continue
    mujoco.mj_contactForce(model, data, c, force)
    on_feet += abs(float(force[0]))
  assert on_feet == pytest.approx(weight, rel=0.05)


def test_rise_is_monotone_and_slow() -> None:
  sim = _sim()
  dt = sim.model.opt.timestep
  heights = []
  for _ in range(int(sim.controller.cfg.total_duration_s / dt)):
    sim.step()
    heights.append(sim.base_height)
  z = np.array(heights)
  # No lurch up and drop back: never falls by more than a solver-noise amount.
  assert np.min(np.diff(z)) > -5e-4
  assert np.max(np.abs(np.gradient(z, dt))) < 0.15, "should be a slow rise"


def test_settles_into_the_default_stance() -> None:
  sim = _sim()
  sim.run(sim.controller.cfg.total_duration_s + 4.0)
  for leg in LEGS:
    for kind in ("hip_pitch", "knee"):
      name = f"{leg}_{kind}"
      got = sim.joint_pos[JOINT_NAMES.index(name)]
      assert got == pytest.approx(DEFAULT_JOINT_POS[name], abs=0.06), name


def test_no_nans_anywhere() -> None:
  sim = _sim()
  sim.run(sim.controller.cfg.total_duration_s + 2.0)
  assert np.isfinite(sim.data.qpos).all()
  assert np.isfinite(sim.data.qvel).all()
