"""Verify 3-D IK against the CAD model and a genuinely airborne side jump."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mujoco")

import mujoco

from pochi_rl.control.leg_kinematics import forward_3d, inverse_3d
from pochi_rl.control.side_jump import SideJumpConfig, SideJumpSim
from pochi_rl.robot import RS02_NO_LOAD_SPEED_RAD_S, RS02_PEAK_TORQUE_NM
from pochi_rl.robot.pochi_constants import FOOT_OFFSET_Y


def test_3d_kinematics_matches_mujoco_and_round_trips() -> None:
  sim = SideJumpSim()
  rng = np.random.default_rng(42)
  for _ in range(30):
    for leg in range(4):
      sign = -1 if leg < 2 else 1
      offset = FOOT_OFFSET_Y * (1 if leg % 2 == 0 else -1)
      q = [
        rng.uniform(-0.5, 0.5),
        -sign * rng.uniform(0.5, 1.1),
        sign * rng.uniform(1.1, 2.1),
      ]
      foot = forward_3d(*q, offset)
      np.testing.assert_allclose(inverse_3d(*foot, sign, offset), q, atol=1e-10)
      sim.data.qpos[sim.qpos_ids[3 * leg : 3 * leg + 3]] = q
      mujoco.mj_forward(sim.model, sim.data)
      actual = sim.feet[leg] - sim.data.xpos[sim.base_id] - sim.hips[leg]
      # Shared rounded geometry differs from the CAD by tens of micrometres.
      np.testing.assert_allclose(foot, actual, atol=8e-5)


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize("speed", [0.25, 0.45, 0.6])
def test_side_jump_leaves_ground_and_lands_on_four_feet(
  direction: int, speed: float
) -> None:
  sim = SideJumpSim(
    SideJumpConfig.for_size("small", direction=direction, lateral_speed=speed)
  )
  phases = set()
  clearance = 0.0
  for _ in range(round(10 / sim.model.opt.timestep)):
    previous_speed = np.abs(sim.data.qvel[sim.dof_ids]).copy()
    sim.step()
    phases.add(sim.phase)
    if sim.phase == "flight" and not sim.contacts.any():
      clearance = max(clearance, float(np.min(sim.feet[:, 2])))
    limit = RS02_PEAK_TORQUE_NM * np.clip(
      1 - previous_speed / RS02_NO_LOAD_SPEED_RAD_S, 0, 1
    )
    assert np.all(np.abs(sim.data.actuator_force[sim.act_ids]) <= limit + 1e-10)
    assert not sim.failed
  result = sim.metrics()
  assert phases == {"settle", "crouch", "thrust", "flight", "land", "recover", "hold"}
  assert result["success"]
  assert 0.06 < direction * result["lateral_distance_m"] < 0.20
  assert abs(result["forward_drift_m"]) < 0.01
  assert 0.15 < result["airtime_s"] < 0.35
  assert clearance > 0.04
  assert result["max_height_m"] > 0.38
  assert result["max_tilt_deg"] < 15
  assert result["nonfoot_ground_force_n"] == 0
  assert result["final_speed_m_s"] < 0.01
  assert sim.contacts.all()
  assert result["height_m"] == pytest.approx(0.3173, abs=0.01)
  assert not np.count_nonzero(sim.data.xfrc_applied)
  assert not np.count_nonzero(sim.data.qfrc_applied)
  sim.reset()
  assert sim.data.time == 0
  assert sim.phase == "settle"
  assert not sim.success and not sim.failed
  assert sim.airtime == 0 and sim.touchdown_time is None


@pytest.mark.parametrize(
  "kwargs",
  [
    {"direction": 0},
    {"lateral_speed": float("nan")},
    {"lateral_speed": 2.5},
    {"crouch_height": 0.1},
    {"thrust_duration": 0},
    {"vertical_acceleration": float("inf")},
  ],
)
def test_invalid_jump_settings(kwargs: dict) -> None:
  with pytest.raises(ValueError):
    SideJumpConfig(**kwargs)


@pytest.mark.parametrize("direction", [-1, 1])
@pytest.mark.parametrize(
  "size, distance, flight_distance",
  [
    ("small", 0.127, 0.06),
    ("medium", 0.325, 0.20),
    ("large", 0.523, 0.31),
  ],
)
def test_jump_sizes_land_without_increasing_motor_limits(
  direction: int,
  size: str,
  distance: float,
  flight_distance: float,
) -> None:
  sim = SideJumpSim(SideJumpConfig.for_size(size, direction=direction))
  for _ in range(round(10 / sim.model.opt.timestep)):
    previous_speed = np.abs(sim.data.qvel[sim.dof_ids]).copy()
    sim.step()
    limit = RS02_PEAK_TORQUE_NM * np.clip(
      1 - previous_speed / RS02_NO_LOAD_SPEED_RAD_S,
      0,
      1,
    )
    assert np.all(np.abs(sim.data.actuator_force[sim.act_ids]) <= limit + 1e-10)
    assert not sim.failed
  result = sim.metrics()
  assert result["success"] and sim.contacts.all()
  assert result["phase"] == "hold"
  assert result["final_tilt_deg"] < 1
  assert abs(result["final_yaw_deg"]) < 1
  assert result["standard_joint_error_deg"] < 1.5
  assert np.min(result["recovery_foot_clearance_m"]) > 0.02
  np.testing.assert_allclose(
    sim.feet - sim.data.xpos[sim.base_id],
    sim.nominal_feet,
    atol=0.008,
  )
  assert np.all(sim.recovery_lifts == 1)
  assert direction * result["lateral_distance_m"] == pytest.approx(distance, abs=0.02)
  assert direction * result["airborne_lateral_distance_m"] == pytest.approx(
    flight_distance, abs=0.02
  )
  assert result["nonfoot_ground_force_n"] == 0
  assert result["max_tilt_deg"] < 13
  assert abs(result["forward_drift_m"]) < 0.01
  assert result["height_m"] == pytest.approx(0.3173, abs=0.01)
  assert result["final_speed_m_s"] < 0.01
  if size != "small":
    assert result["airtime_s"] > 0.30
    assert result["max_height_m"] > 0.48
  assert not np.count_nonzero(sim.data.xfrc_applied)
  assert not np.count_nonzero(sim.data.qfrc_applied)
  sim.reset()
  assert sim.metrics()["airborne_lateral_distance_m"] is None
  assert sim.recovery_leg == -1 and sim.recovery_slot == -1
  assert not sim.recovery_clearance.any()


def test_default_is_large_and_preset_validation() -> None:
  assert SideJumpConfig() == SideJumpConfig.for_size("large")
  with pytest.raises(ValueError, match="size"):
    SideJumpConfig.for_size("unknown")
  with pytest.raises(ValueError, match="lateral_speed"):
    SideJumpConfig.for_size("large", lateral_speed=3)
