"""Physical regressions and optimizer constraints for the independent MPC path."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("osqp")
pytest.importorskip("mujoco")

from pochi_rl.control.centroidal_mpc import CentroidalMPC, MPCConfig
from pochi_rl.control.mpc_walk import MPCWalkSim, WalkConfig, swing_trajectory


def test_force_plan_supports_gravity_and_respects_swing_and_friction() -> None:
  cfg = MPCConfig(horizon=8)
  mpc = CentroidalMPC(10.0, np.diag([0.2, 0.5, 0.6]), cfg)
  state = np.zeros(12)
  state[2] = 0.3
  feet = np.tile(
    [[0.25, 0.17, 0], [0.25, -0.17, 0], [-0.25, 0.17, 0], [-0.25, -0.17, 0]],
    (cfg.horizon, 1, 1),
  )
  reference = np.tile(state, (cfg.horizon, 1))
  contacts = np.ones((cfg.horizon, 4), dtype=bool)
  force = mpc.solve(state, reference, feet, contacts, np.eye(3))
  assert mpc.status == "solved"
  assert force.sum(axis=0) == pytest.approx([0, 0, 98.1], abs=0.5)
  # A future lift changes the current plan before the foot leaves the floor.
  contacts[2:6, 0] = False
  anticipated = mpc.solve(state, reference, feet, contacts, np.eye(3))
  assert np.linalg.norm(anticipated - force) > 0.05
  assert np.max(np.abs(mpc.forces[~contacts])) < 0.01
  assert np.min(mpc.forces[:, :, 2]) > -0.01
  assert np.max(mpc.forces[:, :, 2]) < cfg.max_normal_force + 0.01
  assert np.all(
    np.abs(mpc.forces[:, :, :2])
    <= cfg.friction / np.sqrt(2) * mpc.forces[:, :, 2, None] + 0.01
  )


def test_swing_lifts_and_lands_without_endpoint_velocity() -> None:
  start, end = np.array([0.0, 0, 0.01]), np.array([0.12, 0, 0.01])
  for phase, expected in ((0, start), (1, end)):
    position, velocity = swing_trajectory(start, end, phase, 0.4, 0.045)
    np.testing.assert_allclose(position, expected)
    np.testing.assert_allclose(velocity, 0, atol=1e-12)
  position, _ = swing_trajectory(start, end, 0.5, 0.4, 0.045)
  assert position[2] == pytest.approx(0.055)


def test_crawl_remains_upright_and_actually_steps_for_thirty_seconds() -> None:
  sim = MPCWalkSim()
  result = sim.run(30)
  assert not result["failed"]
  assert result["solver_failures"] == 0
  assert result["distance_m"] == pytest.approx(1.68, abs=0.06)
  assert abs(result["lateral_drift_m"]) < 0.03
  assert result["min_height_m"] > 0.30
  assert result["max_tilt_deg"] < 5
  assert result["nonfoot_ground_force_n"] < 0.5
  assert result["max_torque_nm"] <= 17
  assert np.min(result["liftoffs"]) >= 10
  assert np.min(result["max_foot_clearance_m"]) > 0.025
  # No external forces or prescribed base pose keep the robot upright.
  assert np.count_nonzero(sim.data.xfrc_applied) == 0
  assert np.count_nonzero(sim.data.qfrc_applied) == 0
  sim.reset()
  assert sim.steps == 0 and sim.planner.failures == 0
  assert sim.data.time == 0 and not sim.failed
  assert not sim.liftoffs.any()
  np.testing.assert_allclose(sim.data.qpos[7:], sim.nominal_q)


@pytest.mark.parametrize("speed", [0.0, -0.06, 0.09])
def test_stop_reverse_and_faster_commands(speed: float) -> None:
  sim = MPCWalkSim(WalkConfig(speed=speed))
  result = sim.run(8)
  assert not result["failed"]
  assert result["solver_failures"] == 0
  assert result["nonfoot_ground_force_n"] < 0.5
  assert result["distance_m"] == pytest.approx(sim.command_at(8)[0], abs=0.06)
  if speed == 0:
    assert not sim.liftoffs.any()


def test_crawl_always_has_at_least_three_planned_supports() -> None:
  sim = MPCWalkSim()
  for time in np.linspace(0, 10, 2000):
    contacts, _ = sim.contacts_at(time)
    assert contacts.sum() >= 3


@pytest.mark.parametrize(
  "kwargs",
  [
    {"speed": float("nan")},
    {"period": 0},
    {"swing_duration": 0.7},
    {"step_height": -0.1},
    {"start_delay": -1},
  ],
)
def test_rejects_invalid_gait_settings(kwargs: dict) -> None:
  with pytest.raises(ValueError):
    WalkConfig(**kwargs)
