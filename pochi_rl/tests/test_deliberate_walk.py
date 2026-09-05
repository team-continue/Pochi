"""Footfall timing, stop gating, and task integration."""

from types import SimpleNamespace

import pytest
import torch

from pochi_rl.mjlab.deliberate_walk import (
  SPEC,
  contact_schedule,
  gait_reference,
  insufficient_support,
  phase_observation,
)


def test_one_foot_at_a_time_with_support_pause():
  t = torch.arange(0, SPEC.period_s, 0.001)
  swing, lift = gait_reference(t, torch.ones_like(t, dtype=torch.bool))
  assert (swing.sum(-1) <= 1).all()
  assert float((swing.sum(-1) == 0).float().mean()) == pytest.approx(0.2, abs=0.002)
  assert (lift >= 0).all()
  assert float(lift.max()) == pytest.approx(SPEC.clearance_m, abs=1e-6)
  assert (lift[~swing] == 0).all()
  midpoints = torch.arange(4) * SPEC.period_s / 4 + 0.2
  ordered, _ = gait_reference(midpoints, torch.ones(4, dtype=torch.bool))
  assert ordered.int().argmax(-1).tolist() == [2, 0, 3, 1]


def test_stopping_removes_clock_lift_and_requests_all_feet_down():
  t = torch.linspace(0, 10, 50)
  swing, lift = gait_reference(t, torch.zeros(50, dtype=torch.bool))
  assert not swing.any()
  assert not lift.any()
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_command=lambda _: torch.zeros(2, 3)),
    episode_length_buf=torch.tensor([17, 93]),
    step_dt=0.02,
    scene={"feet": SimpleNamespace(data=SimpleNamespace(found=torch.ones(2, 4)))},
  )
  assert not phase_observation(env).any()
  assert not contact_schedule(env, "feet").any()
  env.scene["feet"].data.found[0, :2] = 0
  assert contact_schedule(env, "feet").tolist() == [2.0, 0.0]
  assert insufficient_support(env, "feet").tolist() == [1.0, 0.0]


def test_phase_is_periodic_and_resets_per_environment():
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_command=lambda _: torch.ones(3, 3)),
    episode_length_buf=torch.tensor([0, 35, 140]),
    step_dt=0.02,
  )
  phase = phase_observation(env)
  torch.testing.assert_close(phase[0], phase[2], atol=1e-6, rtol=0)
  torch.testing.assert_close(phase[1], torch.tensor([1.0, 0.0]), atol=1e-6, rtol=0)


def test_deliberate_walk_rollout_is_finite():
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  cfg = load_env_cfg("Pochi-Deliberate-Walk-v0", play=True)
  cfg.scene.num_envs = 2
  cfg.events.pop("base_mass", None)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    obs, _ = env.reset()
    assert obs["actor"].shape == (2, 50)
    for _ in range(10):
      obs, reward, _, _, _ = env.step(torch.zeros(env.action_space.shape))
      assert torch.isfinite(obs["actor"]).all()
      assert torch.isfinite(reward).all()
    assert env.scene["feet_ground_contact"].data.found.shape == (2, 4)
  finally:
    env.close()
