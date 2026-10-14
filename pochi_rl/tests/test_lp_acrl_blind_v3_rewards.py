"""Duty balance is phase-free, timestep-consistent and isolated across resets."""

import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.managers.reward_manager import RewardManager, RewardTermCfg

from pochi_rl.mjlab.lp_acrl_blind_v3_rewards import moving_contact_duty_variance
from pochi_rl.robot import FOOT_GEOMS


def _env(count=3, dt=0.02, tau=1.0):
  command = torch.zeros(count, 3)
  command[:, 0] = 0.25
  sensor = SimpleNamespace(
    primary_names=[f"robot/{name}" for name in FOOT_GEOMS],
    data=SimpleNamespace(found=torch.ones(count, 4)),
  )
  env = SimpleNamespace(
    num_envs=count,
    device="cpu",
    step_dt=dt,
    max_episode_length_s=20,
    command=command,
    command_manager=SimpleNamespace(get_command=lambda _: command),
    scene={"feet_ground_contact": sensor},
  )
  cfg = RewardTermCfg(
    func=moving_contact_duty_variance, weight=-8.0, params={"tau_s": tau}
  )
  return env, sensor, moving_contact_duty_variance(cfg, env), cfg


def test_constant_unequal_contacts_have_analytic_cost_and_bound():
  env, sensor, term, _ = _env(1)
  sensor.data.found[:] = torch.tensor([0, 0, 1, 1])
  for _ in range(100):
    cost = term(env)
  # Two planted and two hovering feet converge towards the maximum variance .25.
  expected = 0.25 * (1 - math.exp(-2.0)) ** 2
  torch.testing.assert_close(cost, torch.tensor([expected]), atol=1e-6, rtol=0)
  assert 0 < cost.item() < 0.25
  assert -8 * cost.item() < -1.0


def test_left_right_permutation_preserves_cost_and_permutates_duty():
  env, sensor, term, _ = _env(2)
  permutation = torch.tensor([1, 0, 3, 2])
  generator = torch.Generator().manual_seed(17)
  for _ in range(100):
    contacts = torch.randint(0, 2, (4,), generator=generator)
    sensor.data.found[0] = contacts
    sensor.data.found[1] = contacts[permutation]
    cost = term(env)
    torch.testing.assert_close(cost[0], cost[1])
    torch.testing.assert_close(term.duty[0, permutation], term.duty[1])


def test_stops_clear_history_and_restart_matches_a_fresh_term():
  env, sensor, term, _ = _env()
  sensor.data.found[:] = torch.tensor([0, 0, 1, 1])
  for _ in range(50):
    term(env)
  env.command[0].zero_()
  before = term.duty.clone()
  cost = term(env)
  assert cost[0] == 0 and term.duty[0].eq(0.5).all()
  assert not torch.equal(term.duty[1:], before[1:])
  sensor.data.found[0] = torch.tensor([1, 0, 0, 1])
  env.command[0, 0] = 0.25
  fresh_env, fresh_sensor, fresh_term, _ = _env(1)
  fresh_sensor.data.found[0] = sensor.data.found[0]
  expected = fresh_term(fresh_env)
  cost = term(env)
  torch.testing.assert_close(cost[:1], expected)
  torch.testing.assert_close(term.duty[:1], fresh_term.duty)


def test_episode_subset_reset_isolated_and_safe_after_inference_steps():
  env, sensor, term, _ = _env()
  sensor.data.found[:] = torch.tensor([1, 0, 0, 1])
  with torch.inference_mode():
    for _ in range(10):
      term(env)
  before = term.duty.clone()
  term.reset(torch.tensor([1]))
  assert term.duty[1].eq(0.5).all()
  torch.testing.assert_close(term.duty[[0, 2]], before[[0, 2]])
  term.reset(slice(0, 1))
  assert term.duty[0].eq(0.5).all() and not term.duty[2].eq(0.5).all()
  term.reset()
  assert term.duty.eq(0.5).all()


def test_duty_and_cost_are_invariant_to_timestep_for_same_held_contacts():
  outcomes = []
  for dt in (0.01, 0.02, 0.05):
    env, sensor, term, _ = _env(1, dt=dt)
    for seconds, contacts in ((0.3, [1, 0, 1, 0]), (0.7, [0, 1, 0, 1])):
      sensor.data.found[:] = torch.tensor(contacts)
      for _ in range(round(seconds / dt)):
        cost = term(env)
    outcomes.append((term.duty.clone(), cost.clone()))
  for duty, cost in outcomes[1:]:
    torch.testing.assert_close(duty, outcomes[0][0], atol=1e-6, rtol=0)
    torch.testing.assert_close(cost, outcomes[0][1], atol=1e-6, rtol=0)


def test_equal_planted_or_airborne_contacts_get_zero_cost_not_gait_credit():
  env, sensor, term, _ = _env()
  sensor.data.found[1].zero_()
  for step in range(100):
    sensor.data.found[2].fill_(step % 2)
    cost = term(env)
    assert cost.eq(0).all()
  # This term cannot distinguish all-planted, flight, or synchronous hopping.
  assert term.duty[0].gt(0.9).all() and term.duty[1].lt(0.1).all()


def test_pure_yaw_and_backward_commands_activate_cost_but_zero_noise_does_not():
  env, sensor, term, _ = _env(4)
  env.command.zero_()
  env.command[0, 2] = -0.4
  env.command[1, 0] = -0.25
  env.command[3] = 1.0e-7
  sensor.data.found[:] = torch.tensor([1, 1, 0, 0])
  cost = term(env)
  assert cost[0] > 0 and cost[1] == cost[0]
  assert cost[2:].eq(0).all() and term.duty[2:].eq(0.5).all()


def test_reward_manager_applies_dt_weight_once_and_resets_term_memory():
  env, sensor, _, cfg = _env(2)
  sensor.data.found[:] = torch.tensor([0, 0, 1, 1])
  manager = RewardManager({"moving_contact_duty_variance": cfg}, env)
  first = manager.compute(dt=env.step_dt)
  expected = -8 * env.step_dt * 0.25 * (1 - math.exp(-env.step_dt)) ** 2
  torch.testing.assert_close(first, torch.full((2,), expected), atol=1e-9, rtol=0)
  manager.reset(torch.tensor([1]))
  following = manager.compute(dt=env.step_dt)
  torch.testing.assert_close(following[1], torch.tensor(expected), atol=1e-9, rtol=0)
  assert following[0] < following[1]


@pytest.mark.parametrize("tau", [0, -1, float("nan"), float("inf")])
def test_invalid_time_constants_are_rejected(tau):
  with pytest.raises(ValueError, match="tau_s must be finite and positive"):
    _env(tau=tau)
