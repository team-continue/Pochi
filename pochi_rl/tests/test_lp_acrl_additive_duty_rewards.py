"""V14 combines two frozen signals; neither replaces nor advances the other."""

import math
from copy import deepcopy

import pytest
import torch
from mjlab.managers import RewardManager, RewardTermCfg
from test_lp_acrl_qualified_swing_rewards import _setup

from pochi_rl.mjlab.lp_acrl_blind_v3_env_cfg import pochi_lp_acrl_blind_v3_env_cfg
from pochi_rl.mjlab.lp_acrl_blind_v3_rewards import moving_contact_duty_variance
from pochi_rl.mjlab.lp_acrl_commands import _config_identity

RAW = "moving_contact_duty_variance"
QUALIFIED = "moving_qualified_swing_occupancy"


def _manager(count=2, dt=0.02, *, reverse=False):
  env, sensor, _, qualified_cfg = _setup(
    count, dt, target=0.2, tau_s=1.0, min_air=0.06, min_lift=0.025
  )
  raw_cfg = RewardTermCfg(
    func=moving_contact_duty_variance, weight=-8.0, params={"tau_s": 1.0}
  )
  assert _config_identity(raw_cfg) == _config_identity(
    pochi_lp_acrl_blind_v3_env_cfg().rewards[RAW]
  )
  terms = {QUALIFIED: qualified_cfg, RAW: raw_cfg}
  if reverse:
    terms = dict(reversed(list(terms.items())))
  manager = RewardManager(terms, env)
  baseline = RewardManager({QUALIFIED: qualified_cfg}, env)
  return env, sensor, manager, baseline


def _inputs(env, sensor, contact, lift=0.0):
  """Advance only the fixture sensor, before either manager reads its snapshot."""
  sensor.data.found[:] = torch.as_tensor(contact)
  sensor.data.current_air_time.copy_(
    torch.where(sensor.data.found > 0, 0, sensor.data.current_air_time + env.step_dt)
  )
  env.scene["robot"].data.site_pos_w[:, :, 2] = torch.as_tensor(lift)


def _rates(manager, name):
  return manager._step_reward[:, manager.active_terms.index(name)]


@pytest.mark.parametrize("dt", [0.01, 0.02, 0.05])
def test_actual_manager_adds_exact_historical_weight_and_dt_once(dt):
  env, sensor, manager, baseline = _manager(dt=dt)
  _inputs(env, sensor, [0, 0, 1, 1])
  raw_before = sensor.data.found.clone()
  air_before = sensor.data.current_air_time.clone()
  total = manager.compute(dt).clone()
  old_total = baseline.compute(dt).clone()
  alpha = 1 - math.exp(-dt)
  # Population variance of [.5-.5a,.5-.5a,.5+.5a,.5+.5a] is .25*a^2.
  raw_rate = -8 * 0.25 * alpha**2
  qualified_rate = -8 * 0.2**2 * alpha**2
  torch.testing.assert_close(
    total, torch.full((2,), (raw_rate + qualified_rate) * dt), atol=2e-9, rtol=0
  )
  torch.testing.assert_close(
    total - old_total, torch.full((2,), raw_rate * dt), atol=2e-9, rtol=0
  )
  torch.testing.assert_close(
    _rates(manager, RAW), torch.full((2,), raw_rate), atol=3e-8, rtol=0
  )
  torch.testing.assert_close(manager._episode_sums[RAW], total - old_total)
  torch.testing.assert_close(
    manager.get_term_cfg(QUALIFIED).func.occupancy,
    baseline.get_term_cfg(QUALIFIED).func.occupancy,
  )
  # Reward evaluation consumes the same cached reward-time sensor snapshot.
  assert torch.equal(sensor.data.found, raw_before)
  assert torch.equal(sensor.data.current_air_time, air_before)


def test_all_planted_raw_zero_does_not_remove_existing_no_lift_cost():
  env, sensor, manager, baseline = _manager()
  for _ in range(500):
    _inputs(env, sensor, 1)
    total = manager.compute(env.step_dt).clone()
    old_total = baseline.compute(env.step_dt).clone()
  assert _rates(manager, RAW).eq(0).all()
  expected = -8 * 0.2**2 * (1 - math.exp(-10)) ** 2
  torch.testing.assert_close(
    _rates(manager, QUALIFIED), torch.full((2,), expected), atol=2e-6, rtol=0
  )
  torch.testing.assert_close(total, old_total, atol=0, rtol=0)
  assert total.lt(-0.0063).all()


def test_contact_balance_improvement_cannot_impersonate_a_qualifying_lift():
  env, sensor, manager, _ = _manager()
  raw_history = []
  for step in range(1000):
    # World0 has front duty.25/rear.75; world1 has .5 per foot. All lifts5mm.
    phase = step % 20
    unequal = torch.tensor([phase < 5, phase < 5, phase < 15, phase < 15])
    balanced = torch.tensor([phase < 10, phase >= 10, phase >= 10, phase < 10])
    contact = torch.stack((unequal, balanced))
    _inputs(env, sensor, contact, (~contact) * 0.005)
    manager.compute(env.step_dt)
    if step >= 500:
      raw_history.append(_rates(manager, RAW).clone())
  means = torch.stack(raw_history).mean(0)
  assert means[0] < -0.45 and means[1] > -0.04
  q = manager.get_term_cfg(QUALIFIED).func
  assert not q.qualified.any()
  torch.testing.assert_close(q.occupancy[0], q.occupancy[1], atol=0, rtol=0)
  torch.testing.assert_close(
    _rates(manager, QUALIFIED), torch.full((2,), -0.32), atol=1e-6, rtol=0
  )


def test_real_lift_changes_qualification_without_changing_raw_contact_cost():
  env, sensor, manager, _ = _manager()
  _inputs(env, sensor, 1)
  manager.compute(env.step_dt)
  q_history = []
  phases = torch.tensor([0, 10, 10, 0])
  for step in range(1000):
    airborne = (step + phases) % 20 < 6
    contact = (~airborne).expand(2, -1)
    # Identical contacts/durations: first world stays below25mm; second lifts30mm.
    lifts = airborne * torch.tensor([[0.01], [0.03]])
    _inputs(env, sensor, contact, lifts)
    manager.compute(env.step_dt)
    torch.testing.assert_close(_rates(manager, RAW)[0], _rates(manager, RAW)[1])
    if step >= 500:
      q_history.append(_rates(manager, QUALIFIED).clone())
  means = torch.stack(q_history).mean(0)
  assert means[0] < -0.319 and means[1] > -0.02
  assert (
    manager.get_term_cfg(RAW).func.duty[0].equal(manager.get_term_cfg(RAW).func.duty[1])
  )


def test_combined_zero_command_and_subset_episode_reset_clear_both_histories():
  env, sensor, manager, _ = _manager(3)
  _inputs(env, sensor, 1)
  manager.compute(env.step_dt)
  for _ in range(40):
    _inputs(env, sensor, [0, 0, 1, 1], [0.03, 0.03, 0, 0])
    manager.compute(env.step_dt)
  raw = manager.get_term_cfg(RAW).func
  qualified = manager.get_term_cfg(QUALIFIED).func
  env.command[0].zero_()
  env.command[1].zero_()
  env.command[1, 2] = 0.3  # Pure yaw remains moving, as in the frozen kernel.
  env.command[2].fill_(1.0e-7)
  _inputs(env, sensor, [0, 0, 1, 1], [0.03, 0.03, 0, 0])
  manager.compute(env.step_dt)
  assert raw.duty[[0, 2]].eq(0.5).all()
  assert qualified.occupancy[[0, 2]].eq(0.2).all()
  assert not qualified.has_reference[[0, 2]].any()
  assert _rates(manager, RAW)[[0, 2]].eq(0).all()
  assert _rates(manager, QUALIFIED)[[0, 2]].eq(0).all()
  assert _rates(manager, RAW)[1] < 0
  before = (raw.duty.clone(), qualified.occupancy.clone())
  manager.reset(torch.tensor([0]))
  torch.testing.assert_close(raw.duty[1:], before[0][1:], atol=0, rtol=0)
  torch.testing.assert_close(qualified.occupancy[1:], before[1][1:], atol=0, rtol=0)
  manager.reset(torch.tensor([1]))
  assert raw.duty.eq(0.5).all() and qualified.occupancy.eq(0.2).all()
  assert not qualified.qualified.any() and not qualified.has_reference.any()
  env.command[:, 0] = -0.25
  _inputs(env, sensor, 1)
  total = manager.compute(env.step_dt)
  torch.testing.assert_close(total[0], total[1], atol=0, rtol=0)


def test_reward_order_preserves_both_states_and_never_updates_contact_history():
  first = _manager(1)
  second = _manager(1, reverse=True)
  for step in range(60):
    contact = torch.tensor([(step + phase) % 20 >= 6 for phase in (0, 10, 10, 0)])
    values = []
    for env, sensor, manager, _ in (first, second):
      _inputs(env, sensor, contact, (~contact) * 0.03)
      before = deepcopy(vars(sensor.data))
      values.append(manager.compute(env.step_dt).clone())
      for key, value in before.items():
        assert torch.equal(value, getattr(sensor.data, key))
    torch.testing.assert_close(values[0], values[1], atol=0, rtol=0)
    for name, attribute in ((RAW, "duty"), (QUALIFIED, "occupancy")):
      torch.testing.assert_close(
        getattr(first[2].get_term_cfg(name).func, attribute),
        getattr(second[2].get_term_cfg(name).func, attribute),
        atol=0,
        rtol=0,
      )
