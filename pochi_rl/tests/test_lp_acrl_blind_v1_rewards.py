"""Real foot cycles cannot be replaced by contact chatter or stale swing state."""

from types import SimpleNamespace

import pytest
import torch
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pochi_rl.mjlab import lp_acrl_blind_v1_rewards as rewards
from pochi_rl.robot import FOOT_GEOMS

FEET = SceneEntityCfg("robot")


def _env(count=3):
  command = torch.zeros(count, 3)
  data = SimpleNamespace(
    joint_vel=torch.zeros(count, 12),
    root_link_lin_vel_b=torch.zeros(count, 3),
    site_pos_w=torch.zeros(count, 4, 3),
  )
  sensor = SimpleNamespace(
    primary_names=list(FOOT_GEOMS),
    data=SimpleNamespace(
      found=torch.ones(count, 4), last_air_time=torch.zeros(count, 4)
    ),
    touchdown=torch.zeros(count, 4, dtype=torch.bool),
  )
  sensor.compute_first_contact = lambda dt: sensor.touchdown
  env = SimpleNamespace(
    num_envs=count,
    device="cpu",
    step_dt=0.02,
    command=command,
    command_manager=SimpleNamespace(get_command=lambda _: command),
    scene={"robot": SimpleNamespace(data=data), "feet_ground_contact": sensor},
  )
  term = rewards.valid_foot_cycle_age(
    RewardTermCfg(func=rewards.valid_foot_cycle_age, weight=-0.6), env
  )
  return env, data, sensor, term


def test_micro_contact_breaks_never_reset_valid_cycle_age():
  env, data, sensor, term = _env()
  env.command[:, 0] = 0.25
  term(env, FEET)
  observed_valid_cycles = torch.zeros_like(term.valid_cycle)
  for step in range(100):
    grounded = step % 2 == 1
    sensor.data.found.fill_(grounded)
    sensor.touchdown.fill_(grounded)
    sensor.data.last_air_time.fill_(0.02)
    data.site_pos_w[:, :, 2] = 0 if grounded else 0.004
    cost = term(env, FEET)
    observed_valid_cycles |= term.valid_cycle
  assert not observed_valid_cycles.any()
  assert (term.age_s > 2).all()
  torch.testing.assert_close(cost, torch.ones(3))


def test_valid_cycle_requires_both_real_lift_and_sufficient_air_time():
  env, data, sensor, term = _env(4)
  env.command[:, 0] = 0.25
  term(env, FEET)
  term.age_s.fill_(1.2)
  sensor.data.found.zero_()
  data.site_pos_w[:, :, 2] = torch.tensor([0.025, 0.035, 0.010, 0.035])[:, None]
  term(env, FEET)
  sensor.data.found.fill_(1)
  sensor.touchdown.fill_(True)
  sensor.data.last_air_time[:] = torch.tensor([0.06, 0.04, 0.10, 0.10])[:, None]
  data.site_pos_w[:, :, 2] = 0
  cost = term(env, FEET)
  assert term.valid_cycle[:, 0].tolist() == [True, False, False, True]
  assert term.age_s[0].eq(0).all() and term.age_s[3].eq(0).all()
  assert term.age_s[1].gt(1.2).all() and term.age_s[2].gt(1.2).all()
  assert cost[0] == 0 and cost[3] == 0 and cost[1] > 0 and cost[2] > 0


def test_each_leg_has_its_own_clock_and_contact_name_mapping():
  env, data, sensor, term = _env(4)
  env.command[:, 0] = -0.25
  sensor.primary_names = [f"robot/{name}" for name in reversed(FOOT_GEOMS)]
  term(env, FEET)
  term.age_s.fill_(1.0)
  ids = torch.arange(4)
  sensor.data.found[ids, 3 - ids] = 0
  data.site_pos_w[ids, ids, 2] = 0.035
  term(env, FEET)
  sensor.data.found.fill_(1)
  sensor.touchdown[ids, 3 - ids] = True
  sensor.data.last_air_time[ids, 3 - ids] = 0.10
  data.site_pos_w.zero_()
  term(env, FEET)
  torch.testing.assert_close(term.valid_cycle, torch.eye(4, dtype=torch.bool))
  assert term.age_s[ids, ids].eq(0).all()
  assert term.age_s[~torch.eye(4, dtype=torch.bool)].gt(1).all()


def test_spawn_landing_does_not_invent_a_prior_ground_reference():
  env, data, sensor, term = _env()
  env.command[:, 0] = 0.25
  term.age_s.fill_(1)
  data.site_pos_w[:, :, 2] = 2
  sensor.touchdown.fill_(True)
  sensor.data.last_air_time.fill_(0.10)
  term(env, FEET)
  assert not term.valid_cycle.any()
  assert term.has_reference.all() and term.age_s.gt(1).all()


def test_stopping_clears_age_and_unfinished_swing_before_restarting():
  env, data, sensor, term = _env()
  env.command[:, 0] = 0.25
  term(env, FEET)
  term.age_s.fill_(1.6)
  sensor.data.found.zero_()
  data.site_pos_w[:, :, 2] = 0.035
  term(env, FEET)
  env.command[0].zero_()
  cost = term(env, FEET)
  assert cost[0] == 0 and term.age_s[0].eq(0).all()
  assert term.peak_lift[0].eq(0).all() and not term.has_reference[0].any()
  assert term.age_s[1:].gt(1.6).all()
  env.command[0, 0] = 0.25
  sensor.data.found.fill_(1)
  sensor.touchdown.fill_(True)
  sensor.data.last_air_time.fill_(0.10)
  data.site_pos_w.zero_()
  term(env, FEET)
  assert not term.valid_cycle[0].any()
  torch.testing.assert_close(term.age_s[0], torch.full((4,), 0.02))
  assert term.valid_cycle[1:].all()


def test_selective_episode_reset_is_isolated_and_safe_after_inference():
  env, _, _, term = _env()
  env.command[:, 0] = 0.25
  with torch.inference_mode():
    for _ in range(70):
      term(env, FEET)
  before = term.age_s.clone()
  term.reset(torch.tensor([1]))
  torch.testing.assert_close(term.age_s[[0, 2]], before[[0, 2]])
  assert term.age_s[1].eq(0).all() and not term.has_reference[1].any()
  assert term.has_reference[[0, 2]].all()


def test_pure_yaw_needs_foot_cycles_while_zero_and_command_noise_do_not():
  env, _, _, term = _env()
  env.command[0, 2] = -0.4
  env.command[2] = 1e-7
  for _ in range(100):
    cost = term(env, FEET)
  assert cost.tolist() == [1, 0, 0]
  assert rewards.signed_translation_progress(env).eq(0).all()


def test_stopped_velocity_cost_is_bounded_scaled_and_disabled_while_moving():
  env, data, _, _ = _env(5)
  data.joint_vel[:] = torch.tensor([0, 0.10, 0.16, 1e20, 0.16])[:, None]
  env.command[4, 2] = 0.2
  cost = rewards.standing_joint_velocity_bounded(env)
  torch.testing.assert_close(cost, torch.tensor([0, 0.5, 0.0256 / 0.0356, 1, 0]))
  assert torch.isfinite(cost).all() and ((0 <= cost) & (cost <= 1)).all()


def test_signed_progress_rewards_both_directions_and_never_pays_for_overspeed():
  env, data, _, _ = _env(9)
  env.command[:, 0] = torch.tensor([0.25, -0.25, 0.25, 0.25, 0, 0, 0.049, 0, 0.05])
  env.command[4, 2] = 0.4
  env.command[7, 1] = 0.10
  data.root_link_lin_vel_b[:, 0] = torch.tensor(
    [0.25, -0.25, -0.25, 3, 2, 2, 2, 0, 0.05]
  )
  data.root_link_lin_vel_b[7, 1] = 0.10
  torch.testing.assert_close(
    rewards.signed_translation_progress(env),
    torch.tensor([1, 1, -1, 1, 0, 0, 0, 1, 1], dtype=torch.float32),
  )


def test_symmetric_rocking_cancels_and_orthogonal_motion_has_no_progress():
  env, data, _, _ = _env()
  env.command[:, 0] = 0.25
  data.root_link_lin_vel_b[:, 0] = torch.tensor([0.4, -0.4, 0])
  data.root_link_lin_vel_b[2, 1] = 0.5
  progress = rewards.signed_translation_progress(env)
  assert progress[:2].sum() == 0 and progress[2] == 0


@pytest.mark.parametrize(
  "params", [{"min_air": 0}, {"min_lift": float("nan")}, {"ramp_s": 0}, {"grace_s": -1}]
)
def test_invalid_clock_configuration_is_rejected(params):
  env, _, _, _ = _env()
  cfg = RewardTermCfg(func=rewards.valid_foot_cycle_age, weight=-0.6, params=params)
  with pytest.raises(ValueError, match="must be finite"):
    rewards.valid_foot_cycle_age(cfg, env)
