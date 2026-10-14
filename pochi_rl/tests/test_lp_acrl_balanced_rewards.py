"""Standing, real foot lift and phase-free leg participation without height rays."""

from types import SimpleNamespace

import pytest
import torch
from mjlab.managers import RewardTermCfg
from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pochi_rl.mjlab import lp_acrl_balanced_rewards as rewards
from pochi_rl.robot import FOOT_GEOMS, FOOT_SITES

FEET = SceneEntityCfg("robot", site_names=FOOT_SITES, preserve_order=True)


def _env(num_envs=3):
  # Site 0 is intentionally not a foot: accidental all-site indexing must fail.
  site_names = ["imu", *FOOT_SITES]
  robot = SimpleNamespace(
    site_names=site_names,
    num_sites=5,
    find_sites=lambda names, preserve_order: (
      [site_names.index(name) for name in names],
      list(names),
    ),
    data=SimpleNamespace(
      joint_pos=torch.zeros(num_envs, 12),
      default_joint_pos=torch.zeros(num_envs, 12),
      joint_vel=torch.zeros(num_envs, 12),
      projected_gravity_b=torch.tensor([[0.0, 0.0, -1.0]]).repeat(num_envs, 1),
      root_link_pos_w=torch.tensor([[0.0, 0.0, 0.3114]]).repeat(num_envs, 1),
      site_pos_w=torch.zeros(num_envs, 5, 3),
      site_lin_vel_w=torch.zeros(num_envs, 5, 3),
    ),
  )
  sensor = SimpleNamespace(
    primary_names=list(FOOT_GEOMS),
    data=SimpleNamespace(
      found=torch.ones(num_envs, 4),
      current_air_time=torch.zeros(num_envs, 4),
      current_contact_time=torch.zeros(num_envs, 4),
      last_air_time=torch.zeros(num_envs, 4),
    ),
    touchdown=torch.zeros(num_envs, 4, dtype=torch.bool),
  )
  sensor.compute_first_contact = lambda dt: sensor.touchdown
  command = torch.zeros(num_envs, 3)
  env = SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    step_dt=0.02,
    max_episode_length_s=20,
    command=command,
    command_manager=SimpleNamespace(get_command=lambda _: command),
    scene={"robot": robot, "feet_ground_contact": sensor},
  )
  asset_cfg = SceneEntityCfg("robot", site_names=FOOT_SITES, preserve_order=True)
  asset_cfg.resolve(env.scene)
  return env, robot.data, sensor, asset_cfg


def _touchdown_term(env, asset_cfg):
  cfg = RewardTermCfg(
    func=rewards.balanced_touchdown,
    weight=2.0,
    params={"asset_cfg": asset_cfg},
  )
  return rewards.balanced_touchdown(cfg, env)


def test_stop_rewards_prefer_default_quiet_four_foot_stance():
  env, data, sensor, _ = _env()
  data.joint_pos[1] = 0.2
  data.joint_vel[1] = 2
  sensor.data.found[1, [1, 2]] = 0
  data.joint_pos[2] = 0.4
  data.joint_vel[2] = 4
  sensor.data.found[2] = 0
  posture = rewards.standing_posture(env)
  assert posture[0] == 1
  assert posture[0] > posture[1] > posture[2]
  assert rewards.standing_joint_velocity(env).tolist() == [0, 4, 16]
  assert rewards.standing_missing_contacts(env).tolist() == [0, 0.5, 1]


def test_very_slow_motion_and_yaw_disable_standing_rewards():
  env, data, sensor, _ = _env()
  env.command[1, 0] = 0.001
  env.command[2, 2] = -0.001
  data.joint_vel.fill_(1)
  sensor.data.found.zero_()
  assert rewards.standing_posture(env).tolist() == [1, 0, 0]
  assert rewards.standing_joint_velocity(env).tolist() == [1, 0, 0]
  assert rewards.standing_missing_contacts(env).tolist() == [1, 0, 0]


def test_slip_only_penalizes_contact_feet_and_resolves_site_and_sensor_order():
  env, data, sensor, feet = _env()
  sensor.primary_names = [f"robot/{name}" for name in reversed(FOOT_GEOMS)]
  sensor.data.found.zero_()
  sensor.data.found[:, 3] = 1  # FL, despite sensor being in reverse order.
  data.site_lin_vel_w[:, 0, :2] = 100  # IMU must not be used.
  data.site_lin_vel_w[:, 1, 0] = 2  # FL contact speed costs 4 / 4.
  data.site_lin_vel_w[:, 2:, :2] = 50  # Swing speeds are free.
  torch.testing.assert_close(rewards.contact_slip(env, feet), torch.ones(3))


def test_attitude_cost_distinguishes_upright_sideways_and_inverted():
  env, data, _, _ = _env()
  data.projected_gravity_b[1] = torch.tensor([0.0, -1.0, 0.0])
  data.projected_gravity_b[2] = torch.tensor([0.0, 0.0, 1.0])
  assert rewards.upright_error(env).tolist() == [0, 2, 4]


def test_support_height_tracks_stair_elevation_instead_of_spawn_origin():
  env, data, sensor, feet = _env()
  data.site_pos_w[1, 1:, 2] = 2.0
  data.root_link_pos_w[1, 2] += 2.0
  data.site_pos_w[2, 1:, 2] = torch.tensor([0.0, 0.0, 0.10, 0.10])
  data.root_link_pos_w[2, 2] += 0.05
  torch.testing.assert_close(
    rewards.support_height_error(env, feet), torch.zeros(3), atol=1e-10, rtol=0
  )
  # A high swinging foot cannot change the support height reference.
  sensor.data.found[2, 0] = 0
  data.site_pos_w[2, 1, 2] = 4.0
  data.root_link_pos_w[2, 2] = 0.3114 + 0.20 / 3
  assert rewards.support_height_error(env, feet)[2] < 1e-10


def test_height_error_is_bounded_and_undefined_support_in_flight_is_not_used():
  env, data, sensor, feet = _env()
  data.root_link_pos_w[0, 2] += 0.08
  data.root_link_pos_w[1, 2] += 5
  data.root_link_pos_w[2, 2] += 100
  sensor.data.found[2] = 0
  torch.testing.assert_close(
    rewards.support_height_error(env, feet), torch.tensor([1.0, 4.0, 0.0])
  )


def test_prolonged_support_diagonal_is_worse_than_regular_four_leg_motion():
  env, _, sensor, _ = _env()
  env.command[:, 0] = 0.2
  sensor.data.current_contact_time[0, [0, 3]] = 2
  sensor.data.current_air_time[0, [1, 2]] = 2
  sensor.data.current_contact_time[1] = torch.tensor([0.15, 0.0, 0.0, 0.15])
  sensor.data.current_air_time[1] = torch.tensor([0.0, 0.15, 0.15, 0.0])
  sensor.data.current_contact_time[2] = 4
  env.command[2].zero_()  # Planted feet are the intended stopped behavior.
  assert rewards.prolonged_foot_contact(env).tolist() == [1, 0, 0]


@pytest.mark.parametrize("duration", [0.10, 0.15, 0.20, 0.35])
def test_normal_pochi_swings_earn_nonnegative_credit_only_at_touchdown(duration):
  env, data, sensor, feet = _env()
  env.command[:, 0] = 0.2
  term = _touchdown_term(env, feet)
  assert not term(env, feet).any()  # Establish an initial contact reference.
  sensor.data.found.zero_()
  data.site_pos_w[:, 1:, 2] = 0.04
  assert not term(env, feet).any()  # Lift alone receives no credit.
  sensor.data.found.fill_(1)
  sensor.touchdown.fill_(True)
  sensor.data.last_air_time.fill_(duration)
  data.site_pos_w[:, 1:, 2] = 0
  torch.testing.assert_close(term(env, feet), torch.ones(3))
  sensor.touchdown.zero_()
  assert not term(env, feet).any()  # Staying planted cannot collect credit.


@pytest.mark.parametrize("duration", [0.02, 0.06, 0.8, 1.5])
def test_contact_chatter_and_long_hovering_earn_zero_never_negative(duration):
  env, data, sensor, feet = _env()
  env.command[:, 0] = 0.2
  term = _touchdown_term(env, feet)
  term(env, feet)
  sensor.data.found.zero_()
  data.site_pos_w[:, 1:, 2] = 0.1
  term(env, feet)
  sensor.data.found.fill_(1)
  sensor.touchdown.fill_(True)
  sensor.data.last_air_time.fill_(duration)
  assert not term(env, feet).any()


def test_shuffling_without_lift_and_spawn_landings_do_not_get_touchdown_credit():
  env, _, sensor, feet = _env()
  env.command[:, 0] = 0.2
  term = _touchdown_term(env, feet)
  sensor.touchdown.fill_(True)
  sensor.data.last_air_time.fill_(0.2)
  assert not term(env, feet).any()  # First observed landing has no reference.
  sensor.data.found.zero_()
  sensor.touchdown.zero_()
  term(env, feet)  # No change of foot site height.
  sensor.data.found.fill_(1)
  sensor.touchdown.fill_(True)
  assert not term(env, feet).any()


def test_real_lift_is_measured_per_foot_and_above_its_previous_step():
  env, data, sensor, feet = _env()
  env.command[:, 0] = 0.2
  sensor.primary_names = list(reversed(FOOT_GEOMS))
  data.site_pos_w[:, 1:, 2] = 1.0
  term = _touchdown_term(env, feet)
  term(env, feet)
  sensor.data.found[:, 3] = 0  # Only FL takes a step.
  data.site_pos_w[:, 1, 2] += 0.10
  term(env, feet)
  sensor.data.found[:, 3] = 1
  sensor.touchdown[:, 3] = True
  sensor.data.last_air_time[:, 3] = 0.2
  data.site_pos_w[:, 1, 2] = 1.08  # Landing higher is allowed.
  torch.testing.assert_close(term(env, feet), torch.full((3,), 0.25))
  torch.testing.assert_close(term.contact_height[:, 0], torch.full((3,), 1.08))


def test_every_leg_earns_equal_credit_without_any_diagonal_preference():
  env, data, sensor, feet = _env(num_envs=4)
  env.command[:, 0] = 0.2
  term = _touchdown_term(env, feet)
  term(env, feet)
  ids = torch.arange(4)
  sensor.data.found[ids, ids] = 0
  data.site_pos_w[ids, ids + 1, 2] = 0.04
  term(env, feet)
  sensor.data.found.fill_(1)
  sensor.touchdown[ids, ids] = True
  sensor.data.last_air_time[ids, ids] = 0.2
  data.site_pos_w[:, 1:, 2] = 0
  torch.testing.assert_close(term(env, feet), torch.full((4,), 0.25))


def test_touchdown_memory_cannot_leak_through_selective_reset_or_stop_start():
  env, data, sensor, feet = _env()
  env.command[:, 0] = 0.2
  term = _touchdown_term(env, feet)
  term(env, feet)
  sensor.data.found.zero_()
  data.site_pos_w[:, 1:, 2] = 0.04
  term(env, feet)
  term.reset(torch.tensor([0]))
  env.command[1].zero_()
  term(env, feet)  # Clear environment 1's lift while stopped and airborne.
  env.command[1, 0] = 0.2
  sensor.data.found.fill_(1)
  sensor.touchdown.fill_(True)
  sensor.data.last_air_time.fill_(0.2)
  data.site_pos_w[:, 1:, 2] = 0
  assert term(env, feet).tolist() == [0, 0, 1]


def test_reward_manager_applies_dt_once_and_resets_memory_outside_inference():
  env, data, sensor, _ = _env()
  env.command[:, 0] = 0.2
  manager = RewardManager(
    {
      "touchdown": RewardTermCfg(
        func=rewards.balanced_touchdown, weight=2.0, params={"asset_cfg": FEET}
      )
    },
    env,
  )
  with torch.inference_mode():
    manager.compute(env.step_dt)
    sensor.data.found.zero_()
    data.site_pos_w[:, 1:, 2] = 0.04
    manager.compute(env.step_dt)
    sensor.data.found.fill_(1)
    sensor.touchdown.fill_(True)
    sensor.data.last_air_time.fill_(0.2)
    data.site_pos_w[:, 1:, 2] = 0
    torch.testing.assert_close(manager.compute(env.step_dt), torch.full((3,), 0.04))
  manager.reset(torch.tensor([1]))
  term = manager._term_cfgs[0].func
  assert term.has_reference[:, 0].tolist() == [True, False, True]


def test_missing_contact_timers_fail_with_actionable_error():
  env, _, sensor, _ = _env()
  sensor.data.current_air_time = None
  with pytest.raises(ValueError, match="track_air_time=True"):
    rewards.prolonged_foot_contact(env)
