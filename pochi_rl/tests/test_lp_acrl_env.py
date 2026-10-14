"""LP task credit, episode timing, native terrain assignment and reward semantics."""

from types import SimpleNamespace

import pytest
import torch

from pochi_rl.mjlab.lp_acrl_commands import pre_reset_task
from pochi_rl.mjlab.lp_acrl_env_cfg import (
  pochi_lp_acrl_flat_env_cfg,
  pochi_lp_acrl_rough_env_cfg,
)
from pochi_rl.mjlab.lp_acrl_rewards import (
  target_position_rate,
  touchdown_air_time,
  track_linear_velocity,
  track_yaw_velocity,
)


class _Scene(dict):
  pass


def _command_env(*, rough=False, num_envs=3, **command_overrides):
  full_cfg = pochi_lp_acrl_rough_env_cfg() if rough else pochi_lp_acrl_flat_env_cfg()
  cfg = full_cfg.commands["base_velocity"]
  for name, value in command_overrides.items():
    setattr(cfg, name, value)
  scene = _Scene(
    robot=SimpleNamespace(
      data=SimpleNamespace(
        root_link_lin_vel_b=torch.zeros(num_envs, 3),
        root_link_ang_vel_b=torch.zeros(num_envs, 3),
      )
    )
  )
  scene.terrain = SimpleNamespace(
    terrain_origins=torch.arange(60).reshape(4, 5, 3).float(),
    env_origins=torch.zeros(num_envs, 3),
    terrain_types=torch.zeros(num_envs, dtype=torch.long),
    terrain_levels=torch.zeros(num_envs, dtype=torch.long),
  )
  env = SimpleNamespace(
    cfg=full_cfg,
    num_envs=num_envs,
    device="cpu",
    scene=scene,
    step_dt=0.02,
    max_episode_length=1000,
    episode_length_buf=torch.zeros(num_envs, dtype=torch.long),
    termination_manager=SimpleNamespace(dones=torch.zeros(num_envs, dtype=torch.bool)),
    reward_manager=SimpleNamespace(
      _episode_sums={
        "positive": torch.arange(num_envs).float() + 10,
        "negative": -torch.arange(num_envs).float() - 1,
      }
    ),
    extras={"log": {}},
  )
  term = cfg.build(env)
  env.command_manager = SimpleNamespace(
    get_term=lambda _: term,
    get_command=lambda _: term.command,
  )
  return env, term


def test_episode_credit_uses_old_task_and_true_total_before_reward_reset():
  env, term = _command_env()
  term.task_ids[:] = torch.tensor([4, 2, 1])
  env.episode_length_buf[:] = torch.tensor([3, 200, 0])
  # A manual mid-episode reset is not a completed rollout; startup zero-length
  # episodes are also ignored even if a previous termination flag is still set.
  env.termination_manager.dones[:] = torch.tensor([True, False, True])
  pre_reset_task(env, torch.arange(3))
  assert term.curriculum.episode_counts.tolist() == [0, 0, 0, 0, 1]
  assert term.curriculum.return_sums.tolist() == [0, 0, 0, 0, 9]
  assert term.last_completed_task_ids.tolist() == [4, -1, -1]
  assert term.last_episode_steps.tolist() == [3, 0, 0]
  # The hook reads, but does not prematurely clear, the reward manager's sums.
  assert env.reward_manager._episode_sums["positive"].tolist() == [10, 11, 12]


def test_auxiliary_standing_and_stopstart_returns_never_enter_fixed_task_lp():
  env, term = _command_env(
    standing_episode_fraction=0.2, transition_episode_fraction=0.1
  )
  term.task_ids[:] = torch.tensor([4, 2, 1])
  term.episode_mode[:] = torch.tensor([0, 1, 2])
  env.episode_length_buf[:] = 1000
  env.termination_manager.dones[:] = True
  pre_reset_task(env, torch.arange(3))
  assert term.curriculum.episode_counts.tolist() == [0, 0, 0, 0, 1]
  assert term.curriculum.return_sums.tolist() == [0, 0, 0, 0, 9]
  assert term.last_completed_episode_mode.tolist() == [0, 1, 2]
  assert term.last_episode_lp_credited.tolist() == [True, False, False]
  assert term.last_episode_returns.tolist() == [9, 9, 9]


def test_dedicated_standing_episodes_remain_exactly_zero():
  env, term = _command_env(rough=True, standing_episode_fraction=1.0)
  pre_reset_task(env, torch.arange(3))
  term.reset(torch.arange(3))
  assert term.episode_mode.tolist() == [1, 1, 1]
  assert term.is_standing_env.all()
  term.compute(8.0)
  assert not term.command.any()


def test_stopstart_switches_between_fixed_movement_and_zero_without_lp_credit():
  env, term = _command_env(
    rough=True, transition_episode_fraction=1.0,
    transition_interval_s=(0.04, 0.04),
    transition_straight_probability=1.0, transition_forward_range=(0.2, 0.2),
  )
  pre_reset_task(env, torch.arange(3))
  term.reset(torch.arange(3))
  original = term.command.clone()
  standing = term.is_standing_env.clone()
  movement = term._movement_command.clone()
  torch.testing.assert_close(movement[:, 0].abs(), torch.full((3,), 0.2))
  assert not movement[:, 1:].any()
  term.compute(0.0)
  term.compute(0.02)
  torch.testing.assert_close(term.command, original)
  term.compute(0.02)
  assert torch.equal(term.is_standing_env, ~standing)
  torch.testing.assert_close(term.command[standing], movement[standing])
  assert not term.command[~standing].any()
  term.compute(0.04)
  torch.testing.assert_close(term.command, original)
  env.episode_length_buf[:] = 1000
  env.termination_manager.dones[:] = True
  pre_reset_task(env, torch.arange(3))
  assert not term.curriculum.episode_counts.any()


def test_zero_yaw_sampling_stays_inside_its_catalog_bin_and_evaluation_is_fixed():
  env, term = _command_env(
    rough=True, zero_yaw_probability=1.0, zero_lateral_probability=1.0,
    standing_episode_fraction=0.2, transition_episode_fraction=0.1,
  )
  term.set_evaluation(task_ids=torch.tensor([0, 20, 599]))
  pre_reset_task(env, torch.arange(3))
  term.reset(torch.arange(3))
  assert term.command[0, 2] == 0
  assert term.command[1, 2].abs() >= 0.2
  assert term.command[2, 2].abs() >= 1.0
  assert not term.command[:, 1].any()
  assert not term.episode_mode.any()
  before = term.command.clone()
  term.compute(20.0)
  torch.testing.assert_close(term.command, before)
  term.set_evaluation(command=(0, 0, 0))
  pre_reset_task(env, torch.arange(3))
  term.reset(torch.arange(3))
  assert term.is_standing_env.all() and not term.command.any()


def test_auxiliary_identity_changes_only_when_enabled_and_ignores_viewer_flags():
  _, legacy = _command_env()
  assert "command_protocol" not in legacy.checkpoint_identity()
  _, mixed = _command_env(standing_episode_fraction=0.2)
  before = mixed.checkpoint_identity()
  assert before["command_protocol"]["auxiliary_episode_lp_credit"] is False
  mixed.cfg.manual_control_on_play = True
  assert mixed.checkpoint_identity() == before
  mixed.cfg.standing_episode_fraction = 0.3
  assert mixed.checkpoint_identity() != before


def test_manual_play_starts_enabled_and_zero_before_first_step(monkeypatch):
  from mjlab.tasks.velocity.mdp.velocity_command import UniformVelocityCommand

  def fake_gui(term, *args, **kwargs):
    term._joystick_enabled = SimpleNamespace(value=False)
    term._joystick_sliders = [SimpleNamespace(value=0.0) for _ in range(3)]
    term._joystick_get_env_idx = lambda: 0

  monkeypatch.setattr(UniformVelocityCommand, "create_gui", fake_gui)
  _, term = _command_env(manual_control_on_play=True)
  term.set_evaluation()
  term.vel_command_b[0] = 0.3
  term.create_gui()
  assert term._joystick_enabled.value
  assert not term.command[0].any()
  assert term.is_standing_env[0]


def test_reset_task_places_same_terrain_used_to_sample_command():
  env, term = _command_env(rough=True)
  tasks = torch.tensor([0, 37, 599])
  term.set_evaluation(task_ids=tasks)
  pre_reset_task(env, torch.arange(3))
  term.reset(torch.arange(3))
  coordinates = term.catalog.task_indices()[tasks]
  terrain = env.scene.terrain
  torch.testing.assert_close(terrain.terrain_types, coordinates[:, 2])
  torch.testing.assert_close(terrain.terrain_levels, coordinates[:, 3])
  torch.testing.assert_close(
    terrain.env_origins,
    terrain.terrain_origins[coordinates[:, 3], coordinates[:, 2]],
  )
  bounds = term.catalog.task_bounds()[tasks]
  magnitudes = term.command[:, [0, 2]].abs()
  assert ((bounds[:, :, 0] <= magnitudes) & (magnitudes <= bounds[:, :, 1])).all()
  assert (term.command[:, 1].abs() <= 0.1).all()


def test_commands_stay_fixed_across_stage_updates_and_dt_zero_is_not_a_step():
  env, term = _command_env()
  pre_reset_task(env, torch.arange(3))
  term.reset(torch.arange(3))
  before = term.command.clone()
  term.time_left.zero_()
  term.curriculum.stage_steps = 2
  term.compute(0.0)
  assert term.curriculum.steps_in_stage == 0
  term.compute(0.02)
  term.compute(0.02)
  assert term.curriculum.stage_index == 1
  torch.testing.assert_close(term.command, before)
  assert not term.command[:, 1:].any()
  term.set_evaluation(command=(0.12, 0.0, 0.0))
  pre_reset_task(env, torch.arange(3))
  term.reset(torch.arange(3))
  term.compute(0.02)
  assert term.curriculum.steps_in_stage == 0
  torch.testing.assert_close(term.command[:, 0], torch.full((3,), 0.12))


def test_task_directions_cover_both_signs_within_magnitude_bins():
  torch.manual_seed(5)
  env, term = _command_env(rough=True, num_envs=128)
  term.set_evaluation(task_ids=599)
  pre_reset_task(env, torch.arange(128))
  term.reset(torch.arange(128))
  for axis in (0, 1, 2):
    assert (term.command[:, axis] < 0).any()
    assert (term.command[:, axis] > 0).any()


def test_checkpoint_task_identity_rejects_flat_rough_mix():
  _, flat = _command_env()
  _, rough = _command_env(rough=True)
  with pytest.raises(ValueError, match="task space"):
    rough.load_curriculum_state_dict(flat.curriculum_state_dict())


def test_full_resume_rejects_changed_reward_and_terrain_seed():
  flat_env, flat = _command_env()
  state = flat.curriculum_state_dict()
  flat_env.cfg.rewards["joint_acc_l2"].weight = -0.001
  with pytest.raises(ValueError, match="task space"):
    flat.load_curriculum_state_dict(state)
  rough_env, rough = _command_env(rough=True)
  state = rough.curriculum_state_dict()
  rough_env.cfg.scene.terrain.terrain_generator.seed = 123
  # Geometry seeds can change for actor evaluation; full curriculum resume
  # must preserve the exact task geometry underlying its return baselines.
  assert rough.checkpoint_identity() == state["task_space"]
  with pytest.raises(ValueError, match="seed"):
    rough.load_curriculum_state_dict(state)


def test_tracking_excludes_separately_penalized_vertical_and_roll_pitch_velocity():
  env, term = _command_env()
  term.vel_command_b.zero_()
  env.scene["robot"].data.root_link_lin_vel_b[:, 2] = 5.0
  env.scene["robot"].data.root_link_ang_vel_b[:, :2] = 5.0
  torch.testing.assert_close(track_linear_velocity(env), torch.ones(3))
  torch.testing.assert_close(track_yaw_velocity(env), torch.ones(3))


def test_airtime_is_paid_on_touchdown_and_target_rate_includes_radian_scale():
  env, term = _command_env()
  term.vel_command_b[:, 0] = torch.tensor([0.2, 0.0, 0.2])
  env.scene["feet_ground_contact"] = SimpleNamespace(
    data=SimpleNamespace(
      last_air_time=torch.tensor(
        [
          [0.6, 0.7, 0.8, 0.9],
          [0.6, 0.7, 0.8, 0.9],
          [0.6, 0.7, 0.8, 0.9],
        ]
      )
    ),
    compute_first_contact=lambda dt: torch.tensor(
      [
        [True, False, False, False],
        [True, True, True, True],
        [False, False, False, False],
      ]
    ),
  )
  torch.testing.assert_close(touchdown_air_time(env), torch.tensor([0.1, 0.0, 0.0]))
  env.action_manager = SimpleNamespace(
    action=torch.ones(3, 12), prev_action=torch.zeros(3, 12)
  )
  torch.testing.assert_close(target_position_rate(env), torch.full((3,), 0.75))


def test_new_tasks_leave_other_tasks_and_robot_constants_unchanged():
  from pochi_rl.mjlab.entity_cfg import POCHI_ACTUATOR
  from pochi_rl.mjlab.velocity_flat_env_cfg import pochi_velocity_flat_env_cfg

  flat, rough = pochi_lp_acrl_flat_env_cfg(), pochi_lp_acrl_rough_env_cfg()
  for cfg in (flat, rough):
    actuator = cfg.scene.entities["robot"].articulation.actuators[0]
    assert (actuator.stiffness, actuator.damping) == (140.0, 1.5)
    assert cfg.sim.mujoco.timestep * cfg.decimation == pytest.approx(0.02)
    assert "gait_phase" not in cfg.observations["actor"].terms
    assert "contact_schedule" not in cfg.rewards
    assert list(cfg.curriculum) == ["learning_progress"]
  assert POCHI_ACTUATOR.stiffness == 60.0
  assert pochi_velocity_flat_env_cfg().commands["base_velocity"].heading_command
  scanner = rough.scene.sensors[-1]
  assert scanner.pattern.generate_rays(None, "cpu")[0].shape == (275, 3)
  generator = rough.scene.terrain.terrain_generator
  assert generator.num_rows == 4
  assert len(generator.sub_terrains) == 5
  # The maximum20s travel incl. lateral commands fits within its task tile.
  assert generator.size[0] / 2 > 20 * (0.5**2 + 0.1**2) ** 0.5 + 0.5


def test_native_terrain_levels_change_geometry_and_center_up_down_direction():
  from mjlab.terrains import TerrainEntity

  from pochi_rl.mjlab.lp_acrl_env_cfg import lp_acrl_terrain_cfg

  terrain = TerrainEntity(lp_acrl_terrain_cfg(), device="cpu")
  origins = terrain.terrain_origins
  assert origins.shape == (4, 5, 3)
  heights = origins[:, :, 2]
  # Outward ascending stairs/uphill start below their outer edge. Harder
  # levels put their centers deeper; descending terrain starts higher.
  for index in (0, 3):
    assert (heights[:, index] < 0).all()
    assert (heights[1:, index] < heights[:-1, index]).all()
  for index in (1, 4):
    assert (heights[:, index] > 0).all()
    assert (heights[1:, index] > heights[:-1, index]).all()
  assert (heights[1:, 2] > heights[:-1, 2]).all()


def test_real_flat_rollout_credits_complete_return_before_auto_reset():
  from mjlab.envs import ManagerBasedRlEnv

  cfg = pochi_lp_acrl_flat_env_cfg()
  cfg.scene.num_envs = 2
  cfg.episode_length_s = 0.08
  cfg.commands["base_velocity"].stage_steps = 100
  cfg.events.pop("base_mass", None)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    obs, _ = env.reset(seed=123)
    term = env.command_manager.get_term("base_velocity")
    old_ids = term.task_ids.clone()
    commands = term.command.clone()
    assert obs["actor"].shape == (2, 48)
    accumulated = torch.zeros(2)
    for step in range(4):
      obs, reward, terminated, truncated, _ = env.step(torch.zeros(2, 12))
      assert torch.isfinite(obs["actor"]).all()
      accumulated += reward
      if step < 3:
        torch.testing.assert_close(term.command, commands)
        assert not (terminated | truncated).any()
    assert truncated.all()
    torch.testing.assert_close(term.last_episode_returns, accumulated)
    torch.testing.assert_close(term.last_completed_task_ids, old_ids)
    assert term.curriculum.episode_counts.sum() == 2
    assert term.curriculum.steps_in_stage == 4
    assert all(not values.any() for values in env.reward_manager._episode_sums.values())
  finally:
    env.close()
