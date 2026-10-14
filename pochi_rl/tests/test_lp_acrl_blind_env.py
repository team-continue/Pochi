"""Contracts for the separate proprioception-only standing/walking task."""

import math

import pytest

from pochi_rl.mjlab.lp_acrl_blind_env_cfg import pochi_lp_acrl_blind_env_cfg
from pochi_rl.mjlab.lp_acrl_env_cfg import pochi_lp_acrl_rough_env_cfg


def test_blind_actor_and_critic_have_no_terrain_sensor_or_observation():
  cfg = pochi_lp_acrl_blind_env_cfg()
  assert [sensor.name for sensor in cfg.scene.sensors] == [
    "feet_ground_contact", "limb_ground_contact"
  ]
  expected = (
    "base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands",
    "joint_pos_rel", "joint_vel_rel", "last_action",
  )
  for group in cfg.observations.values():
    assert tuple(group.terms) == expected
  assert cfg.commands["base_velocity"].rough
  assert cfg.sim.mujoco.timestep * cfg.decimation == pytest.approx(0.02)


def test_new_blind_task_keeps_original_teacher_configuration_unchanged():
  original = pochi_lp_acrl_rough_env_cfg()
  blind = pochi_lp_acrl_blind_env_cfg()
  assert "feet_air_time" in original.rewards
  assert "feet_air_time" not in blind.rewards
  assert original.scene.sensors[-1].name == "terrain_scan"
  assert "height_scan" in original.observations["actor"].terms
  assert original.commands["base_velocity"].standing_episode_fraction == 0
  assert original.commands["base_velocity"].transition_episode_fraction == 0
  assert original.rewards["undesired_contacts"].weight == -0.001
  assert blind.rewards["undesired_contacts"].weight == -1.0
  assert (
    blind.scene.entities["robot"].init_state
    == original.scene.entities["robot"].init_state
  )


def test_blind_play_starts_with_manual_zero_and_same_training_contract():
  train, play = pochi_lp_acrl_blind_env_cfg(), pochi_lp_acrl_blind_env_cfg(play=True)
  command = play.commands["base_velocity"]
  assert command.manual_control_on_play
  assert not command.adaptive
  assert train.commands["base_velocity"].adaptive
  assert command.standing_episode_fraction == 0.2
  assert command.transition_episode_fraction == 0.1
  assert command.zero_yaw_probability == 0.75
  assert command.zero_lateral_probability == 0.5
  assert train.rewards.keys() == play.rewards.keys()


def test_blind_heightfield_spacing_keeps_torso_candidate_bound_below_warp_cap():
  old = pochi_lp_acrl_rough_env_cfg().scene.terrain.terrain_generator
  new = pochi_lp_acrl_blind_env_cfg().scene.terrain.terrain_generator
  assert (new.num_rows, new.num_cols, new.size) == (4, 5, (24.0, 24.0))
  torso_radius = math.sqrt(sum(half**2 for half in (0.1985, 0.10925, 0.06)))
  for name in ("rough", "uphill", "downhill"):
    assert old.sub_terrains[name].terrain.horizontal_scale == 0.1
    spacing = new.sub_terrains[name].terrain.horizontal_scale
    assert spacing == 0.2
    conservative_triangles = 2 * (math.ceil(2 * torso_radius / spacing) + 1) ** 2
    assert conservative_triangles < 50
