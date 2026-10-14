"""Prospective v4 replaces one objective without altering the rest of v3."""

import importlib
from dataclasses import asdict

import pytest
from mjlab.tasks import registry

from pochi_rl.mjlab.lp_acrl_blind_v2 import QUIET_STOP_ACTOR
from pochi_rl.mjlab.lp_acrl_blind_v3 import pochi_lp_acrl_blind_v3_ppo_runner_cfg
from pochi_rl.mjlab.lp_acrl_blind_v3_env_cfg import pochi_lp_acrl_blind_v3_env_cfg
from pochi_rl.mjlab.lp_acrl_blind_v4_env_cfg import (
  pochi_lp_acrl_blind_v4_env_cfg,
  pochi_lp_acrl_blind_v4_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_qualified_swing_rewards import (
  moving_qualified_swing_occupancy,
)
from pochi_rl.mjlab.lp_acrl_quiet_stop import QuietStopMLPModel
from pochi_rl.robot import FOOT_SITES


@pytest.mark.parametrize("play", [False, True])
def test_v4_changes_only_the_approved_reward_replacement(play):
  old = pochi_lp_acrl_blind_v3_env_cfg(play=play)
  new = pochi_lp_acrl_blind_v4_env_cfg(play=play)
  assert "moving_contact_duty_variance" not in new.rewards
  assert "moving_qualified_swing_occupancy" not in old.rewards
  added = new.rewards.pop("moving_qualified_swing_occupancy")
  assert added.func is moving_qualified_swing_occupancy
  assert added.weight == -8.0
  feet = added.params["asset_cfg"]
  assert feet.name == "robot"
  assert tuple(feet.site_names) == FOOT_SITES and feet.preserve_order
  assert {key: value for key, value in added.params.items() if key != "asset_cfg"} == {
    "target": 0.20, "tau_s": 1.0, "min_air": 0.06, "min_lift": 0.025,
  }
  assert new.rewards["contact_slip"].weight == old.rewards["contact_slip"].weight == -8
  new.rewards["moving_contact_duty_variance"] = old.rewards[
    "moving_contact_duty_variance"
  ]
  # Both factories construct equivalent fresh terrain callbacks.
  assert (
    new.scene.terrain.spec_fn.__qualname__ == old.scene.terrain.spec_fn.__qualname__
  )
  new.scene.terrain.spec_fn = old.scene.terrain.spec_fn
  assert new == old


def test_v4_preserves_48d_proprioception_and_standing_transition_protocol():
  cfg = pochi_lp_acrl_blind_v4_env_cfg()
  expected_sizes = {
    "base_lin_vel": 3, "base_ang_vel": 3, "projected_gravity": 3,
    "velocity_commands": 3, "joint_pos_rel": 12, "joint_vel_rel": 12,
    "last_action": 12,
  }
  assert sum(expected_sizes.values()) == 48
  for group in cfg.observations.values():
    assert tuple(group.terms) == tuple(expected_sizes)
  assert [sensor.name for sensor in cfg.scene.sensors] == [
    "feet_ground_contact", "limb_ground_contact"
  ]
  commands = cfg.commands["base_velocity"]
  assert commands.standing_episode_fraction == 0.20
  assert commands.transition_episode_fraction == 0.10
  assert commands.zero_yaw_probability == 0.75
  assert commands.zero_lateral_probability == 0.5
  assert cfg.sim.mujoco.timestep * cfg.decimation == pytest.approx(0.02)


def test_v4_ppo_keeps_quiet_exploration_and_all_optimizer_symmetry_settings():
  old = pochi_lp_acrl_blind_v3_ppo_runner_cfg()
  new = pochi_lp_acrl_blind_v4_ppo_runner_cfg()
  assert asdict(new) == asdict(old)
  assert new.actor.class_name == QUIET_STOP_ACTOR
  assert QuietStopMLPModel.QUIET_STD == 0.01
  assert QuietStopMLPModel.STOP_THRESHOLD == 1e-6


def test_factory_calls_do_not_mutate_v3_or_other_v4_instances():
  old = pochi_lp_acrl_blind_v3_env_cfg()
  first = pochi_lp_acrl_blind_v4_env_cfg()
  second = pochi_lp_acrl_blind_v4_env_cfg()
  first.rewards["moving_qualified_swing_occupancy"].params["target"] = 0.5
  first.rewards["contact_slip"].weight = -1
  first.sim.mujoco.timestep = 0.1
  assert "moving_qualified_swing_occupancy" not in old.rewards
  assert "moving_contact_duty_variance" in old.rewards
  assert second.rewards["moving_qualified_swing_occupancy"].params["target"] == 0.2
  assert (
    second.rewards["contact_slip"].weight == old.rewards["contact_slip"].weight == -8
  )
  assert second.sim.mujoco.timestep == old.sim.mujoco.timestep == 0.0025


def test_import_and_factory_do_not_register_a_task_or_bind_a_pending_backend():
  before = dict(registry._REGISTRY)
  importlib.reload(importlib.import_module("pochi_rl.mjlab.lp_acrl_blind_v4_env_cfg"))
  pochi_lp_acrl_blind_v4_env_cfg()
  pochi_lp_acrl_blind_v4_env_cfg(play=True)
  pochi_lp_acrl_blind_v4_ppo_runner_cfg()
  assert registry._REGISTRY.keys() == before.keys()
  assert all(registry._REGISTRY[name] is value for name, value in before.items())
