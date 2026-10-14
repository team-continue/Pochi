"""Actual CPU PPO checks scope migration without reward/observation drift."""

from copy import deepcopy

import pytest
import torch
from test_lp_acrl_flat_focus_identity import _identity
from test_lp_acrl_no_touchdown_refinement import fixture as source_fixture

from pochi_rl.lp_acrl import LPACRLCurriculum
from pochi_rl.mjlab import lp_acrl_no_touchdown_refinement as source
from pochi_rl.mjlab import lp_acrl_rough_no_touchdown_refinement as rough
from pochi_rl.mjlab.lp_acrl_blind_v15_env_cfg import pochi_lp_acrl_blind_v15_env_cfg
from pochi_rl.mjlab.lp_acrl_blind_v16_env_cfg import (
  SOURCE_LEARNING_RATE,
  pochi_lp_acrl_blind_v16_env_cfg,
)
from pochi_rl.mjlab.lp_acrl_commands import _config_identity
from pochi_rl.mjlab.lp_acrl_resume import _equal_tree


def fixture(tmp_path, monkeypatch):
  base, saved, path, parent = source_fixture(tmp_path, monkeypatch)
  runner = rough.LPACRLRoughNoTouchdownOnPolicyRunner.__new__(
    rough.LPACRLRoughNoTouchdownOnPolicyRunner
  )
  runner.__dict__.update(base.__dict__)
  for key in source.LEGACY_STATE_KEYS:
    saved["infos"].pop(key, None)
  saved["iter"] = 600
  saved["infos"][source.STATE_KEY] = {
    "version": 1, "completed_updates": 601,
    "learning_rate": SOURCE_LEARNING_RATE, "initialization": source.INITIALIZATION,
  }
  saved["infos"]["lp_acrl"]["task_space"] = source.expected_no_touchdown_task_space()
  saved["infos"]["env_state"]["common_step_counter"] = 14424
  for group in saved["optimizer_state_dict"]["param_groups"]:
    group["lr"] = SOURCE_LEARNING_RATE
  runner.alg.learning_rate = SOURCE_LEARNING_RATE
  for group in runner.alg.optimizer.param_groups:
    group["lr"] = SOURCE_LEARNING_RATE
  parent["task"] = rough.SOURCE_TASK
  cfg = pochi_lp_acrl_blind_v16_env_cfg()
  current = _identity(cfg)
  term = runner.command_term
  term._env.cfg = runner.env.unwrapped.cfg = cfg
  term.curriculum = LPACRLCurriculum(600)
  term.checkpoint_identity = lambda: deepcopy(current)
  term.curriculum_state_dict = lambda: {
    "task_space": deepcopy(current), "training_terrain_seed": 42,
    "curriculum": term.curriculum.state_dict(),
  }
  term.load_curriculum_state_dict = lambda state: term.curriculum.load_state_dict(
    state["curriculum"]
  )
  torch.save(saved, path)
  return runner, saved, path, parent


def test_actual_scope_preserves_rewards_and_blind_contract():
  flat, cfg = pochi_lp_acrl_blind_v15_env_cfg(), pochi_lp_acrl_blind_v16_env_cfg()
  assert _identity(cfg) == rough.expected_rough_task_space()
  assert _config_identity(flat.rewards) == _config_identity(cfg.rewards)
  assert len(cfg.rewards) == 22 and "balanced_touchdown" not in cfg.rewards
  assert all("height_scan" not in group.terms for group in cfg.observations.values())
  assert all(sensor.name != "terrain_scan" for sensor in cfg.scene.sensors)
  assert cfg.commands["base_velocity"].rough
  assert cfg.commands["base_velocity"].ranges.lin_vel_y == (-.1, .1)
  assert cfg.commands["base_velocity"].ranges.ang_vel_z == (-1.2, 1.2)


def test_transfer_exact_models_fresh_adam600lp_and_save_resume(tmp_path, monkeypatch):
  runner, saved, _, parent = fixture(tmp_path, monkeypatch)
  original = deepcopy(saved)
  result = rough.initialize_rough_no_touchdown_refinement(runner, saved, parent)
  assert _equal_tree(saved, original)
  assert _equal_tree(saved["actor_state_dict"], runner.alg._raw_actor.state_dict())
  assert _equal_tree(saved["critic_state_dict"], runner.alg._raw_critic.state_dict())
  assert saved["optimizer_state_dict"]["state"] and not runner.alg.optimizer.state
  assert result["initial_lp_num_tasks"] == 600
  assert runner.completed_updates() == 0
  initial = tmp_path / "initial.pt"
  runner.save(str(initial))
  state = torch.load(initial, weights_only=False)
  assert rough.validate_rough_no_touchdown_checkpoint(state) == 0
  assert all(key not in state["infos"] for key in rough.LEGACY_STATE_KEYS)
  runner.learn(2)
  path = tmp_path / "model_1.pt"
  state = torch.load(path, weights_only=False)
  assert rough.validate_rough_no_touchdown_checkpoint(state) == 2
  target, _, _, _ = fixture(tmp_path / "resume", monkeypatch)
  target.load(str(path))
  assert target.completed_updates() == 2
  assert _equal_tree(state["optimizer_state_dict"], target.alg.optimizer.state_dict())
  assert _equal_tree(
    state["infos"]["lp_acrl"], target.command_term.curriculum_state_dict()
  )
  target.learn(1)
  assert target.completed_updates() == 3


@pytest.mark.parametrize("field", ("source_reward", "source_std", "source_count",
                                  "parent_seed", "target_seed", "target_tasks"))
def test_rejects_unapproved_changes_before_model_mutation(tmp_path, monkeypatch, field):
  runner, saved, _, parent = fixture(tmp_path, monkeypatch)
  before = deepcopy(runner.alg._raw_actor.state_dict())
  if field == "source_reward":
    saved["infos"]["lp_acrl"]["task_space"]["environment"]["rewards"].clear()
  elif field == "source_std":
    saved["actor_state_dict"]["distribution.std_param"].fill_(.03)
  elif field == "source_count":
    saved["infos"][source.STATE_KEY]["completed_updates"] = 600
  elif field == "parent_seed":
    parent["seed"] = 43
  elif field == "target_seed":
    runner.env.unwrapped.cfg.scene.terrain.terrain_generator.seed = 43
  else:
    runner.command_term.curriculum = LPACRLCurriculum(599)
  with pytest.raises(ValueError):
    rough.initialize_rough_no_touchdown_refinement(runner, saved, parent)
  assert _equal_tree(before, runner.alg._raw_actor.state_dict())
