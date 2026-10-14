"""Actual CPU PPO verifies fresh model transfer and independent local resumes."""

from copy import deepcopy

import pytest
import torch
from rsl_rl.algorithms import PPO
from test_lp_acrl_flat_focus_identity import _identity
from test_lp_acrl_wider_exploration import fixture as wider_fixture
from test_lp_acrl_wider_exploration import snapshot

from pochi_rl.lp_acrl import LPACRLCurriculum
from pochi_rl.mjlab import lp_acrl_flat_focus_refinement as flat
from pochi_rl.mjlab import lp_acrl_wider_exploration as wider
from pochi_rl.mjlab.lp_acrl_blind_v11_env_cfg import pochi_lp_acrl_blind_v11_env_cfg
from pochi_rl.mjlab.lp_acrl_blind_v12_env_cfg import (
  SOURCE_LEARNING_RATE,
)
from pochi_rl.mjlab.lp_acrl_blind_v13_env_cfg import pochi_lp_acrl_blind_v13_env_cfg
from pochi_rl.mjlab.lp_acrl_commands import _config_identity
from pochi_rl.mjlab.lp_acrl_flat_focus_identity import canonical_rough_task_space
from pochi_rl.mjlab.lp_acrl_resume import _equal_tree
from pochi_rl.mjlab.lp_acrl_stride_center_refinement import (
  STATE_KEY as SOURCE_STATE_KEY,
)


def fixture(tmp_path, monkeypatch):
  base, saved, path, parent = wider_fixture(tmp_path, monkeypatch)
  runner = flat.LPACRLFlatFocusOnPolicyRunner.__new__(
    flat.LPACRLFlatFocusOnPolicyRunner
  )
  runner.__dict__.update(base.__dict__)
  cfg = pochi_lp_acrl_blind_v13_env_cfg()
  current = _identity(cfg)
  term = runner.command_term
  term._env.cfg = cfg
  runner.env.unwrapped.cfg = cfg
  term.checkpoint_identity = lambda: current
  term.curriculum = LPACRLCurriculum(5)
  term.curriculum_state_dict = lambda: {
    "task_space": deepcopy(current), "training_terrain_seed": None,
    "curriculum": term.curriculum.state_dict(),
  }
  term.load_curriculum_state_dict = lambda state: term.curriculum.load_state_dict(
    state["curriculum"]
  )
  source_identity = canonical_rough_task_space()
  source_identity["environment"]["rewards"] = _config_identity(
    pochi_lp_acrl_blind_v11_env_cfg().rewards
  )
  saved["infos"]["lp_acrl"]["task_space"] = source_identity
  saved["iter"] = 300
  saved["actor_state_dict"][wider.STD_PARAMETER].fill_(0.15)
  saved["infos"]["lp_acrl_training_exploration"] = flat.expected_exploration_identity()
  saved["infos"].pop(wider.STATE_KEY, None)
  saved["infos"][SOURCE_STATE_KEY] = {
    "version": 1,
    "completed_updates": 301,
    "initialization": flat.INITIALIZATION,
    "learning_rate": SOURCE_LEARNING_RATE,
  }
  saved["infos"]["env_state"]["common_step_counter"] = 7224
  for group in saved["optimizer_state_dict"]["param_groups"]:
    group["lr"] = SOURCE_LEARNING_RATE
  runner.alg.learning_rate = SOURCE_LEARNING_RATE
  for group in runner.alg.optimizer.param_groups:
    group["lr"] = SOURCE_LEARNING_RATE
  parent["task"] = flat.SOURCE_TASK
  torch.save(saved, path)
  return runner, saved, path, parent


def initialize(runner, saved, parent):
  return flat.initialize_flat_focus_refinement(runner, saved, parent)


def initial_checkpoint(tmp_path, monkeypatch):
  runner, saved, _, parent = fixture(tmp_path, monkeypatch)
  initialize(runner, saved, parent)
  path = tmp_path / "initial_state.pt"
  runner.save(str(path), {"caller": "kept"})
  return runner, torch.load(path, weights_only=False), path


def test_model_only_transfer_is_bitwise_while_adam_lp_and_local_progress_are_fresh(
  tmp_path,
  monkeypatch,
):
  runner, saved, _, parent = fixture(tmp_path, monkeypatch)
  source = deepcopy(saved)
  fresh_lp = deepcopy(runner.command_term.curriculum_state_dict())
  result = initialize(runner, saved, parent)
  for key, model in (
    ("actor_state_dict", runner.alg._raw_actor),
    ("critic_state_dict", runner.alg._raw_critic),
  ):
    assert _equal_tree(saved[key], model.state_dict())
  assert _equal_tree(source, saved)
  assert _equal_tree(fresh_lp, runner.command_term.curriculum_state_dict())
  assert saved["optimizer_state_dict"]["state"] and not runner.alg.optimizer.state
  assert runner.completed_updates() == runner.current_learning_iteration == 0
  assert runner.env.unwrapped.common_step_counter == 0
  assert runner.alg.learning_rate == SOURCE_LEARNING_RATE
  assert runner.env.seed_calls == [42] and runner.env.reset_calls == [True]
  assert result["source_completed_updates"] == 301
  assert result["initial_completed_updates"] == 0
  assert result["initial_lp_num_tasks"] == 5
  assert runner.command_term.curriculum.num_tasks == 5
  assert "training_terrain_seed" in result["allowed_environment_identity_difference"]
  assert saved["infos"]["lp_acrl"]["curriculum"]["probabilities"].numel() == 600


def test_actual_ppo_initial_save_train_resume_and_repeated_learn_have_local_counts(
  tmp_path,
  monkeypatch,
):
  runner, initial, _ = initial_checkpoint(tmp_path, monkeypatch)
  assert initial["iter"] == -1
  assert flat.validate_flat_focus_checkpoint(initial) == 0
  assert all(key not in initial["infos"] for key in flat.LEGACY_STATE_KEYS)
  assert initial["infos"]["caller"] == "kept"
  runner.learn(2)
  assert runner.logged_iterations == [0, 1]
  assert runner.completed_updates() == runner.current_learning_iteration == 2
  saved = torch.load(tmp_path / "model_1.pt", weights_only=False)
  assert flat.validate_flat_focus_checkpoint(saved) == 2
  assert saved["infos"]["env_state"]["common_step_counter"] == 8
  target, _, _, _ = fixture(tmp_path / "resume", monkeypatch)
  target.load(str(tmp_path / "model_1.pt"))
  assert target.current_learning_iteration == 2
  assert _equal_tree(saved["optimizer_state_dict"], target.alg.optimizer.state_dict())
  assert _equal_tree(
    saved["infos"]["lp_acrl"], target.command_term.curriculum_state_dict()
  )
  assert target.alg.learning_rate == saved["infos"][flat.STATE_KEY]["learning_rate"]
  target.learn(1)
  target.learn(1)
  assert target.logged_iterations == [2, 3] and target.completed_updates() == 4
  final = torch.load(tmp_path / "resume/model_3.pt", weights_only=False)
  assert flat.validate_flat_focus_checkpoint(final) == 4
  assert final["infos"]["lp_acrl"]["curriculum"]["steps_in_stage"] == 16


def test_initial_checkpoint_can_resume_before_first_update(tmp_path, monkeypatch):
  _, _, path = initial_checkpoint(tmp_path, monkeypatch)
  target, _, _, _ = fixture(tmp_path / "resume", monkeypatch)
  target.load(str(path))
  assert target.current_learning_iteration == 0 and not target.alg.optimizer.state
  target.learn(1)
  saved = torch.load(tmp_path / "resume/model_0.pt", weights_only=False)
  assert saved["iter"] == 0 and flat.validate_flat_focus_checkpoint(saved) == 1


def test_actor_only_load_does_not_restore_training_state_and_cannot_train_or_save(
  tmp_path,
  monkeypatch,
):
  _, saved, path = initial_checkpoint(tmp_path, monkeypatch)
  target, _, _, _ = fixture(tmp_path / "target", monkeypatch)
  before = deepcopy(
    (
      target.alg._raw_critic.state_dict(),
      target.alg.optimizer.state_dict(),
      target.command_term.curriculum_state_dict(),
    )
  )
  target.load(str(path), {"actor": True})
  assert _equal_tree(saved["actor_state_dict"], target.alg._raw_actor.state_dict())
  assert _equal_tree(
    before,
    (
      target.alg._raw_critic.state_dict(),
      target.alg.optimizer.state_dict(),
      target.command_term.curriculum_state_dict(),
    ),
  )
  with pytest.raises(ValueError, match="cannot save/train"):
    target.learn(1)
  with pytest.raises(ValueError, match="cannot save/train"):
    target.save(str(tmp_path / "invalid.pt"))


@pytest.mark.parametrize(
  "fault",
  [
    "task",
    "iter",
    "seed",
    "physics",
    "source_reward",
    "target_reward",
    "stop_weight",
    "stop_params",
    "stride_weight",
    "yaw_delta",
    "preserved_stride",
    "terrain",
    "backend",
    "lr",
    "optimizer_lr",
    "std",
    "source_progress",
    "critic_shape",
    "normalizer_nan",
    "environment_counter",
    "lp_count",
    "target_terrain_seed",
    "source_yaw_progress",
  ],
)
def test_bad_refinement_rejected_before_model_mutation(tmp_path, monkeypatch, fault):
  runner, saved, _, parent = fixture(tmp_path, monkeypatch)
  current = runner.command_term.checkpoint_identity()
  rewards = current["environment"]["rewards"]
  if fault == "task":
    parent["task"] = "Pochi-LP-ACRL-Blind-v8"
  elif fault == "iter":
    saved["iter"] = 300.0
  elif fault == "seed":
    parent["seed"] = 123
  elif fault == "physics":
    current["environment"]["physics"] = 0.005
  elif fault == "source_reward":
    saved["infos"]["lp_acrl"]["task_space"]["environment"]["rewards"][
      "moving_fore_aft_stride_center"
    ]["weight"] = -3
  elif fault == "target_reward":
    rewards["extra"] = {}
  elif fault == "stop_weight":
    rewards["standing_joint_velocity"]["weight"] = -0.5
  elif fault == "stop_params":
    rewards["standing_joint_velocity"]["params"]["std"] = 0.2
  elif fault == "stride_weight":
    rewards[flat.NEW_REWARD]["weight"] = -40
  elif fault == "yaw_delta":
    rewards[flat.NEW_REWARD]["params"]["delta"] = 0.5
  elif fault == "preserved_stride":
    rewards["moving_fore_aft_stride_center"]["weight"] = -40
  elif fault == "terrain":
    saved["infos"]["lp_acrl"]["training_terrain_seed"] = 123
  elif fault == "lp_count":
    runner.command_term.curriculum = LPACRLCurriculum(600)
  elif fault == "target_terrain_seed":
    from types import SimpleNamespace
    terrain = runner.command_term._env.cfg.scene.terrain
    terrain.terrain_generator = SimpleNamespace(seed=42)
  elif fault == "source_yaw_progress":
    saved["infos"]["lp_acrl_yaw_stability_state"] = {}
  elif fault == "backend":
    parent["contact_backend"] = {}
  elif fault == "lr":
    runner.alg.learning_rate = 0.001
  elif fault == "optimizer_lr":
    runner.alg.optimizer.param_groups[0]["lr"] = 0.001
  elif fault == "std":
    saved["actor_state_dict"][wider.STD_PARAMETER].fill_(0.03)
  elif fault == "source_progress":
    saved["infos"][SOURCE_STATE_KEY]["completed_updates"] = 300
  elif fault == "critic_shape":
    saved["critic_state_dict"]["mlp.0.weight"] = torch.zeros(3, 3)
  elif fault == "normalizer_nan":
    name = next(k for k in saved["actor_state_dict"] if k.startswith("obs_normalizer."))
    saved["actor_state_dict"][name].fill_(float("nan"))
  else:
    runner.env.unwrapped.common_step_counter = 24
  before = snapshot(runner)
  with pytest.raises(ValueError):
    initialize(runner, saved, parent)
  assert _equal_tree(before, snapshot(runner))


@pytest.mark.parametrize("actor_only", [False, True])
@pytest.mark.parametrize(
  "fault",
  [
    "legacy",
    "legacy_stride",
    "legacy_top",
    "legacy_stride_top",
    "legacy_yaw",
    "legacy_yaw_top",
    "own_top",
    "missing",
    "count",
    "iter",
    "lr",
    "tag",
  ],
)
def test_portable_metadata_rejected_before_loading(
  tmp_path, monkeypatch, fault, actor_only
):
  _, saved, path = initial_checkpoint(tmp_path, monkeypatch)
  runner, _, _, _ = fixture(tmp_path / "target", monkeypatch)
  if fault == "legacy":
    saved["infos"][wider.STATE_KEY] = {}
  elif fault == "legacy_stride":
    saved["infos"][SOURCE_STATE_KEY] = {}
  elif fault == "legacy_top":
    saved[wider.STATE_KEY] = {}
  elif fault == "legacy_stride_top":
    saved[SOURCE_STATE_KEY] = {}
  elif fault == "legacy_yaw":
    saved["infos"]["lp_acrl_yaw_stability_state"] = {}
  elif fault == "legacy_yaw_top":
    saved["lp_acrl_yaw_stability_state"] = {}
  elif fault == "own_top":
    saved[flat.STATE_KEY] = deepcopy(saved["infos"][flat.STATE_KEY])
  elif fault == "missing":
    saved["infos"].pop(flat.STATE_KEY)
  elif fault == "count":
    saved["infos"][flat.STATE_KEY]["completed_updates"] = True
  elif fault == "iter":
    saved["iter"] = 0
  elif fault == "lr":
    saved["infos"][flat.STATE_KEY]["learning_rate"] = 0.001
  else:
    saved["infos"][flat.STATE_KEY]["initialization"] = "full_state_continuation"
  torch.save(saved, path)
  before = snapshot(runner)
  with pytest.raises(ValueError):
    runner.load(str(path), {"actor": True} if actor_only else None)
  assert _equal_tree(before, snapshot(runner))


def test_failed_reset_and_partial_ppo_update_poison_saving(tmp_path, monkeypatch):
  runner, saved, _, parent = fixture(tmp_path / "reset", monkeypatch)
  runner.env.reset = lambda: runner.env.episode_length_buf.fill_(1)
  with pytest.raises(ValueError, match="length zero"):
    initialize(runner, saved, parent)
  with pytest.raises(ValueError, match="cannot save/train"):
    runner.save(str(tmp_path / "failed_reset.pt"))
  runner, _, _ = initial_checkpoint(tmp_path / "update", monkeypatch)

  def fail(algorithm):
    with torch.no_grad():
      next(algorithm.actor.parameters()).add_(0.1)
    raise RuntimeError("partial update")

  monkeypatch.setattr(PPO, "update", fail)
  with pytest.raises(RuntimeError, match="partial update"):
    runner.learn(1)
  with pytest.raises(ValueError, match="cannot save/train"):
    runner.save(str(tmp_path / "failed_update.pt"))
