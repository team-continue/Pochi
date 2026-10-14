"""Refinement must preserve weights while restarting optimization and LP."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO

from pochi_rl.lp_acrl.curriculum import LPACRLCurriculum
from pochi_rl.mjlab.lp_acrl_refinement import (
  initialize_refinement,
  output_budget_bytes,
  validate_refinement_identity,
  without_rewards,
)


def _fixture():
  identity = {
    "command_protocol": {"standing_episode_fraction": .2},
    "environment": {
      "physics": {"timestep": .0025, "solver": "cg"},
      "observations": {"actor": ("velocity", "joints")},
      "actions": {"scale": .25},
      "rewards": {"tracking": 1.0},
    },
  }
  symmetry = {
    "use_data_augmentation": True,
    "use_mirror_loss": True,
    "mirror_loss_coeff": 1.0,
  }
  actor = torch.nn.Linear(3, 2)
  actor.register_buffer("normalizer_mean", torch.zeros(3))
  critic = torch.nn.Linear(3, 1)
  critic.register_buffer("normalizer_mean", torch.zeros(3))
  optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()])
  # Use the installed RSL implementation to check its actual load contract.
  algorithm = SimpleNamespace(
    _raw_actor=actor, _raw_critic=critic, optimizer=optimizer, rnd=None,
  )
  algorithm.load = lambda saved, cfg, strict: PPO.load(
    algorithm, saved, cfg, strict
  )
  term = SimpleNamespace(
    _env=SimpleNamespace(cfg=SimpleNamespace(scene=SimpleNamespace(
      terrain=SimpleNamespace(terrain_generator=SimpleNamespace(seed=42))
    ))),
    curriculum=LPACRLCurriculum(600), checkpoint_identity=lambda: identity,
  )
  runner = SimpleNamespace(
    command_term=term, current_learning_iteration=0, alg=algorithm,
    symmetry_identity=lambda: symmetry,
  )
  saved = {
    "iter": 200,
    "actor_state_dict": {
      name: torch.full_like(value, .7) for name, value in actor.state_dict().items()
    },
    "critic_state_dict": {
      name: torch.full_like(value, .3) for name, value in critic.state_dict().items()
    },
    "optimizer_state_dict": {"invalid_if_loaded": True},
    "infos": {
      "lp_acrl_checkpoint_version": 1,
      "lp_acrl_training_symmetry": deepcopy(symmetry),
      "lp_acrl": {
        "task_space": deepcopy(identity), "training_terrain_seed": 42,
        "curriculum": {"invalid_if_loaded": True},
      },
    },
  }
  saved["infos"]["lp_acrl"]["task_space"]["environment"]["rewards"] = {
    "old_tracking": 2.0,
  }
  return runner, saved, {"task": "Pochi-LP-ACRL-Blind-v0"}


def test_model_only_refinement_loads_both_normalizers_and_restarts_training():
  runner, saved, parent = _fixture()
  result = initialize_refinement(runner, saved, parent)
  assert result["source_iteration"] == 200
  assert result["initial_iteration"] == result["initial_lp_stage"] == 0
  assert result["initial_lp_episode_count"] == 0
  assert runner.alg.optimizer.state == {}
  assert runner.command_term.curriculum.previous_valid.sum() == 0
  for model, value in ((runner.alg._raw_actor, .7), (runner.alg._raw_critic, .3)):
    for tensor in model.state_dict().values():
      torch.testing.assert_close(tensor, torch.full_like(tensor, value))


@pytest.mark.parametrize("field", ["physics", "observations", "actions"])
def test_refinement_rejects_every_nonreward_environment_change(field):
  runner, saved, parent = _fixture()
  saved["infos"]["lp_acrl"]["task_space"]["environment"][field] = "different"
  with pytest.raises(ValueError, match="outside rewards"):
    initialize_refinement(runner, saved, parent)


@pytest.mark.parametrize("changed", ["legacy", "terrain", "symmetry", "format"])
def test_refinement_rejects_unrelated_family_geometry_or_training_contract(changed):
  runner, saved, parent = _fixture()
  if changed == "legacy":
    parent["task"] = "Pochi-LP-ACRL-Rough-v0"
  elif changed == "terrain":
    saved["infos"]["lp_acrl"]["training_terrain_seed"] = 123
  elif changed == "symmetry":
    saved["infos"]["lp_acrl_training_symmetry"]["mirror_loss_coeff"] = .5
  else:
    saved["infos"]["lp_acrl_checkpoint_version"] = 99
  with pytest.raises(ValueError):
    initialize_refinement(runner, saved, parent)


@pytest.mark.parametrize("dirty", ["iteration", "lp", "optimizer"])
def test_refinement_rejects_reusing_nonfresh_training_state(dirty):
  runner, saved, parent = _fixture()
  if dirty == "iteration":
    runner.current_learning_iteration = 1
  elif dirty == "lp":
    runner.command_term.curriculum.previous_valid[0] = True
  else:
    runner.alg.optimizer.state[next(runner.alg._raw_actor.parameters())] = {}
  with pytest.raises(ValueError, match="requires"):
    initialize_refinement(runner, saved, parent)


@pytest.mark.parametrize("corruption", ["shape", "nan", "keys"])
def test_invalid_critic_is_rejected_before_actor_is_mutated(corruption):
  runner, saved, parent = _fixture()
  original = deepcopy(runner.alg._raw_actor.state_dict())
  if corruption == "shape":
    saved["critic_state_dict"]["weight"] = torch.zeros(2, 3)
  elif corruption == "nan":
    saved["critic_state_dict"]["weight"][0, 0] = float("nan")
  else:
    saved["critic_state_dict"].pop("bias")
  with pytest.raises(ValueError):
    initialize_refinement(runner, saved, parent)
  for name, value in runner.alg._raw_actor.state_dict().items():
    torch.testing.assert_close(value, original[name])


def test_terrain_seed_cannot_be_silently_absent():
  runner, saved, parent = _fixture()
  del saved["infos"]["lp_acrl"]["training_terrain_seed"]
  with pytest.raises(ValueError, match="terrain"):
    validate_refinement_identity(
      saved, parent, runner.command_term.checkpoint_identity(),
      runner.symmetry_identity(), 42,
    )


def test_disk_budget_counts_final_checkpoint_once_and_guard_failure_model():
  reserve = 40 * 1024**2
  assert output_budget_bytes(100, 3, 100, 30) == reserve + 250 + 30
  assert output_budget_bytes(100, 1, 100, 30) == reserve + 125 + 30
  assert output_budget_bytes(100, 1000, 100, 30) == reserve + 1375 + 30
  assert output_budget_bytes(100, 1, 100, 0, guard_model_bytes=1000) == (
    reserve + 125 + 2000
  )


def test_registered_blind_versions_differ_only_in_reward_identity():
  from pochi_rl.lp_acrl import ROUGH_TASK_CATALOG
  from pochi_rl.mjlab.lp_acrl_blind_env_cfg import pochi_lp_acrl_blind_env_cfg
  from pochi_rl.mjlab.lp_acrl_blind_v1_env_cfg import pochi_lp_acrl_blind_v1_env_cfg
  from pochi_rl.mjlab.lp_acrl_commands import LPACRLVelocityCommand

  identities = []
  for factory in (pochi_lp_acrl_blind_env_cfg, pochi_lp_acrl_blind_v1_env_cfg):
    cfg = factory()
    term = LPACRLVelocityCommand.__new__(LPACRLVelocityCommand)
    term._env = SimpleNamespace(cfg=cfg)
    term.cfg = cfg.commands["base_velocity"]
    term.catalog = ROUGH_TASK_CATALOG
    identities.append(term.checkpoint_identity())
  assert identities[0] != identities[1]
  assert without_rewards(identities[0]) == without_rewards(identities[1])
