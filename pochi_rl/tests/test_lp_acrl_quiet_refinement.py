"""Reward refinement cannot silently change physical or exploration semantics."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from pochi_rl.lp_acrl.curriculum import LPACRLCurriculum
from pochi_rl.mjlab.lp_acrl_quiet_refinement import initialize_quiet_refinement
from pochi_rl.mjlab.lp_acrl_quiet_stop import QuietStopMLPModel


def fixture():
  identity = {
    "command_protocol": {"standing_episode_fraction": .2},
    "environment": {
      "physics": {"timestep": .0025, "solver": "cg"},
      "observations": {"actor": ("velocity", "joints")},
      "actions": {"scale": .25}, "rewards": {"new_contact_duty": -8.},
    },
  }
  symmetry = {
    "use_data_augmentation": True, "use_mirror_loss": True,
    "mirror_loss_coeff": 1.0,
  }
  obs = TensorDict({"actor": torch.zeros(2, 48)}, batch_size=[2])
  actor = QuietStopMLPModel(
    obs, {"actor": ["actor"]}, "actor", 12, hidden_dims=(16, 8),
    obs_normalization=True,
    distribution_cfg={"class_name": "GaussianDistribution", "init_std": .17},
  )
  critic = MLPModel(
    obs, {"critic": ["actor"]}, "critic", 1, hidden_dims=(16, 8),
    obs_normalization=True,
  )
  optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()])
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
    exploration_identity=actor.exploration_identity,
  )
  saved = {
    "iter": 100,
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
      "lp_acrl_training_exploration": actor.exploration_identity(),
      "lp_acrl": {
        "task_space": deepcopy(identity), "training_terrain_seed": 42,
        "curriculum": {"invalid_if_loaded": True},
      },
    },
  }
  saved["infos"]["lp_acrl"]["task_space"]["environment"]["rewards"] = {
    "old_contact_slip": -.15,
  }
  return runner, saved, {"task": "Pochi-LP-ACRL-Blind-v2"}


@pytest.mark.parametrize("version", [2, 3])
def test_quiet_refinement_loads_weights_normalizers_but_fresh_training(version):
  runner, saved, parent = fixture()
  parent["task"] = f"Pochi-LP-ACRL-Blind-v{version}"
  result = initialize_quiet_refinement(runner, saved, parent)
  assert result["initial_iteration"] == result["initial_lp_stage"] == 0
  assert result["initial_lp_steps"] == result["initial_lp_episode_count"] == 0
  assert result["initial_optimizer_state_entries"] == 0
  assert result["source_iteration"] == 100
  assert result["initial_exploration_identity"] == runner.exploration_identity()
  for model, value in ((runner.alg._raw_actor, .7), (runner.alg._raw_critic, .3)):
    for tensor in model.state_dict().values():
      torch.testing.assert_close(tensor, torch.full_like(tensor, value))


@pytest.mark.parametrize("field", ["physics", "observations", "actions"])
def test_nonreward_environment_changes_are_rejected(field):
  runner, saved, parent = fixture()
  saved["infos"]["lp_acrl"]["task_space"]["environment"][field] = "different"
  with pytest.raises(ValueError, match="outside rewards"):
    initialize_quiet_refinement(runner, saved, parent)


def test_command_protocol_change_is_rejected():
  runner, saved, parent = fixture()
  saved["infos"]["lp_acrl"]["task_space"]["command_protocol"] = {"changed": True}
  with pytest.raises(ValueError, match="outside rewards"):
    initialize_quiet_refinement(runner, saved, parent)


@pytest.mark.parametrize("field,value", [
  ("standing_action_std", .03), ("standing_max_abs_command", .001),
  ("command_columns", (8, 9, 10)), ("moving_std_range", (.01, 1e6)),
  ("actor_class", "other:Actor"),
])
def test_any_exploration_difference_is_rejected(field, value):
  runner, saved, parent = fixture()
  saved["infos"]["lp_acrl_training_exploration"][field] = value
  with pytest.raises(ValueError, match="exploration"):
    initialize_quiet_refinement(runner, saved, parent)


def test_missing_exploration_identity_is_rejected():
  runner, saved, parent = fixture()
  saved["infos"].pop("lp_acrl_training_exploration")
  with pytest.raises(ValueError, match="exploration"):
    initialize_quiet_refinement(runner, saved, parent)


@pytest.mark.parametrize("task", [
  "Pochi-LP-ACRL-Blind-v0", "Pochi-LP-ACRL-Blind-v1", "Pochi-LP-ACRL-Rough-v0",
])
def test_other_source_families_are_rejected(task):
  runner, saved, parent = fixture()
  parent["task"] = task
  with pytest.raises(ValueError, match="Blind-v2/v3"):
    initialize_quiet_refinement(runner, saved, parent)


@pytest.mark.parametrize("dirty", ["iteration", "lp", "optimizer"])
def test_nonfresh_training_state_is_rejected(dirty):
  runner, saved, parent = fixture()
  if dirty == "iteration":
    runner.current_learning_iteration = 1
  elif dirty == "lp":
    runner.command_term.curriculum.previous_valid[0] = True
  else:
    runner.alg.optimizer.state[next(runner.alg._raw_actor.parameters())] = {}
  with pytest.raises(ValueError, match="requires"):
    initialize_quiet_refinement(runner, saved, parent)


@pytest.mark.parametrize("corruption", ["shape", "dtype", "nan", "keys"])
def test_corrupt_critic_is_rejected_before_actor_mutation(corruption):
  runner, saved, parent = fixture()
  before = deepcopy(runner.alg._raw_actor.state_dict())
  source = saved["critic_state_dict"]
  if corruption == "shape":
    source["mlp.0.weight"] = torch.zeros(1, 48)
  elif corruption == "dtype":
    source["mlp.0.weight"] = source["mlp.0.weight"].double()
  elif corruption == "nan":
    source["obs_normalizer._mean"][0, 0] = float("nan")
  else:
    source.pop("mlp.0.bias")
  with pytest.raises(ValueError):
    initialize_quiet_refinement(runner, saved, parent)
  for name, value in runner.alg._raw_actor.state_dict().items():
    torch.testing.assert_close(value, before[name])


@pytest.mark.parametrize("changed", ["terrain", "symmetry", "format"])
def test_training_contract_mismatch_rejected(changed):
  runner, saved, parent = fixture()
  if changed == "terrain":
    saved["infos"]["lp_acrl"]["training_terrain_seed"] = 123
  elif changed == "symmetry":
    saved["infos"]["lp_acrl_training_symmetry"]["mirror_loss_coeff"] = .5
  else:
    saved["infos"]["lp_acrl_checkpoint_version"] = 99
  with pytest.raises(ValueError):
    initialize_quiet_refinement(runner, saved, parent)
