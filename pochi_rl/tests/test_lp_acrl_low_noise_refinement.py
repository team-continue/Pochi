"""Only declared v5 changes survive strict model-only transfer from v4."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from pochi_rl.lp_acrl.curriculum import LPACRLCurriculum
from pochi_rl.mjlab.lp_acrl_low_noise_refinement import (
  SOURCE_OBSERVATION_CORRUPTION,
  TARGET_OBSERVATION_CORRUPTION,
  initialize_low_noise_refinement,
  read_parent_observation_corruption,
)
from pochi_rl.mjlab.lp_acrl_quiet_stop import QuietStopMLPModel


def fixture():
  obs = TensorDict({"actor": torch.zeros(2, 48)}, batch_size=[2])
  actor = QuietStopMLPModel(
    obs,
    {"actor": ["actor"]},
    "actor",
    12,
    hidden_dims=(16, 8),
    obs_normalization=True,
    distribution_cfg={
      "class_name": "GaussianDistribution",
      "init_std": 0.03,
      "std_range": (0.03, 0.03),
      "learn_std": False,
    },
  )
  critic = MLPModel(
    obs,
    {"critic": ["actor"]},
    "critic",
    1,
    hidden_dims=(16, 8),
    obs_normalization=True,
  )
  optimizer = torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=1e-4)
  algorithm = SimpleNamespace(
    _raw_actor=actor,
    _raw_critic=critic,
    optimizer=optimizer,
    rnd=None,
    learning_rate=1e-4,
  )
  algorithm.load = lambda saved, cfg, strict: PPO.load(algorithm, saved, cfg, strict)
  old_rewards = {
    "track_ang_vel_z_exp": {"weight": 0.5, "params": {"std": 0.35}},
    "standing_posture": {"weight": 0.5, "params": {}},
    "standing_missing_contacts": {"weight": -1.0, "params": {}},
    "moving_qualified_swing_occupancy": {"weight": -8.0, "params": {"target": 0.2}},
  }
  new_rewards = deepcopy(old_rewards)
  new_rewards["track_ang_vel_z_exp"]["params"]["std"] = 0.15
  new_rewards["standing_posture"] = {"weight": 1.5, "params": {"std": 0.15}}
  new_rewards["standing_missing_contacts"]["weight"] = -2.0
  identity = {
    "command_protocol": {"standing_episode_fraction": 0.2},
    "environment": {
      "physics": {"timestep": 0.0025, "solver": "cg"},
      "observations": {"actor": ["joint", "command"]},
      "actions": {"scale": 0.25},
      "rewards": new_rewards,
    },
  }
  symmetry = {
    "use_data_augmentation": False,
    "use_mirror_loss": True,
    "mirror_loss_coeff": 1.0,
  }
  backend = {"patch_id": "same-patched-backend", "discrete_epa_epsilon": 1e-7}
  term = SimpleNamespace(
    _env=SimpleNamespace(
      cfg=SimpleNamespace(
        scene=SimpleNamespace(
          terrain=SimpleNamespace(terrain_generator=SimpleNamespace(seed=42))
        )
      )
    ),
    curriculum=LPACRLCurriculum(600),
    checkpoint_identity=lambda: identity,
  )
  runner = SimpleNamespace(
    command_term=term,
    current_learning_iteration=0,
    alg=algorithm,
    symmetry_identity=lambda: symmetry,
    exploration_identity=actor.exploration_identity,
    contact_backend_identity=lambda: backend,
    observation_corruption_identity=lambda: deepcopy(TARGET_OBSERVATION_CORRUPTION),
  )
  saved = {
    "iter": 100,
    "actor_state_dict": {
      k: torch.full_like(v, 0.7) for k, v in actor.state_dict().items()
    },
    "critic_state_dict": {
      k: torch.full_like(v, 0.3) for k, v in critic.state_dict().items()
    },
    "optimizer_state_dict": {"must_not_load": True},
    "infos": {
      "lp_acrl_checkpoint_version": 1,
      "lp_acrl_training_symmetry": {**symmetry, "use_data_augmentation": True},
      "lp_acrl_contact_backend": deepcopy(backend),
      "lp_acrl_training_exploration": {
        **actor.exploration_identity(),
        "moving_std_range": (0.1, 1e6),
        "moving_std_learned": True,
      },
      "lp_acrl": {"task_space": deepcopy(identity), "training_terrain_seed": 42},
    },
  }
  saved["infos"]["lp_acrl"]["task_space"]["environment"]["rewards"] = old_rewards
  parent = {
    "task": "Pochi-LP-ACRL-Blind-v4",
    "contact_backend": deepcopy(backend),
    "learning_rate": 1e-3,
  }
  return runner, saved, parent, deepcopy(SOURCE_OBSERVATION_CORRUPTION)


def test_actual_rsl_load_preserves_mean_critic_normalizers_and_only_resets_std():
  runner, saved, parent, noise = fixture()
  source_before = deepcopy(saved)
  report = initialize_low_noise_refinement(runner, saved, parent, noise)
  assert report["all_other_actor_and_critic_states_bitwise_preserved"]
  assert report["reset_actor_state"] == {"distribution.std_param": 0.03}
  assert not runner.alg.optimizer.state and runner.current_learning_iteration == 0
  assert runner.command_term.curriculum.stage_index == 0
  for key, model in (
    ("actor_state_dict", runner.alg._raw_actor),
    ("critic_state_dict", runner.alg._raw_critic),
  ):
    for name, value in model.state_dict().items():
      expected = (
        torch.full_like(value, 0.03)
        if name == "distribution.std_param"
        else saved[key][name]
      )
      torch.testing.assert_close(value, expected, rtol=0, atol=0)
      assert torch.equal(saved[key][name], source_before[key][name])
  assert not runner.alg._raw_actor.distribution.std_param.requires_grad


@pytest.mark.parametrize("field", ["physics", "observations", "actions"])
def test_nonreward_environment_differences_rejected_before_mutation(field):
  runner, saved, parent, noise = fixture()
  saved["infos"]["lp_acrl"]["task_space"]["environment"][field] = "changed"
  before = deepcopy(runner.alg._raw_actor.state_dict())
  with pytest.raises(ValueError, match="outside rewards"):
    initialize_low_noise_refinement(runner, saved, parent, noise)
  assert all(
    torch.equal(v, before[k]) for k, v in runner.alg._raw_actor.state_dict().items()
  )


@pytest.mark.parametrize(
  "change",
  ["qualified_weight", "qualified_target", "new_term", "wrong_yaw", "wrong_stop"],
)
def test_other_reward_changes_are_rejected(change):
  runner, saved, parent, noise = fixture()
  rewards = runner.command_term.checkpoint_identity()["environment"]["rewards"]
  if change == "qualified_weight":
    rewards["moving_qualified_swing_occupancy"]["weight"] = -9
  elif change == "qualified_target":
    rewards["moving_qualified_swing_occupancy"]["params"]["target"] = 0.3
  elif change == "new_term":
    rewards["another"] = {}
  elif change == "wrong_yaw":
    rewards["track_ang_vel_z_exp"]["params"]["std"] = 0.2
  else:
    rewards["standing_posture"]["weight"] = 2
  with pytest.raises(ValueError, match="three approved"):
    initialize_low_noise_refinement(runner, saved, parent, noise)


@pytest.mark.parametrize(
  "change",
  [
    "source_backend",
    "manifest_backend",
    "source_exploration",
    "target_exploration",
    "source_noise",
    "target_noise",
    "symmetry",
    "source_augmentation",
    "target_augmentation",
    "target_mirror_loss",
    "source_initial_lr",
    "target_initial_lr",
    "optimizer_initial_lr",
    "terrain",
    "task",
  ],
)
def test_contract_changes_are_rejected(change):
  runner, saved, parent, noise = fixture()
  if change == "source_backend":
    saved["infos"]["lp_acrl_contact_backend"] = {}
  elif change == "manifest_backend":
    parent["contact_backend"] = {}
  elif change == "source_exploration":
    saved["infos"]["lp_acrl_training_exploration"]["standing_action_std"] = 0.02
  elif change == "target_exploration":
    runner.exploration_identity = lambda: {}
  elif change == "source_noise":
    noise["groups"]["actor"] = False
  elif change == "target_noise":
    runner.observation_corruption_identity = lambda: SOURCE_OBSERVATION_CORRUPTION
  elif change == "symmetry":
    saved["infos"]["lp_acrl_training_symmetry"]["mirror_loss_coeff"] = 0.5
  elif change == "source_augmentation":
    saved["infos"]["lp_acrl_training_symmetry"]["use_data_augmentation"] = False
  elif change == "target_augmentation":
    runner.symmetry_identity()["use_data_augmentation"] = True
  elif change == "target_mirror_loss":
    runner.symmetry_identity()["use_mirror_loss"] = False
  elif change == "source_initial_lr":
    parent["learning_rate"] = 1e-4
  elif change == "target_initial_lr":
    runner.alg.learning_rate = 1e-3
  elif change == "optimizer_initial_lr":
    runner.alg.optimizer.param_groups[0]["lr"] = 1e-3
  elif change == "terrain":
    saved["infos"]["lp_acrl"]["training_terrain_seed"] = 123
  else:
    parent["task"] = "Pochi-LP-ACRL-Blind-v3"
  with pytest.raises(ValueError):
    initialize_low_noise_refinement(runner, saved, parent, noise)


@pytest.mark.parametrize("dirty", ["optimizer", "iteration", "lp"])
def test_nonfresh_training_state_rejected(dirty):
  runner, saved, parent, noise = fixture()
  if dirty == "optimizer":
    runner.alg.optimizer.state[next(runner.alg._raw_actor.parameters())] = {}
  elif dirty == "iteration":
    runner.current_learning_iteration = 1
  else:
    runner.command_term.curriculum.previous_valid[0] = True
  with pytest.raises(ValueError, match="requires"):
    initialize_low_noise_refinement(runner, saved, parent, noise)


@pytest.mark.parametrize("corruption", ["nan", "dtype", "shape", "keys"])
def test_invalid_critic_is_rejected_before_any_actor_transfer(corruption):
  runner, saved, parent, noise = fixture()
  before = deepcopy(runner.alg._raw_actor.state_dict())
  if corruption == "nan":
    saved["critic_state_dict"]["mlp.0.weight"][0, 0] = float("nan")
  elif corruption == "dtype":
    saved["critic_state_dict"]["mlp.0.weight"] = saved["critic_state_dict"][
      "mlp.0.weight"
    ].double()
  elif corruption == "shape":
    saved["critic_state_dict"]["mlp.0.weight"] = torch.zeros(1, 48)
  else:
    saved["critic_state_dict"].pop("mlp.0.weight")
  with pytest.raises(ValueError, match="Model|Invalid"):
    initialize_low_noise_refinement(runner, saved, parent, noise)
  assert all(
    torch.equal(v, before[k]) for k, v in runner.alg._raw_actor.state_dict().items()
  )


def test_yaml_evidence_reads_booleans_without_executing_python_tags(tmp_path, capsys):
  path = tmp_path / "env.yaml"
  path.write_text("""unrelated: !!python/object/apply:builtins.print [must-not-execute]
observations:
  actor:
    enable_corruption: true
  critic:
    enable_corruption: false
""")
  proof = read_parent_observation_corruption(path)
  assert proof["identity"] == SOURCE_OBSERVATION_CORRUPTION
  assert len(proof["sha256"]) == 64 and proof["path"] == str(path)
  assert capsys.readouterr().out == ""


@pytest.mark.parametrize(
  "actor,critic",
  [("false", "false"), ("true", "true"), ('"true"', "false"), ("null", "false")],
)
def test_yaml_missing_or_wrong_source_corruption_is_rejected(tmp_path, actor, critic):
  path = tmp_path / "env.yaml"
  path.write_text(
    f"observations:\n  actor:\n    enable_corruption: {actor}\n"
    f"  critic:\n    enable_corruption: {critic}\n"
  )
  with pytest.raises(ValueError):
    read_parent_observation_corruption(path)


def test_yaml_duplicate_corruption_key_is_rejected(tmp_path):
  path = tmp_path / "env.yaml"
  path.write_text(
    "observations:\n  actor:\n    enable_corruption: true\n"
    "    enable_corruption: false\n  critic:\n    enable_corruption: false\n"
  )
  with pytest.raises(ValueError, match="exactly one"):
    read_parent_observation_corruption(path)
