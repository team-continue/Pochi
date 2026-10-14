"""V8 transfers actual RSL networks bitwise and changes only one reward."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from pochi_rl.lp_acrl.curriculum import LPACRLCurriculum
from pochi_rl.mjlab.lp_acrl_blind_v8_env_cfg import common_hip_reward_cfg
from pochi_rl.mjlab.lp_acrl_commands import _config_identity
from pochi_rl.mjlab.lp_acrl_common_hip_refinement import (
  initialize_common_hip_refinement,
)
from pochi_rl.mjlab.lp_acrl_low_noise import (
  CORRUPTION_IDENTITY,
  expected_symmetry_identity,
)
from pochi_rl.mjlab.lp_acrl_quiet_stop import QuietStopMLPModel


def _fixture():
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
  alg = SimpleNamespace(
    _raw_actor=actor,
    _raw_critic=critic,
    rnd=None,
    learning_rate=1e-4,
    optimizer=torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=1e-4),
  )
  alg.load = lambda saved, cfg, strict: PPO.load(alg, saved, cfg, strict)
  source_identity = {
    "command_protocol": {"standing_episode_fraction": 0.2},
    "environment": {
      "physics": {"timestep": 0.0025, "solver": "cg"},
      "observations": {"actor": "48D"},
      "actions": {"scale": 0.25},
      "rewards": {"qualified_swing": {"weight": -8}},
    },
  }
  identity = deepcopy(source_identity)
  identity["environment"]["rewards"]["moving_common_hip_pitch_deviation"] = (
    _config_identity(common_hip_reward_cfg())
  )
  backend = {"patch_id": "identical-contact-backend", "discrete_epa_epsilon": 1e-7}
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
    alg=alg,
    current_learning_iteration=0,
    exploration_identity=actor.exploration_identity,
    symmetry_identity=expected_symmetry_identity,
    observation_corruption_identity=lambda: deepcopy(CORRUPTION_IDENTITY),
    contact_backend_identity=lambda: backend,
  )
  saved = {
    "iter": 200,
    "actor_state_dict": {
      k: torch.full_like(v, 0.7) for k, v in actor.state_dict().items()
    },
    "critic_state_dict": {
      k: torch.full_like(v, 0.3) for k, v in critic.state_dict().items()
    },
    "optimizer_state_dict": {"must_not_load": True},
    "infos": {
      "lp_acrl_checkpoint_version": 1,
      "lp_acrl_training_symmetry": expected_symmetry_identity(),
      "lp_acrl_training_exploration": actor.exploration_identity(),
      "lp_acrl_observation_corruption": deepcopy(CORRUPTION_IDENTITY),
      "lp_acrl_contact_backend": deepcopy(backend),
      "lp_acrl": {"task_space": source_identity, "training_terrain_seed": 42},
    },
  }
  saved["actor_state_dict"]["distribution.std_param"].fill_(0.03)
  parent = {
    "task": "Pochi-LP-ACRL-Blind-v5",
    "contact_backend": deepcopy(backend),
    "learning_rate": 1e-4,
  }
  return runner, saved, parent


def test_actual_rsl_load_preserves_every_actor_critic_and_normalizer_tensor():
  runner, saved, parent = _fixture()
  source = deepcopy(saved)
  report = initialize_common_hip_refinement(runner, saved, parent)
  assert report["reset_actor_state"] == {}
  assert report["all_actor_and_critic_states_bitwise_preserved"]
  assert not runner.alg.optimizer.state and runner.current_learning_iteration == 0
  assert runner.command_term.curriculum.steps_in_stage == 0
  for key, model in (
    ("actor_state_dict", runner.alg._raw_actor),
    ("critic_state_dict", runner.alg._raw_critic),
  ):
    for name, value in model.state_dict().items():
      assert torch.equal(value, source[key][name])
      assert torch.equal(saved[key][name], source[key][name])


@pytest.mark.parametrize(
  "fault",
  [
    "source_task",
    "iteration",
    "physics",
    "commands",
    "observation",
    "action",
    "extra_reward",
    "weight",
    "posture_names",
    "existing_reward",
    "source_noise",
    "source_symmetry",
    "source_exploration",
    "source_std",
    "source_backend",
    "parent_backend",
    "target_noise",
    "target_symmetry",
    "target_exploration",
    "terrain_seed",
    "learning_rate",
    "optimizer_lr",
  ],
)
def test_undeclared_contract_changes_rejected_before_model_mutation(fault):
  runner, saved, parent = _fixture()
  current = runner.command_term.checkpoint_identity()
  metadata = saved["infos"]
  if fault == "source_task":
    parent["task"] = "Pochi-LP-ACRL-Blind-v4"
  elif fault == "iteration":
    saved["iter"] = 100
  elif fault == "physics":
    current["environment"]["physics"]["timestep"] = 0.005
  elif fault == "commands":
    current["command_protocol"]["standing_episode_fraction"] = 0.3
  elif fault == "observation":
    current["environment"]["observations"]["actor"] = "323D"
  elif fault == "action":
    current["environment"]["actions"]["scale"] = 0.5
  elif fault == "extra_reward":
    current["environment"]["rewards"]["extra"] = {}
  elif fault == "weight":
    current["environment"]["rewards"]["moving_common_hip_pitch_deviation"][
      "weight"
    ] = -5
  elif fault == "posture_names":
    current["environment"]["rewards"]["moving_common_hip_pitch_deviation"]["params"][
      "asset_cfg"
    ]["joint_names"] = ("hip",)
  elif fault == "existing_reward":
    current["environment"]["rewards"]["qualified_swing"]["weight"] = -7
  elif fault == "source_noise":
    metadata["lp_acrl_observation_corruption"]["groups"]["actor"] = True
  elif fault == "source_symmetry":
    metadata["lp_acrl_training_symmetry"]["use_data_augmentation"] = True
  elif fault == "source_exploration":
    metadata["lp_acrl_training_exploration"]["standing_action_std"] = 0.02
  elif fault == "source_std":
    saved["actor_state_dict"]["distribution.std_param"].fill_(0.04)
  elif fault == "source_backend":
    metadata["lp_acrl_contact_backend"] = {}
  elif fault == "parent_backend":
    parent["contact_backend"] = {}
  elif fault == "target_noise":
    runner.observation_corruption_identity = lambda: {}
  elif fault == "target_symmetry":
    runner.symmetry_identity = lambda: {}
  elif fault == "target_exploration":
    runner.exploration_identity = lambda: {}
  elif fault == "terrain_seed":
    metadata["lp_acrl"]["training_terrain_seed"] = 123
  elif fault == "learning_rate":
    runner.alg.learning_rate = 1e-3
  else:
    runner.alg.optimizer.param_groups[0]["lr"] = 1e-3
  before = deepcopy(runner.alg._raw_actor.state_dict())
  with pytest.raises(ValueError):
    initialize_common_hip_refinement(runner, saved, parent)
  assert all(
    torch.equal(v, before[k]) for k, v in runner.alg._raw_actor.state_dict().items()
  )


@pytest.mark.parametrize("fault", ["shape", "dtype", "nan", "keys"])
def test_invalid_critic_rejected_before_any_actor_transfer(fault):
  runner, saved, parent = _fixture()
  critic = saved["critic_state_dict"]
  if fault == "shape":
    critic["mlp.0.weight"] = torch.zeros(1, 48)
  elif fault == "dtype":
    critic["mlp.0.weight"] = critic["mlp.0.weight"].double()
  elif fault == "nan":
    critic["mlp.0.weight"][0, 0] = float("nan")
  else:
    critic.pop("mlp.0.weight")
  before = deepcopy(runner.alg._raw_actor.state_dict())
  with pytest.raises(ValueError):
    initialize_common_hip_refinement(runner, saved, parent)
  assert all(
    torch.equal(v, before[k]) for k, v in runner.alg._raw_actor.state_dict().items()
  )


@pytest.mark.parametrize("fault", ["optimizer", "lp", "iteration"])
def test_partial_training_state_cannot_be_reused(fault):
  runner, saved, parent = _fixture()
  if fault == "optimizer":
    runner.alg.optimizer.state[next(runner.alg._raw_actor.parameters())] = {}
  elif fault == "lp":
    runner.command_term.curriculum.previous_valid[0] = True
  else:
    runner.current_learning_iteration = 1
  with pytest.raises(ValueError):
    initialize_common_hip_refinement(runner, saved, parent)
