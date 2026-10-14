"""Strict v5model200-to-v8 model-only transfer with one added reward term."""

from __future__ import annotations

from copy import deepcopy

import torch

from pochi_rl.mjlab.lp_acrl_blind_v8_env_cfg import common_hip_reward_cfg
from pochi_rl.mjlab.lp_acrl_commands import _config_identity
from pochi_rl.mjlab.lp_acrl_low_noise import (
  CORRUPTION_IDENTITY,
  expected_exploration_identity,
  expected_symmetry_identity,
  validate_low_noise_checkpoint,
)
from pochi_rl.mjlab.lp_acrl_refinement import (
  MODEL_ONLY_LOAD,
  _assert_fresh_training_state,
  without_rewards,
)

SOURCE_TASK = "Pochi-LP-ACRL-Blind-v5"
SOURCE_ITERATION = 200
REWARD_NAME = "moving_common_hip_pitch_deviation"


def validate_common_hip_refinement_identity(runner, saved: dict, parent: dict) -> None:
  if parent.get("task") != SOURCE_TASK or saved.get("iter") != SOURCE_ITERATION:
    raise ValueError("Common-hip refinement requires the pinned Blind-v5 model200")
  validate_low_noise_checkpoint(saved)
  metadata = saved["infos"]
  if metadata.get("lp_acrl_checkpoint_version") != 1:
    raise ValueError("Expected an independent LP-ACRL checkpoint")
  if (
    parent.get("learning_rate") != 1e-4
    or runner.alg.learning_rate != 1e-4
    or any(group["lr"] != 1e-4 for group in runner.alg.optimizer.param_groups)
  ):
    raise ValueError("Initial PPO learning rate must remain1e-4")
  term = runner.command_term
  current = term.checkpoint_identity()
  source = (metadata.get("lp_acrl") or {}).get("task_space")
  if without_rewards(source) != without_rewards(current):
    raise ValueError("Common-hip refinement differs outside rewards")
  if not source.get("command_protocol"):
    raise ValueError("Source lacks the blind stop/start command protocol")
  rewards = deepcopy(source["environment"]["rewards"])
  if REWARD_NAME in rewards:
    raise ValueError("Source already contains the prospective common-hip-pitch reward")
  rewards[REWARD_NAME] = _config_identity(common_hip_reward_cfg())
  if current["environment"]["rewards"] != rewards:
    raise ValueError("Only the exact moving common-hip-pitch reward may be added")
  terrain = term._env.cfg.scene.terrain
  generator = terrain.terrain_generator if terrain is not None else None
  if metadata["lp_acrl"].get("training_terrain_seed") != (
    None if generator is None else generator.seed
  ):
    raise ValueError("Training terrain seed must remain unchanged")
  if (
    runner.exploration_identity() != expected_exploration_identity()
    or runner.symmetry_identity() != expected_symmetry_identity()
    or runner.observation_corruption_identity() != CORRUPTION_IDENTITY
  ):
    raise ValueError("Target observation, exploration and symmetry must remain exact")
  backend = runner.contact_backend_identity()
  if (
    not backend
    or metadata.get("lp_acrl_contact_backend") != backend
    or parent.get("contact_backend") != backend
  ):
    raise ValueError("Checkpoint, parent manifest and target backend must match")


def initialize_common_hip_refinement(runner, saved: dict, parent: dict) -> dict:
  """Keep every model tensor exactly; optimizer, LP and iteration remain fresh."""
  validate_common_hip_refinement_identity(runner, saved, parent)
  _assert_fresh_training_state(runner)
  for key, model in (
    ("actor_state_dict", runner.alg._raw_actor),
    ("critic_state_dict", runner.alg._raw_critic),
  ):
    source, target = saved.get(key), model.state_dict()
    if not isinstance(source, dict) or source.keys() != target.keys():
      raise ValueError(f"Model state keys differ: {key}")
    for name, value in source.items():
      if (
        not isinstance(value, torch.Tensor)
        or value.shape != target[name].shape
        or value.dtype != target[name].dtype
        or (value.is_floating_point() and not torch.isfinite(value).all())
      ):
        raise ValueError(f"Invalid model tensor: {key}/{name}")
  if runner.alg._raw_actor.distribution.std_param.requires_grad:
    raise ValueError("Moving standard deviation must remain fixed")
  if runner.alg.load(saved, dict(MODEL_ONLY_LOAD), strict=True):
    raise ValueError("Model-only refinement unexpectedly restored iteration")
  for key, model in (
    ("actor_state_dict", runner.alg._raw_actor),
    ("critic_state_dict", runner.alg._raw_critic),
  ):
    if any(
      not torch.equal(value, saved[key][name].to(value.device))
      for name, value in model.state_dict().items()
    ):
      raise ValueError(f"Model transfer changed source state: {key}")
  _assert_fresh_training_state(runner)
  return {
    "loaded": ["actor", "critic", "actor_normalizer", "critic_normalizer"],
    "not_loaded": ["optimizer", "LP_statistics", "iteration", "RND"],
    "reset_actor_state": {},
    "all_actor_and_critic_states_bitwise_preserved": True,
    "source_iteration": SOURCE_ITERATION,
    "initial_iteration": 0,
    "initial_lp_stage": 0,
    "initial_lp_steps": 0,
    "initial_lp_episode_count": 0,
    "initial_optimizer_state_entries": 0,
    "initial_learning_rate": runner.alg.learning_rate,
    "initial_exploration_identity": runner.exploration_identity(),
    "observation_corruption_unchanged": runner.observation_corruption_identity(),
    "symmetry_unchanged": runner.symmetry_identity(),
    "contact_backend_unchanged": runner.contact_backend_identity(),
    "allowed_environment_identity_difference": [f"environment.rewards.{REWARD_NAME}"],
  }
