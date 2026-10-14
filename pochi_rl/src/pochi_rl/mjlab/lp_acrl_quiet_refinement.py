"""Reward-only refinement after quiet-stop exploration has been established.

Actor/critic weights and normalizers transfer within Blind-v2/v3 only. Every
nonreward environment field, symmetry and exploration semantics stay exact;
optimizer, iteration and all LP statistics start fresh.
"""

from __future__ import annotations

from copy import deepcopy

from pochi_rl.mjlab.lp_acrl_quiet_stop import QuietStopMLPModel
from pochi_rl.mjlab.lp_acrl_refinement import (
  MODEL_ONLY_LOAD,
  _assert_fresh_training_state,
  without_rewards,
)

QUIET_SOURCE_TASKS = ("Pochi-LP-ACRL-Blind-v2", "Pochi-LP-ACRL-Blind-v3")


def validate_quiet_refinement_identity(
  saved: dict, parent_manifest: dict, current_identity: dict,
  current_symmetry: dict, current_exploration: dict, terrain_seed: int | None,
) -> None:
  """Permit reward changes only, including exact conditional exploration."""
  if parent_manifest.get("task") not in QUIET_SOURCE_TASKS:
    raise ValueError("Quiet refinement source must be Blind-v2/v3")
  metadata = saved.get("infos") or {}
  if metadata.get("lp_acrl_checkpoint_version") != 1:
    raise ValueError("Expected an independent LP-ACRL checkpoint")
  lp_state = metadata.get("lp_acrl") or {}
  identity = lp_state.get("task_space")
  if without_rewards(identity) != without_rewards(current_identity):
    raise ValueError("Quiet refinement identity differs outside rewards")
  if not identity.get("command_protocol"):
    raise ValueError("Source lacks the blind standing/stop-start protocol")
  if lp_state.get("training_terrain_seed") != terrain_seed:
    raise ValueError("Quiet refinement must preserve training terrain geometry seed")
  if not current_symmetry or not (
    current_symmetry.get("use_data_augmentation")
    and current_symmetry.get("use_mirror_loss")
  ):
    raise ValueError("Quiet refinement requires both symmetry training modes")
  if metadata.get("lp_acrl_training_symmetry") != current_symmetry:
    raise ValueError("Quiet refinement symmetry configuration differs")
  expected_class = f"{QuietStopMLPModel.__module__}:{QuietStopMLPModel.__qualname__}"
  if (
    not isinstance(current_exploration, dict)
    or current_exploration.get("actor_class") != expected_class
    or current_exploration.get("standing_action_std") != QuietStopMLPModel.QUIET_STD
    or current_exploration.get("standing_max_abs_command")
    != QuietStopMLPModel.STOP_THRESHOLD
    or metadata.get("lp_acrl_training_exploration") != current_exploration
  ):
    raise ValueError("Quiet refinement exploration identity is missing or differs")


def initialize_quiet_refinement(runner, saved: dict, parent_manifest: dict) -> dict:
  """Strictly load both networks, preserving empty optimizer/LP/progress state."""
  import torch

  term = runner.command_term
  terrain = term._env.cfg.scene.terrain
  generator = terrain.terrain_generator if terrain is not None else None
  exploration = runner.exploration_identity()
  validate_quiet_refinement_identity(
    saved, parent_manifest, term.checkpoint_identity(), runner.symmetry_identity(),
    exploration, None if generator is None else generator.seed,
  )
  _assert_fresh_training_state(runner)
  # Validate both networks before mutating either. The state dictionaries
  # include all observation-normalizer buffers as well as network parameters.
  for key, model in (
    ("actor_state_dict", runner.alg._raw_actor),
    ("critic_state_dict", runner.alg._raw_critic),
  ):
    source = saved.get(key)
    target = model.state_dict()
    if not isinstance(source, dict) or source.keys() != target.keys():
      raise ValueError(f"Quiet refinement {key} architecture/state keys differ")
    for name, value in source.items():
      if (
        not isinstance(value, torch.Tensor)
        or value.shape != target[name].shape
        or value.dtype != target[name].dtype
      ):
        raise ValueError(f"Quiet refinement {key}/{name} shape or dtype differs")
      if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"Quiet refinement {key}/{name} contains nonfinite values")
  load_iteration = runner.alg.load(saved, dict(MODEL_ONLY_LOAD), strict=True)
  if load_iteration:
    raise ValueError("Model-only refinement unexpectedly requested iteration restore")
  _assert_fresh_training_state(runner)
  return {
    "loaded": ["actor", "critic", "actor_normalizer", "critic_normalizer"],
    "not_loaded": ["optimizer", "LP_statistics", "iteration", "RND"],
    "source_iteration": int(saved["iter"]),
    "initial_iteration": runner.current_learning_iteration,
    "initial_lp_stage": term.curriculum.stage_index,
    "initial_lp_steps": term.curriculum.steps_in_stage,
    "initial_lp_episode_count": int(term.curriculum.episode_counts.sum()),
    "initial_optimizer_state_entries": len(runner.alg.optimizer.state),
    "initial_exploration_identity": deepcopy(exploration),
    "allowed_environment_identity_difference": ["environment.rewards"],
  }
