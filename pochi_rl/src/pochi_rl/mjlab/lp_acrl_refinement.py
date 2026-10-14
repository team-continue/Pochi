"""Strict weight-only refinement within the independently trained blind family."""

from __future__ import annotations

import math
from copy import deepcopy

BLIND_SOURCE_TASKS = ("Pochi-LP-ACRL-Blind-v0", "Pochi-LP-ACRL-Blind-v1")
MODEL_ONLY_LOAD = {
  "actor": True, "critic": True, "optimizer": False,
  "iteration": False, "rnd": False,
}


def without_rewards(identity: dict) -> dict:
  """Remove exactly the reward description; keep every other identity field."""
  if not isinstance(identity, dict):
    raise ValueError("Missing checkpoint task identity")
  result = deepcopy(identity)
  environment = result.get("environment")
  if not isinstance(environment, dict) or "rewards" not in environment:
    raise ValueError("Missing environment/reward identity")
  environment.pop("rewards")
  return result


def validate_refinement_identity(
  saved: dict, parent_manifest: dict, current_identity: dict,
  current_symmetry: dict, terrain_seed: int | None,
) -> None:
  """Reject old policies and every non-reward environment difference."""
  if parent_manifest.get("task") not in BLIND_SOURCE_TASKS:
    raise ValueError("Refinement source must be independently trained Blind-v0/v1")
  metadata = saved.get("infos") or {}
  if metadata.get("lp_acrl_checkpoint_version") != 1:
    raise ValueError("Expected an independent LP-ACRL checkpoint")
  lp_state = metadata.get("lp_acrl") or {}
  identity = lp_state.get("task_space")
  if without_rewards(identity) != without_rewards(current_identity):
    raise ValueError("Refinement identity differs outside rewards")
  if not identity.get("command_protocol"):
    raise ValueError("Source lacks the blind standing/stop-start protocol")
  if lp_state.get("training_terrain_seed") != terrain_seed:
    raise ValueError("Refinement must preserve training terrain geometry seed")
  if not current_symmetry or not (
    current_symmetry.get("use_data_augmentation")
    and current_symmetry.get("use_mirror_loss")
  ):
    raise ValueError("Blind refinement requires both symmetry training modes")
  if metadata.get("lp_acrl_training_symmetry") != current_symmetry:
    raise ValueError("Refinement symmetry configuration differs")


def _assert_fresh_training_state(runner) -> None:
  import torch

  curriculum = runner.command_term.curriculum
  if runner.current_learning_iteration != 0 or runner.alg.optimizer.state:
    raise ValueError("Refinement requires a new runner and empty optimizer state")
  if curriculum.stage_index or curriculum.steps_in_stage:
    raise ValueError("Refinement requires a fresh LP stage counter")
  for name in (
    "episode_counts", "return_sums", "previous_means", "previous_valid",
    "learning_progress",
  ):
    if getattr(curriculum, name).any():
      raise ValueError("Refinement requires empty LP episode statistics")
  expected = torch.full_like(curriculum.probabilities, 1 / curriculum.num_tasks)
  if not torch.equal(curriculum.probabilities, expected):
    raise ValueError("Refinement requires uniform initial task probabilities")


def initialize_refinement(runner, saved: dict, parent_manifest: dict) -> dict:
  """Load actor, critic and their normalizers without touching training state."""
  import torch

  term = runner.command_term
  terrain = term._env.cfg.scene.terrain
  generator = terrain.terrain_generator if terrain is not None else None
  validate_refinement_identity(
    saved, parent_manifest, term.checkpoint_identity(), runner.symmetry_identity(),
    None if generator is None else generator.seed,
  )
  _assert_fresh_training_state(runner)
  # Validate both networks before mutating either. Normalizer buffers are part
  # of these model state dictionaries and receive the same strict checks.
  for key, model in (
    ("actor_state_dict", runner.alg._raw_actor),
    ("critic_state_dict", runner.alg._raw_critic),
  ):
    source = saved.get(key)
    target = model.state_dict()
    if not isinstance(source, dict) or source.keys() != target.keys():
      raise ValueError(f"Refinement {key} architecture/state keys differ")
    for name, value in source.items():
      if (
        not isinstance(value, torch.Tensor)
        or value.shape != target[name].shape
        or value.dtype != target[name].dtype
      ):
        raise ValueError(f"Refinement {key}/{name} shape or dtype differs")
      if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"Refinement {key}/{name} contains nonfinite values")
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
  }


def output_budget_bytes(
  checkpoint_bytes: int, iterations: int, save_interval: int,
  source_bytes: int, *, guard_model_bytes: int = 0,
) -> int:
  """Budget exact save count with25% checkpoint growth, logs and reserve.

  A NaN guard can write a compiled model plus small selected-world histories.
  Its actual compiled-model size can be added after simulator construction.
  """
  if min(checkpoint_bytes, iterations, save_interval) < 1:
    raise ValueError("Checkpoint size, iterations and save interval must be positive")
  if min(source_bytes, guard_model_bytes) < 0:
    raise ValueError("Artifact sizes cannot be negative")
  saves = len(set(range(0, iterations, save_interval)) | {iterations - 1})
  return (
    math.ceil(checkpoint_bytes * 1.25) * saves + source_bytes
    + 8 * 1024**2 + 32 * 1024**2 + 2 * guard_model_bytes
  )
