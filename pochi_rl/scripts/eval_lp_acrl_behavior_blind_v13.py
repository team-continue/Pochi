"""Validate v13 backend, noise, symmetry and complete reward contracts."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

TASK_ID = "Pochi-LP-ACRL-Blind-v13"


def validate_backend_identity(saved: dict, active: dict) -> None:
  """The frozen evaluator's base actor loader cannot enforce this new identity."""
  expected = saved.get("infos", {}).get("lp_acrl_contact_backend")
  if not isinstance(expected, dict) or expected != active:
    raise ValueError(
      "Checkpoint lp_acrl_contact_backend must exactly match the active v13 "
      "backend; missing identities and legacy-checkpoint migration are rejected"
    )


def validate_reward_identity(saved: dict, expected: dict) -> None:
  """Enforce the complete v13 reward identity before the base actor loader."""
  info = saved.get("infos") or {}
  state = info.get("lp_acrl") or {}
  space = state.get("task_space") or {}
  environment = space.get("environment") or {}
  if not expected or environment.get("rewards") != expected:
    raise ValueError("Checkpoint rewards must exactly match the declared v13 factory")



def validate_task_space(saved: dict) -> dict:
  """Require the complete declared plane/five-task identity before any env."""
  from pochi_rl.mjlab.lp_acrl_flat_focus_identity import (
    canonical_rough_task_space,
    expected_flat_focus_task_space,
    validate_flat_focus_refinement_identity,
  )

  rough = canonical_rough_task_space()
  expected = expected_flat_focus_task_space(rough)
  curriculum = saved.get("infos", {}).get("lp_acrl", {})
  if (
    "training_terrain_seed" not in curriculum
    or curriculum["training_terrain_seed"] is not None
  ):
    raise ValueError("Flat training terrain seed must be present and None")
  actual = curriculum.get("task_space")
  try:
    validate_flat_focus_refinement_identity(
      rough, actual, source_training_terrain_seed=42,
      target_training_terrain_seed=curriculum["training_terrain_seed"],
    )
  except ValueError as error:
    raise ValueError(
      "Checkpoint task space must exactly match declared V13 flat focus"
    ) from error
  return expected


def main():
  parser = argparse.ArgumentParser(add_help=False)
  parser.add_argument("--task", default=TASK_ID)
  parser.add_argument("--checkpoint", type=Path)
  parser.add_argument("--output", type=Path)
  args, _ = parser.parse_known_args()
  if args.task != TASK_ID:
    parser.error(f"This wrapper only evaluates {TASK_ID}")
  if not any(arg == "--task" or arg.startswith("--task=") for arg in sys.argv[1:]):
    sys.argv.extend(("--task", TASK_ID))
  if "--help" in sys.argv or "-h" in sys.argv:
    from eval_lp_acrl_behavior_plane import main as plane_main

    plane_main()
    return
  if args.output is None or args.checkpoint is None:
    parser.error("--checkpoint and --output are required")
  if args.output.exists():
    parser.error("--output already exists; preserve every earlier result")
  if not args.checkpoint.is_file():
    parser.error("--checkpoint must name an existing file")

  import torch

  from pochi_rl.mjlab.warp_contact_backend import (
    active_contact_backend_identity,
    install_contact_backend,
  )

  backend = install_contact_backend()
  if active_contact_backend_identity() != backend:
    raise ValueError("Installed contact backend is not the active identity")
  saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  validate_backend_identity(saved, backend)
  from pochi_rl.mjlab.lp_acrl_flat_focus_refinement import (
    CORRUPTION_IDENTITY,
    STATE_KEY,
    expected_exploration_identity,
    expected_symmetry_identity,
    validate_flat_focus_checkpoint,
  )

  completed_updates = validate_flat_focus_checkpoint(saved)
  from pochi_rl.mjlab.lp_acrl_blind_v13_env_cfg import (
    pochi_lp_acrl_blind_v13_env_cfg,
  )
  from pochi_rl.mjlab.lp_acrl_commands import _config_identity

  rewards = _config_identity(pochi_lp_acrl_blind_v13_env_cfg(play=False).rewards)
  validate_reward_identity(saved, rewards)
  task_space = validate_task_space(saved)
  checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()

  from pochi_rl.mjlab.lp_acrl_blind_v13 import register_blind_v13

  register_blind_v13()
  from eval_lp_acrl_behavior_plane import main as plane_main

  plane_main()
  report = json.loads(args.output.read_text())
  if report["checkpoint_sha256"] != checkpoint_hash:
    raise ValueError("Checkpoint changed after backend identity validation")
  if active_contact_backend_identity() != backend:
    raise ValueError("Contact backend changed during evaluation")
  report["contact_backend"] = backend
  report["contact_backend_validation"] = {
    "checkpoint_identity_checked_before_environment_creation": True,
    "checkpoint_migration_permitted": False,
    "frozen_plane_metrics_profiles_and_thresholds_unchanged": True,
    "backend_transition": (
      "The exact pinned v4 process-local discrete EPA correction is retained; old "
      "checkpoint results retain their original backend and are not relabeled"
    ),
  }
  report["reward_identity"] = rewards
  report["reward_identity_validation"] = {
    "full_declared_reward_configuration_checked": True,
    "checked_before_environment_creation": True,
    "checkpoint_reward_migration_permitted": False,
  }
  report["observation_corruption"] = CORRUPTION_IDENTITY
  report["training_exploration"] = expected_exploration_identity()
  report["training_symmetry"] = expected_symmetry_identity()
  report["flat_focus_task_space"] = task_space
  report["flat_focus_task_space_validation"] = {
    "complete_canonical_projection_checked_before_environment_creation": True,
    "task_count": 5,
    "true_plane_training": True,
    "saved_training_terrain_seed_is_none": True,
    "checkpoint_identity_rewritten": False,
  }
  report["flat_focus_state"] = saved["infos"][STATE_KEY]
  report["training_common_step_counter"] = (
    saved["infos"]["env_state"]["common_step_counter"]
  )
  report["flat_focus_validation"] = {
    "checked_before_environment_creation": True,
    "checkpoint_migration_permitted": False,
    "full_exploration_identity_checked": True,
    "saved_fixed_moving_std_checked": True,
    "moving_action_std": 0.15,
    "completed_updates_checked_against_iteration": True,
    "adaptive_learning_rate_checked_against_optimizer": True,
    "completed_updates": completed_updates,
    "saved_environment_counter_checked": True,
    "legacy_progress_states_rejected": True,
    "initialization": "model_only_fresh_optimizer_lp",
    "standing_action_std": 0.01,
    "frozen_plane_metrics_profiles_and_thresholds_unchanged": True,
  }
  report["source_sha256"][Path(__file__).name] = hashlib.sha256(
    Path(__file__).read_bytes()
  ).hexdigest()
  args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
  main()
