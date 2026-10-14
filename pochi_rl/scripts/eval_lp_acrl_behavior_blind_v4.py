"""Validate the explicit v4 contact backend, then use frozen plane criteria."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

TASK_ID = "Pochi-LP-ACRL-Blind-v4"


def validate_backend_identity(saved: dict, active: dict) -> None:
  """The frozen evaluator's base actor loader cannot enforce this new identity."""
  expected = saved.get("infos", {}).get("lp_acrl_contact_backend")
  if not isinstance(expected, dict) or expected != active:
    raise ValueError(
      "Checkpoint lp_acrl_contact_backend must exactly match the active v4 "
      "backend; missing identities and legacy-checkpoint migration are rejected"
    )


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
  checkpoint_hash = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()

  from pochi_rl.mjlab.lp_acrl_blind_v4 import register_blind_v4

  register_blind_v4()
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
      "Explicit v4 process-local discrete EPA tolerance correction; old "
      "checkpoint results retain their original backend and are not relabeled"
    ),
  }
  report["source_sha256"][Path(__file__).name] = hashlib.sha256(
    Path(__file__).read_bytes()
  ).hexdigest()
  args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
  main()
