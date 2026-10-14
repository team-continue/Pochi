"""Separate lateral/yaw diagnostic; never changes the registered V16 selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

COMMANDS = {
  "left": (0.0, 0.1, 0.0),
  "right": (0.0, -0.1, 0.0),
  "counterclockwise": (0.0, 0.0, 0.6),
  "clockwise": (0.0, 0.0, -0.6),
}
LIMITS = {
  "vx_error_mean_m_s": 0.05,
  "vy_error_mean_m_s": 0.05,
  "yaw_error_mean_rad_s": 0.10,
  "contact_slip_mean_m_s": 0.10,
  "tilt_p95_deg": 15.0,
  "nonfoot_contact_fraction": 0.01,
  "foot_contact_duty_spread": 0.30,
}


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--device", default="cuda:0")
  args = parser.parse_args()
  if args.output.exists():
    parser.error("Preserve previous results: output must not exist")
  import torch
  from eval_lp_acrl_behavior_blind_v16 import (
    validate_backend_identity,
    validate_task_space,
  )

  from pochi_rl.mjlab.lp_acrl_blind_v16 import register_blind_v16
  from pochi_rl.mjlab.lp_acrl_rough_no_touchdown_refinement import (
    validate_rough_no_touchdown_checkpoint,
  )
  from pochi_rl.mjlab.warp_contact_backend import (
    active_contact_backend_identity,
    install_contact_backend,
  )

  backend = install_contact_backend()
  saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  validate_backend_identity(saved, backend)
  validate_task_space(saved)
  validate_rough_no_touchdown_checkpoint(saved)
  checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
  register_blind_v16()
  import eval_lp_acrl_behavior_plane as plane

  # Reuse its verified plane, physical identity check and first-episode rollout.
  # These process-local hooks apply only to this additional direction diagnostic.
  frozen = plane.frozen
  original_segment = frozen.summarize_segment
  plane.PROFILES = frozen.PROFILES = tuple(COMMANDS)
  frozen.profile_command = lambda profile, _time: COMMANDS[profile]

  def summarize_segment(samples, index, start, end, command, dt, *, settle_s):
    result = original_segment(
      samples, index, start, end, command, dt, settle_s=settle_s
    )
    if result["metrics"] is None:
      return result
    window = ((samples["time_s"] > start + settle_s + 1e-9)
              & (samples["time_s"] <= end + 1e-9)
              & samples["valid"][:, index])
    velocity = samples["velocity"][window, index]
    error = np.abs(velocity - np.asarray(command)).mean(axis=0)
    metrics = result["metrics"]
    metrics.update(zip(
      ("vx_error_mean_m_s", "vy_error_mean_m_s", "yaw_error_mean_rad_s"),
      map(float, error), strict=True,
    ))
    duty = np.asarray(metrics["foot_contact_fraction"])
    checks = {name: metrics[name] <= limit for name, limit in LIMITS.items()}
    checks["all_legs_contact_and_swing"] = bool(
      (duty >= 0.15).all() and (duty <= 0.85).all()
    )
    checks["all_legs_footfalls"] = min(metrics["footfalls_per_leg"]) >= 2
    result.update(checks=checks, passed=result["complete"] and all(checks.values()))
    return result

  frozen.summarize_segment = summarize_segment
  sys.argv = [sys.argv[0], "--task", "Pochi-LP-ACRL-Blind-v16",
              "--checkpoint", str(args.checkpoint), "--output", str(args.output),
              "--device", args.device, "--num-envs", "32", "--seed", "5456"]
  plane.main()
  report = json.loads(args.output.read_text())
  if report["checkpoint_sha256"] != checkpoint_sha:
    raise ValueError("Checkpoint changed during evaluation")
  if active_contact_backend_identity() != backend:
    raise ValueError("Contact backend changed during evaluation")
  report["protocol"] = "pochi-v16-lateral-yaw-diagnostic-v1"
  report["contact_backend"] = backend
  report["diagnostic_only_not_selection_or_holdout"] = True
  report["checkpoint_contract_checked_before_environment_creation"] = True
  conventions = report["metric_conventions"]
  conventions["profile_commands"] = COMMANDS
  conventions.pop("standing_limits")
  conventions["walking_limits"] = {
    **LIMITS, "foot_contact_duty_min": 0.15, "foot_contact_duty_max": 0.85,
    "footfalls_per_leg_min": 2,
  }
  conventions["transient_exclusion_s"] = {"constant_profiles": 2.0}
  conventions["purpose"] = "Additional direction diagnostic; not model selection"
  conventions["tracking"] = (
    "Mean absolute body-frame error relative to each nonzero desired command; "
    "turning is not graded as unwanted yaw and lateral motion is not graded as drift"
  )
  report["source_sha256"][Path(__file__).name] = hashlib.sha256(
    Path(__file__).read_bytes()
  ).hexdigest()
  args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
  main()
