"""Evaluate closed-loop MPC walking without opening a viewer.

uv run --extra mpc python scripts/eval_mpc_walk.py --duration 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from pochi_rl.control.mpc_walk import MPCWalkSim, WalkConfig


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--duration", type=float, default=30.0)
  parser.add_argument("--speed", type=float, default=0.06)
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  if not np.isfinite(args.duration) or args.duration <= 0:
    parser.error("--duration must be finite and positive")
  sim = MPCWalkSim(WalkConfig(speed=args.speed))
  result = sim.run(args.duration)
  expected, _ = sim.command_at(args.duration)
  result["expected_distance_m"] = expected
  result["tracking_error_m"] = result["distance_m"] - expected
  result["passed"] = bool(
    not result["failed"]
    and result["solver_failures"] == 0
    and result["nonfoot_ground_force_n"] < 0.5
    and result["min_height_m"] > 0.28
    and result["max_tilt_deg"] < 10
    and abs(result["tracking_error_m"]) < 0.08
    and abs(result["lateral_drift_m"]) < 0.05
  )
  report = json.dumps(result, indent=2) + "\n"
  print(report, end="")
  if args.output:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report)
  return 0 if result["passed"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
