"""Evaluate the IK side jump in CPU MuJoCo; a failed landing exits nonzero."""

from __future__ import annotations

import argparse
import json

import numpy as np

from pochi_rl.control.side_jump import JUMP_SIZES, SideJumpConfig, SideJumpSim


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--direction", choices=("left", "right"), default="left")
  parser.add_argument("--jump-size", choices=JUMP_SIZES, default="large")
  parser.add_argument(
    "--lateral-speed",
    type=float,
    default=None,
    help="override the selected preset takeoff speed (m/s)",
  )
  parser.add_argument("--duration", type=float, default=10.0)
  args = parser.parse_args()
  if not np.isfinite(args.duration) or args.duration <= 0:
    parser.error("--duration must be finite and positive")
  try:
    cfg = SideJumpConfig.for_size(
      args.jump_size,
      direction=1 if args.direction == "left" else -1,
      lateral_speed=args.lateral_speed,
    )
  except ValueError as exc:
    parser.error(str(exc))
  result = SideJumpSim(cfg).run(args.duration)
  print(json.dumps(result, indent=2))
  return 0 if result["success"] else 1


if __name__ == "__main__":
  raise SystemExit(main())
