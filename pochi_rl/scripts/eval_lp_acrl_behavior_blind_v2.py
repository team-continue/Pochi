"""Register the isolated Blind-v2 task, then run the frozen plane diagnostic."""

from __future__ import annotations

import argparse
import sys

TASK_ID = "Pochi-LP-ACRL-Blind-v2"


def main():
  task_parser = argparse.ArgumentParser(add_help=False)
  task_parser.add_argument("--task", default=TASK_ID)
  args, _ = task_parser.parse_known_args()
  if args.task != TASK_ID:
    task_parser.error(f"This wrapper only evaluates {TASK_ID}")

  from eval_lp_acrl_behavior_plane import main as plane_main

  from pochi_rl.mjlab.lp_acrl_blind_v2 import register_blind_v2

  register_blind_v2()
  if not any(arg == "--task" or arg.startswith("--task=") for arg in sys.argv[1:]):
    sys.argv.extend(("--task", TASK_ID))
  plane_main()


if __name__ == "__main__":
  main()
