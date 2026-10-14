"""Start an isolated LP-ACRL run with random weights and recorded provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
TASKS = (
  "Pochi-LP-ACRL-Flat-v0", "Pochi-LP-ACRL-Rough-v0", "Pochi-LP-ACRL-Blind-v0"
)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", choices=TASKS, default=TASKS[1])
  parser.add_argument("--gpu", type=int, default=1)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--iterations", type=int, default=3000)
  parser.add_argument("--save-interval", type=int, default=250)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--linear-tracking-std", type=float)
  parser.add_argument("--yaw-tracking-std", type=float)
  parser.add_argument("--initial-action-std", type=float)
  parser.add_argument("--minimum-action-std", type=float)
  parser.add_argument("--fall-penalty", type=float, default=0.0)
  parser.add_argument("--enable-nan-guard", action="store_true")
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  if min(args.num_envs, args.iterations, args.save_interval) < 1 or args.gpu < 0:
    parser.error("Environment/iteration counts must be positive; GPU must be >= 0")
  for value in (
    args.linear_tracking_std,
    args.yaw_tracking_std,
    args.initial_action_std,
    args.minimum_action_std,
  ):
    if value is not None and (not math.isfinite(value) or value <= 0):
      parser.error("Standard deviations must be finite and positive")
  if not math.isfinite(args.fall_penalty) or args.fall_penalty > 0:
    parser.error("--fall-penalty must be finite and nonpositive")

  # Set device visibility before importing Torch or the simulator.
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
  os.environ["MUJOCO_GL"] = "egl"
  os.environ.setdefault("OMP_NUM_THREADS", "4")
  os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
  from mjlab.scripts.train import TrainConfig, run_train

  import pochi_rl  # noqa: F401

  cfg = TrainConfig.from_task(args.task)
  if args.minimum_action_std is not None and args.minimum_action_std > (
    args.initial_action_std if args.initial_action_std is not None
    else cfg.agent.actor.distribution_cfg["init_std"]
  ):
    parser.error("--minimum-action-std must not exceed the initial action std")
  cfg.env.scene.num_envs = args.num_envs
  for name, value in (
    ("track_lin_vel_xy_exp", args.linear_tracking_std),
    ("track_ang_vel_z_exp", args.yaw_tracking_std),
  ):
    if value is not None:
      cfg.env.rewards[name].params["std"] = value
  if args.fall_penalty:
    from mjlab.envs import mdp
    from mjlab.managers.reward_manager import RewardTermCfg

    cfg.env.rewards["termination"] = RewardTermCfg(
      func=mdp.is_terminated, weight=args.fall_penalty
    )
  if args.initial_action_std is not None:
    cfg.agent.actor.distribution_cfg["init_std"] = args.initial_action_std
  if args.minimum_action_std is not None:
    cfg.agent.actor.distribution_cfg["std_range"] = (
      args.minimum_action_std,
      1.0e6,
    )
  cfg.agent.max_iterations = args.iterations
  cfg.agent.save_interval = args.save_interval
  cfg.agent.seed = args.seed
  cfg.agent.resume = False
  cfg.agent.upload_model = False
  cfg.agent.run_name = "scratch"
  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_scratch")
  output = (
    args.output or PROJECT / "logs/rsl_rl" / cfg.agent.experiment_name / stamp
  ).resolve()
  # Refuse reuse, including runs with partial checkpoints or an old manifest.
  output.mkdir(parents=True, exist_ok=False)
  free = shutil.disk_usage(output).free
  if free < 256 * 1024**2:
    parser.error("Less than 256 MiB disk space available for this new run")
  if args.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    cfg.env.sim.nan_guard.buffer_size = 16
    cfg.env.sim.nan_guard.max_envs_to_dump = 2
    cfg.env.sim.nan_guard.output_dir = str(output / "nan_dumps")

  source_paths = [
    *sorted((PROJECT / "src/pochi_rl").rglob("*.py")),
    PROJECT / "scripts/train_lp_acrl.py",
    PROJECT / "scripts/eval_lp_acrl.py",
    PROJECT / "assets/pochi/pochi.xml",
    PROJECT / "pyproject.toml",
    PROJECT / "uv.lock",
  ]
  hashes = {}
  for source in source_paths:
    relative = source.relative_to(PROJECT)
    hashes[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    destination = output / "source" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
  for source in sorted((PROJECT / "assets/pochi/meshes").glob("*.obj")):
    hashes[str(source.relative_to(PROJECT))] = hashlib.sha256(
      source.read_bytes()
    ).hexdigest()
  revision = subprocess.run(
    ["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True
  ).stdout.strip()
  manifest = {
    "paper": "https://arxiv.org/abs/2601.17428",
    "task": args.task,
    "initialization": "random; no checkpoint loaded",
    "seed": args.seed,
    "physical_gpu": args.gpu,
    "num_envs": args.num_envs,
    "iterations_requested": args.iterations,
    "save_interval": args.save_interval,
    "tracking_kernel_std_overrides": {
      "linear": args.linear_tracking_std,
      "yaw": args.yaw_tracking_std,
    },
    "initial_action_std_override": args.initial_action_std,
    "minimum_action_std_override": args.minimum_action_std,
    "fall_penalty": args.fall_penalty,
    "physics_dt": cfg.env.sim.mujoco.timestep,
    "physics_solver": cfg.env.sim.mujoco.solver,
    "friction_cone": cfg.env.sim.mujoco.cone,
    "decimation": cfg.env.decimation,
    "nan_guard_enabled": args.enable_nan_guard,
    "command_training_protocol": {
      key: getattr(cfg.env.commands["base_velocity"], key)
      for key in (
        "standing_episode_fraction", "transition_episode_fraction",
        "transition_interval_s", "transition_straight_probability",
        "transition_forward_range", "zero_yaw_probability",
        "zero_lateral_probability",
      )
    },
    "auxiliary_episode_lp_credit": False,
    "training_symmetry": asdict(cfg.agent.algorithm).get("symmetry_cfg"),
    "git_revision": revision,
    "source_sha256": hashes,
    "started_at": datetime.now(UTC).isoformat(),
    "status": "running",
  }
  manifest_path = output / "run.json"

  def write_manifest() -> None:
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")

  write_manifest()
  print(f"Fresh LP-ACRL run: {output}", flush=True)
  try:
    run_train(args.task, cfg, output)
  except BaseException as exc:
    manifest["status"] = (
      "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
    )
    manifest["error"] = f"{type(exc).__name__}: {exc}"
    raise
  else:
    manifest["status"] = "completed"
  finally:
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    write_manifest()


if __name__ == "__main__":
  main()
