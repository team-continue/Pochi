"""Isolated quiet-stop exploration experiment within the new blind family.

This is NOT from-scratch training or a full optimizer/curriculum resume.
Only actor/critic states, including their normalizers, initialize the new run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
TARGET_TASK = "Pochi-LP-ACRL-Blind-v2"


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def verify_parent_assets(parent: dict) -> dict[str, str]:
  """Physical asset equality is necessary in addition to config identity."""
  expected = {
    name: value for name, value in parent.get("source_sha256", {}).items()
    if name.startswith("assets/")
  }
  if "assets/pochi/pochi.xml" not in expected or not any(
    name.startswith("assets/pochi/meshes/") for name in expected
  ):
    raise ValueError("Parent manifest must include robot XML and mesh hashes")
  for name, digest in expected.items():
    source = (PROJECT / name).resolve()
    if not source.is_relative_to(PROJECT) or sha256(source) != digest:
      raise ValueError(f"Physical asset differs from parent: {name}")
  return expected


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", choices=[TARGET_TASK], default=TARGET_TASK)
  parser.add_argument("--init-checkpoint", type=Path, required=True)
  parser.add_argument("--gpu", type=int, default=1)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--iterations", type=int, default=1000)
  parser.add_argument("--save-interval", type=int, default=100)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--enable-nan-guard", action="store_true")
  parser.add_argument("--output", type=Path)
  args = parser.parse_args()
  if min(args.num_envs, args.iterations, args.save_interval) < 1 or args.gpu < 0:
    parser.error("Counts must be positive and GPU must be nonnegative")
  checkpoint = args.init_checkpoint.resolve(strict=True)
  parent_path = checkpoint.parent / "run.json"
  parent = json.loads(parent_path.read_text())
  parent_assets = verify_parent_assets(parent)

  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
  os.environ["MUJOCO_GL"] = "egl"
  os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
  os.environ.setdefault("OMP_NUM_THREADS", "4")
  os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
  import mujoco
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.scripts.train import TrainConfig
  from mjlab.utils.os import dump_yaml
  from mjlab.utils.torch import configure_torch_backends

  import pochi_rl  # noqa: F401
  from pochi_rl.mjlab.lp_acrl_blind_v2 import register_blind_v2
  from pochi_rl.mjlab.lp_acrl_quiet_stop import LPACRLQuietStopOnPolicyRunner
  from pochi_rl.mjlab.lp_acrl_refinement import (
    initialize_refinement,
    output_budget_bytes,
  )

  register_blind_v2()

  if parent.get("task") != "Pochi-LP-ACRL-Blind-v1":
    parser.error("Quiet-stop comparison must initialize from Blind-v1")
  saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
  cfg = TrainConfig.from_task(args.task)
  cfg.env.scene.num_envs = args.num_envs
  cfg.env.seed = cfg.agent.seed = args.seed
  cfg.agent.max_iterations = args.iterations
  cfg.agent.save_interval = args.save_interval
  cfg.agent.resume = False
  cfg.agent.upload_model = False
  cfg.agent.run_name = "refinement"

  stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_refinement")
  output = (
    args.output or PROJECT / "logs/rsl_rl" / cfg.agent.experiment_name / stamp
  ).resolve()
  if output.exists():
    raise FileExistsError(f"Refinement output must be new: {output}")
  source_paths = [
    *sorted((PROJECT / "src/pochi_rl").rglob("*.py")),
    PROJECT / "scripts/train_lp_acrl_blind_quiet_stop.py",
    PROJECT / "scripts/eval_lp_acrl.py",
    PROJECT / "scripts/eval_lp_acrl_behavior.py",
    PROJECT / "assets/pochi/pochi.xml",
    PROJECT / "pyproject.toml", PROJECT / "uv.lock",
  ]
  source_paths = [path for path in source_paths if path.exists()]
  source_bytes = sum(path.stat().st_size for path in source_paths)
  budget = output_budget_bytes(
    checkpoint.stat().st_size, args.iterations, args.save_interval, source_bytes,
  )
  capacity_path = output.parent
  while not capacity_path.exists():
    capacity_path = capacity_path.parent
  available = shutil.disk_usage(capacity_path).free
  if available < budget:
    raise OSError(f"Refinement needs estimated {budget} bytes; only {available} free")
  output.mkdir(parents=True, exist_ok=False)
  if args.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    cfg.env.sim.nan_guard.buffer_size = 16
    cfg.env.sim.nan_guard.max_envs_to_dump = 2
    cfg.env.sim.nan_guard.output_dir = str(output / "nan_dumps")
  hashes = dict(parent_assets)
  for source in source_paths:
    relative = source.relative_to(PROJECT)
    hashes[str(relative)] = sha256(source)
    destination = output / "source" / relative
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
  configure_torch_backends()
  manifest = {
    "paper": "https://arxiv.org/abs/2601.17428",
    "task": args.task,
    "initialization": "explicit blind-family refinement; NOT from scratch",
    "source_checkpoint": str(checkpoint),
    "source_checkpoint_sha256": sha256(checkpoint),
    "parent_manifest": str(parent_path),
    "parent_manifest_sha256": sha256(parent_path),
    "parent_task": parent["task"],
    "parent_initialization": parent.get("initialization"),
    "source_iteration": int(saved.get("iter", -1)),
    "allowed_environment_identity_difference": [],
    "experiment": "quiet-stop exploration only; v1 environment/rewards unchanged",
    "training_exploration": {
      "exact_zero_command_std": 0.01,
      "moving_command_std": "inherited learned Gaussian, unchanged",
      "conditional_distribution_recomputed_in_ppo": True,
      "inference_action_override": False,
    },
    "seed": args.seed, "physical_gpu": args.gpu, "num_envs": args.num_envs,
    "iterations_requested": args.iterations, "save_interval": args.save_interval,
    "learning_rate": cfg.agent.algorithm.learning_rate,
    "tracking_kernel_std": {
      name: cfg.env.rewards[name].params["std"]
      for name in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp")
    },
    "action_std_range": cfg.agent.actor.distribution_cfg.get("std_range"),
    "fall_penalty": cfg.env.rewards["termination"].weight,
    "physics_dt": cfg.env.sim.mujoco.timestep,
    "physics_solver": cfg.env.sim.mujoco.solver,
    "friction_cone": cfg.env.sim.mujoco.cone,
    "decimation": cfg.env.decimation,
    "nan_guard_enabled": cfg.env.sim.nan_guard.enabled,
    "training_symmetry": asdict(cfg.agent.algorithm).get("symmetry_cfg"),
    "source_sha256": hashes,
    "git_revision": subprocess.run(
      ["git", "rev-parse", "HEAD"], cwd=PROJECT, capture_output=True, text=True,
    ).stdout.strip(),
    "disk_budget_bytes": budget, "disk_free_before_bytes": available,
    "started_at": datetime.now(UTC).isoformat(), "status": "initializing",
  }
  (output / "parent_run.json").write_text(json.dumps(parent, indent=2) + "\n")

  def write_manifest():
    (output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")

  env = None
  write_manifest()
  try:
    env = ManagerBasedRlEnv(cfg=cfg.env, device="cuda:0")
    guard_bytes = (
      mujoco.mj_sizeModel(env.sim.mj_model)
      if cfg.env.sim.nan_guard.enabled else 0
    )
    final_budget = output_budget_bytes(
      checkpoint.stat().st_size, args.iterations, args.save_interval, 0,
      guard_model_bytes=guard_bytes,
    )
    free = shutil.disk_usage(output).free
    if free < final_budget:
      raise OSError(
        f"Compiled simulator raises output budget to {final_budget}; free {free}"
      )
    manifest["disk_budget_after_model_bytes"] = final_budget
    manifest["guard_compiled_model_bytes"] = guard_bytes
    wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
    agent_cfg = asdict(cfg.agent)
    dump_yaml(output / "params/env.yaml", asdict(cfg.env))
    dump_yaml(output / "params/agent.yaml", agent_cfg)
    runner = LPACRLQuietStopOnPolicyRunner(
      wrapped, agent_cfg, str(output), "cuda:0"
    )
    source_identity = saved.get("infos", {}).get("lp_acrl", {}).get("task_space")
    if source_identity != runner.command_term.checkpoint_identity():
      raise ValueError("Quiet-stop comparison must preserve the entire v1 environment")
    manifest["refinement_load"] = initialize_refinement(runner, saved, parent)
    manifest["command_training_protocol"] = (
      runner.command_term.command_protocol_identity()
    )
    manifest["status"] = "running"
    write_manifest()
    print(json.dumps({"refinement_load": manifest["refinement_load"]}), flush=True)
    runner.learn(num_learning_iterations=args.iterations, init_at_random_ep_len=False)
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
    if env is not None:
      env.close()


if __name__ == "__main__":
  main()
