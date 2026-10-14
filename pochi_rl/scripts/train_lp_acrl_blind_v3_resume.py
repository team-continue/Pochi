"""Resume clean Blind-v3 training state into a new directory and fresh simulator.

No reward, physics, exploration, symmetry or source adaptation is allowed.
The saved simulator/RNG state does not exist: resets explicitly restart at the
parent seed, while actor/critic/normalizers/optimizer/LP continue exactly.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--gpu", type=int, default=2)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--additional-iterations", type=int)
  parser.add_argument("--smoke", action="store_true")
  parser.add_argument("--nan-guard-dir", type=Path)
  args = parser.parse_args()
  if args.gpu < 0 or args.num_envs < 1:
    parser.error("GPU must be nonnegative and environment count positive")
  if args.smoke and (args.num_envs > 64 or args.additional_iterations not in (1, 2, 3)):
    parser.error("Smoke requires at most64 environments and1–3 additional updates")
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
  os.environ["MUJOCO_GL"] = "egl"
  os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
  os.environ.setdefault("OMP_NUM_THREADS", "4")
  os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
  os.environ.setdefault("WARP_CACHE_PATH", "/dev/shm/pochi_lp_acrl_warp_cache")

  import mujoco
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.scripts.train import TrainConfig
  from mjlab.utils.os import dump_yaml
  from mjlab.utils.torch import configure_torch_backends

  import pochi_rl  # noqa: F401
  from pochi_rl.mjlab.lp_acrl_blind_v3 import register_blind_v3
  from pochi_rl.mjlab.lp_acrl_quiet_stop import LPACRLQuietStopOnPolicyRunner
  from pochi_rl.mjlab.lp_acrl_refinement import output_budget_bytes
  from pochi_rl.mjlab.lp_acrl_resume import (
    TASK_ID,
    resume_training_state,
    sha256,
    validate_resume_sources,
  )

  checkpoint = args.checkpoint.resolve(strict=True)
  parent_path = checkpoint.parent / "run.json"
  parent = json.loads(parent_path.read_text())
  parent_sources = validate_resume_sources(PROJECT, parent)
  if args.seed != parent["seed"] or (
    not args.smoke and args.num_envs != parent["num_envs"]
  ):
    parser.error("Resume must preserve parent seed and environment count")
  saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
  next_iteration = int(saved["iter"]) + 1
  original_target = parent["iterations_requested"]
  updates = args.additional_iterations
  if updates is None:
    updates = original_target - next_iteration
  if updates < 1 or next_iteration + updates > original_target:
    parser.error("Additional updates must remain within the original iteration budget")
  register_blind_v3()
  cfg = TrainConfig.from_task(TASK_ID)
  cfg.env.scene.num_envs = args.num_envs
  cfg.env.seed = cfg.agent.seed = args.seed
  cfg.agent.max_iterations = next_iteration + updates
  cfg.agent.save_interval = parent["save_interval"]
  cfg.agent.resume = False
  cfg.agent.upload_model = False
  cfg.agent.run_name = "resume"
  output = args.output.resolve()
  if output.exists():
    raise FileExistsError(f"Resume output must be new: {output}")
  guard_dir = (
    args.nan_guard_dir or Path("/dev/shm/pochi_lp_acrl_resume_nan_dumps") / output.name
  ).resolve()
  if not guard_dir.is_relative_to(Path("/dev/shm")) or guard_dir.exists():
    raise ValueError("NaN guard requires a new task-owned directory under /dev/shm")
  cfg.env.sim.nan_guard.enabled = True
  cfg.env.sim.nan_guard.buffer_size = 16
  cfg.env.sim.nan_guard.max_envs_to_dump = 2
  cfg.env.sim.nan_guard.output_dir = str(guard_dir)
  additions = {
    "scripts/train_lp_acrl_blind_v3_resume.py",
    "src/pochi_rl/mjlab/lp_acrl_resume.py", "tests/test_lp_acrl_resume.py",
  }
  sources = {**parent_sources, **{
    name: sha256(PROJECT / name) for name in additions
  }}
  source_bytes = sum(
    (PROJECT / name).stat().st_size for name in sources
    if not name.startswith("assets/pochi/meshes/")
  )
  budget = output_budget_bytes(
    checkpoint.stat().st_size, updates, cfg.agent.save_interval, source_bytes,
  )
  ancestor = output.parent
  while not ancestor.exists():
    ancestor = ancestor.parent
  free = shutil.disk_usage(ancestor).free
  if free < budget:
    raise OSError(f"Resume requires estimated {budget} disk bytes; only {free} free")
  output.mkdir(parents=True, exist_ok=False)
  guard_dir.mkdir(parents=True, exist_ok=False)
  for name in sources:
    if name.startswith("assets/pochi/meshes/"):
      continue
    destination = output / "source" / name
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(PROJECT / name, destination)
  shutil.copy2(parent_path, output / "parent_run.json")
  failure_path = checkpoint.parent / "failure.json"
  failure = json.loads(failure_path.read_text()) if failure_path.exists() else None
  if failure_path.exists():
    shutil.copy2(failure_path, output / "parent_failure.json")
  manifest = {
    "paper": parent.get("paper"), "task": TASK_ID,
    "initialization": "full training-state resume; fresh simulator and RNG reset",
    "source_checkpoint": str(checkpoint),
    "source_checkpoint_sha256": sha256(checkpoint),
    "parent_manifest": str(parent_path), "parent_manifest_sha256": sha256(parent_path),
    "parent_task": parent["task"], "parent_initialization": parent["initialization"],
    "parent_status": parent["status"], "source_iteration": int(saved["iter"]),
    "next_iteration": next_iteration,
    "discarded_unsaved_iteration_range": (
      [next_iteration, failure["iteration"]] if failure else None
    ),
    "environment_identity_changes": [],
    "observability_only_change": (
      "NaN guard16 pre-substep frames, own /dev/shm model dump"
    ),
    "simulator_and_rng_checkpoint_available": False,
    "fresh_reset_seed": args.seed, "partial_simulator_episodes_discarded": True,
    "seed": args.seed, "physical_gpu": args.gpu, "num_envs": args.num_envs,
    "smoke_diagnostic": args.smoke,
    "iterations_requested": next_iteration + updates,
    "additional_iterations_requested": updates,
    "original_iteration_budget": original_target,
    "save_interval": cfg.agent.save_interval,
    "source_sha256": sources, "nan_guard_enabled": True,
    "nan_guard_output_dir": str(guard_dir), "nan_guard_history_frames": 16,
    "disk_budget_bytes": budget, "disk_free_before_bytes": free,
    "started_at": datetime.now(UTC).isoformat(), "status": "initializing",
  }
  for name in (
    "physics_dt", "physics_solver", "friction_cone", "decimation",
    "tracking_kernel_std", "fall_penalty", "action_std_range",
    "training_symmetry", "training_exploration", "command_training_protocol",
  ):
    manifest[name] = parent.get(name)

  def write_manifest():
    (output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")

  write_manifest()
  configure_torch_backends()
  env = None
  try:
    env = ManagerBasedRlEnv(cfg=cfg.env, device="cuda:0")
    model_bytes = mujoco.mj_sizeModel(env.sim.mj_model)
    shm_free = shutil.disk_usage(guard_dir).free
    if shm_free < 3 * model_bytes + 128 * 1024**2:
      raise OSError("Insufficient /dev/shm reserve for compiled model and guard dump")
    model_path = guard_dir / "compiled_model.mjb"
    mujoco.mj_saveModel(env.sim.mj_model, str(model_path))
    manifest["compiled_model"] = {
      "path": str(model_path), "bytes": model_path.stat().st_size,
      "sha256": sha256(model_path), "shm_free_before_bytes": shm_free,
    }
    wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
    agent_cfg = asdict(cfg.agent)
    dump_yaml(output / "params/env.yaml", asdict(cfg.env))
    dump_yaml(output / "params/agent.yaml", agent_cfg)
    runner = LPACRLQuietStopOnPolicyRunner(wrapped, agent_cfg, str(output), "cuda:0")
    manifest["resume_load"] = resume_training_state(
      runner, checkpoint, parent, args.seed
    )
    if sha256(checkpoint) != manifest["source_checkpoint_sha256"]:
      raise ValueError("Source checkpoint changed while loading")
    validate_resume_sources(PROJECT, parent)
    manifest["status"] = "running"
    write_manifest()
    print(json.dumps({"resume_load": manifest["resume_load"]}), flush=True)
    runner.learn(num_learning_iterations=updates, init_at_random_ep_len=False)
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
