"""Explicit v3-to-v4 model-only refinement with a recorded contact correction.

Actor, critic and their normalizers initialize from Blind-v3. Optimizer, LP
statistics, simulator and iteration start fresh. This is neither training from
scratch nor full resume. The contact backend change is separately recorded;
the existing refinement helper still requires exact nonreward environment,
symmetry and quiet-stop exploration identities.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
SOURCE_TASK = "Pochi-LP-ACRL-Blind-v3"
TARGET_TASK = "Pochi-LP-ACRL-Blind-v4"
BACKEND_PATCH = "pochi-warp-discrete-epa-epsilon-1e-7-v1"
RAM_ROOT = Path("/dev/shm")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def validate_parent_sources(parent: dict, project: Path = PROJECT) -> dict[str, str]:
  """Check source family, all recorded runtime modules and physical assets."""
  if parent.get("task") != SOURCE_TASK:
    raise ValueError("Qualified refinement requires a Blind-v3 parent")
  sources = {
    name: digest
    for name, digest in parent.get("source_sha256", {}).items()
    if name.startswith(("assets/", "src/pochi_rl/"))
  }
  required = {
    "assets/pochi/pochi.xml",
    "src/pochi_rl/mjlab/lp_acrl_blind_v3_env_cfg.py",
    "src/pochi_rl/mjlab/lp_acrl_quiet_stop.py",
  }
  if not required <= sources.keys() or not any(
    name.startswith("assets/pochi/meshes/") for name in sources
  ):
    raise ValueError("Parent manifest lacks required runtime and physical asset hashes")
  project = project.resolve()
  for name, digest in sources.items():
    path = (project / name).resolve(strict=True)
    if not path.is_relative_to(project) or sha256(path) != digest:
      raise ValueError(f"Parent runtime source or physical asset differs: {name}")
  return sources


def backend_transition(parent: dict, saved: dict, backend: dict) -> dict:
  """Allow only the explicit legacy-v3 to pinned corrected-backend transition."""
  if parent.get("task") != SOURCE_TASK:
    raise ValueError("Contact transition requires a Blind-v3 parent")
  if (saved.get("infos") or {}).get("lp_acrl_contact_backend") is not None:
    raise ValueError("Expected a legacy v3 checkpoint without contact backend metadata")
  if parent.get("contact_backend") is not None:
    raise ValueError("Expected the unpatched legacy parent manifest")
  if (
    backend.get("patch_id") != BACKEND_PATCH
    or backend.get("discrete_epa_epsilon") != 1e-7
    or not backend.get("versions")
    or not backend.get("scope")
  ):
    raise ValueError("Require the approved pinned EPA-epsilon contact backend")
  for field in ("source_sha256", "patched_sha256"):
    digest = backend.get(field)
    if (
      not isinstance(digest, str)
      or len(digest) != 64
      or any(letter not in "0123456789abcdef" for letter in digest)
    ):
      raise ValueError(f"Invalid backend {field}")
  return {
    "source": {
      "kind": "legacy unpatched installed backend",
      "discrete_epa_epsilon": 1e-15,
      "source_sha256": backend["source_sha256"],
      "versions": backend["versions"],
      "checkpoint_backend_metadata_available": False,
      "qualification": (
        "Legacy checkpoint did not save installed backend identity; original "
        "source/version attribution follows the pinned installation and run provenance."
      ),
    },
    "target": backend,
    "purpose": "Explicit model-only refinement, never a silent full-state resume",
    "environment_identity_relaxation": ["environment.rewards"],
    "backend_change_recorded_separately": True,
  }


def archive_backend_sources(
  output: Path, paths: dict[str, Path], identity: dict
) -> dict:
  """Preserve exact original/generated kernels and installer with verified hashes."""
  required = {"installed_collision_gjk", "generated_collision_gjk", "installer"}
  if paths.keys() != required:
    raise ValueError(
      "Contact backend source paths must contain all three named sources"
    )
  for key, field in (
    ("installed_collision_gjk", "source_sha256"),
    ("generated_collision_gjk", "patched_sha256"),
  ):
    if sha256(paths[key]) != identity[field]:
      raise ValueError(f"Contact backend source hash differs: {key}")
  destination = output / "contact_backend_sources"
  destination.mkdir(exist_ok=False)
  records = {}
  for key, source in paths.items():
    target = destination / f"{key}.py"
    shutil.copy2(source, target)
    if sha256(target) != sha256(source):
      raise ValueError(f"Contact backend archive changed while copying: {key}")
    records[key] = {
      "source_path": str(source),
      "archived_path": str(target.relative_to(output)),
      "sha256": sha256(target),
      "bytes": target.stat().st_size,
    }
  return records


def require_ram_path(path: Path, *, new: bool) -> Path:
  resolved = path.resolve()
  if not resolved.is_relative_to(RAM_ROOT) or resolved == RAM_ROOT:
    raise ValueError("Task RAM path must be a directory below /dev/shm")
  if new and resolved.exists():
    raise FileExistsError(f"Task RAM directory must be new: {resolved}")
  return resolved


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--init-checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--gpu", type=int, default=0)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--iterations", type=int, default=1000)
  parser.add_argument("--save-interval", type=int, default=100)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--nan-guard-dir", type=Path)
  parser.add_argument("--smoke", action="store_true")
  args = parser.parse_args()
  if min(args.num_envs, args.iterations, args.save_interval) < 1 or args.gpu < 0:
    parser.error("Counts must be positive and GPU nonnegative")
  if args.smoke and (args.num_envs > 64 or args.iterations > 3):
    parser.error("Smoke requires at most64 environments and1–3 updates")
  checkpoint = args.init_checkpoint.resolve(strict=True)
  parent_path = checkpoint.parent / "run.json"
  parent = json.loads(parent_path.read_text())
  parent_sources = validate_parent_sources(parent)
  output = args.output.resolve()
  if output.exists():
    raise FileExistsError(f"Refinement output must be new: {output}")
  guard_dir = require_ram_path(
    args.nan_guard_dir or RAM_ROOT / "pochi_lp_acrl_v4_nan_dumps" / output.name,
    new=True,
  )
  os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
  os.environ["MUJOCO_GL"] = "egl"
  os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
  os.environ.setdefault("OMP_NUM_THREADS", "4")
  os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
  os.environ.setdefault("TMPDIR", str(RAM_ROOT / "pochi_lp_acrl_compile_tmp"))
  os.environ.setdefault("WARP_CACHE_PATH", str(RAM_ROOT / "pochi_lp_acrl_warp_cache"))
  for name in ("TMPDIR", "WARP_CACHE_PATH"):
    require_ram_path(Path(os.environ[name]), new=False).mkdir(
      parents=True, exist_ok=True
    )

  import mujoco
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.scripts.train import TrainConfig
  from mjlab.utils.os import dump_yaml
  from mjlab.utils.torch import configure_torch_backends

  import pochi_rl  # noqa: F401
  from pochi_rl.mjlab.lp_acrl_blind_v4 import register_blind_v4
  from pochi_rl.mjlab.lp_acrl_contact_runner import LPACRLContactOnPolicyRunner
  from pochi_rl.mjlab.lp_acrl_quiet_refinement import initialize_quiet_refinement
  from pochi_rl.mjlab.lp_acrl_refinement import output_budget_bytes
  from pochi_rl.mjlab.warp_contact_backend import (
    active_contact_backend_identity,
    contact_backend_source_paths,
    install_contact_backend,
  )

  saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
  backend = install_contact_backend(
    cache_dir=RAM_ROOT / "pochi_lp_acrl_contact_backend"
  )
  transition = backend_transition(parent, saved, backend)
  backend_paths = contact_backend_source_paths()
  register_blind_v4()
  cfg = TrainConfig.from_task(TARGET_TASK)
  cfg.env.scene.num_envs = args.num_envs
  cfg.env.seed = cfg.agent.seed = args.seed
  cfg.agent.max_iterations = args.iterations
  cfg.agent.save_interval = args.save_interval
  cfg.agent.resume = False
  cfg.agent.upload_model = False
  cfg.agent.run_name = "qualified_swing_refinement"
  cfg.env.sim.nan_guard.enabled = True
  cfg.env.sim.nan_guard.buffer_size = 16
  cfg.env.sim.nan_guard.max_envs_to_dump = 2
  cfg.env.sim.nan_guard.output_dir = str(guard_dir)
  source_paths = [
    *sorted((PROJECT / "src/pochi_rl").rglob("*.py")),
    Path(__file__),
    PROJECT / "scripts/eval_lp_acrl.py",
    PROJECT / "scripts/eval_lp_acrl_behavior.py",
    PROJECT / "scripts/eval_lp_acrl_behavior_plane.py",
    PROJECT / "pyproject.toml",
    PROJECT / "uv.lock",
    PROJECT / "assets/pochi/pochi.xml",
    PROJECT / "tests/test_lp_acrl_blind_v4.py",
    PROJECT / "tests/test_lp_acrl_blind_v4_registration.py",
    PROJECT / "tests/test_lp_acrl_qualified_refine_launcher.py",
    PROJECT / "tests/test_lp_acrl_qualified_swing_rewards.py",
  ]
  source_paths = [path for path in source_paths if path.exists()]
  source_bytes = sum(p.stat().st_size for p in source_paths) + sum(
    p.stat().st_size for p in backend_paths.values()
  )
  budget = output_budget_bytes(
    checkpoint.stat().st_size,
    args.iterations,
    args.save_interval,
    source_bytes,
  )
  ancestor = output.parent
  while not ancestor.exists():
    ancestor = ancestor.parent
  free = shutil.disk_usage(ancestor).free
  if free < budget:
    raise OSError(f"Refinement needs estimated {budget} disk bytes; only {free} free")
  output.mkdir(parents=True, exist_ok=False)
  guard_dir.mkdir(parents=True, exist_ok=False)
  sources = dict(parent_sources)
  for source in source_paths:
    relative = source.relative_to(PROJECT)
    sources[str(relative)] = sha256(source)
    target = output / "source" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
  archived_backend = archive_backend_sources(output, backend_paths, backend)
  shutil.copy2(parent_path, output / "parent_run.json")
  manifest = {
    "paper": parent.get("paper"),
    "task": TARGET_TASK,
    "initialization": "explicit model-only reward/contact refinement; NOT from scratch",
    "source_checkpoint": str(checkpoint),
    "source_checkpoint_sha256": sha256(checkpoint),
    "parent_manifest": str(parent_path),
    "parent_manifest_sha256": sha256(parent_path),
    "parent_task": parent["task"],
    "parent_status": parent["status"],
    "parent_initialization": parent["initialization"],
    "source_iteration": int(saved["iter"]),
    "allowed_environment_identity_difference": ["environment.rewards"],
    "contact_backend": backend,
    "contact_backend_transition": transition,
    "contact_backend_source_archives": archived_backend,
    "reward_changes": {
      "removed": ["moving_contact_duty_variance"],
      "moving_qualified_swing_occupancy": {
        "weight": -8.0,
        "target": 0.20,
        "tau_s": 1.0,
        "min_air": 0.06,
        "min_lift": 0.025,
      },
    },
    "seed": args.seed,
    "physical_gpu": args.gpu,
    "num_envs": args.num_envs,
    "iterations_requested": args.iterations,
    "save_interval": args.save_interval,
    "smoke_diagnostic": args.smoke,
    "source_sha256": sources,
    "physics_dt": cfg.env.sim.mujoco.timestep,
    "decimation": cfg.env.decimation,
    "physics_solver": cfg.env.sim.mujoco.solver,
    "friction_cone": cfg.env.sim.mujoco.cone,
    "tracking_kernel_std": {
      name: cfg.env.rewards[name].params["std"]
      for name in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp")
    },
    "action_std_range": cfg.agent.actor.distribution_cfg.get("std_range"),
    "fall_penalty": cfg.env.rewards["termination"].weight,
    "training_symmetry": asdict(cfg.agent.algorithm).get("symmetry_cfg"),
    "learning_rate": cfg.agent.algorithm.learning_rate,
    "nan_guard_enabled": True,
    "nan_guard_history_frames": 16,
    "nan_guard_output_dir": str(guard_dir),
    "disk_budget_bytes": budget,
    "disk_free_before_bytes": free,
    "started_at": datetime.now(UTC).isoformat(),
    "status": "initializing",
  }

  def write_manifest():
    (output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")

  write_manifest()
  configure_torch_backends()
  env = None
  try:
    if active_contact_backend_identity() != backend:
      raise ValueError("Contact backend changed before environment creation")
    env = ManagerBasedRlEnv(cfg=cfg.env, device="cuda:0")
    model_bytes = mujoco.mj_sizeModel(env.sim.mj_model)
    shm_free = shutil.disk_usage(guard_dir).free
    if shm_free < 3 * model_bytes + 128 * 1024**2:
      raise OSError("Insufficient RAM reserve for compiled model and16-frame dump")
    model_path = guard_dir / "compiled_model.mjb"
    mujoco.mj_saveModel(env.sim.mj_model, str(model_path))
    manifest["compiled_model"] = {
      "path": str(model_path),
      "bytes": model_path.stat().st_size,
      "sha256": sha256(model_path),
      "shm_free_before_bytes": shm_free,
    }
    wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
    agent_cfg = asdict(cfg.agent)
    dump_yaml(output / "params/env.yaml", asdict(cfg.env))
    dump_yaml(output / "params/agent.yaml", agent_cfg)
    runner = LPACRLContactOnPolicyRunner(wrapped, agent_cfg, str(output), "cuda:0")
    manifest["refinement_load"] = initialize_quiet_refinement(runner, saved, parent)
    manifest["training_exploration"] = runner.exploration_identity()
    manifest["command_training_protocol"] = (
      runner.command_term.command_protocol_identity()
    )
    if sha256(checkpoint) != manifest["source_checkpoint_sha256"]:
      raise ValueError("Source checkpoint changed during model initialization")
    validate_parent_sources(parent)
    if active_contact_backend_identity() != backend:
      raise ValueError("Contact backend changed before training")
    if not math.isfinite(runner.alg.learning_rate) or runner.alg.optimizer.state:
      raise ValueError("Refinement must begin with a fresh finite optimizer")
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
