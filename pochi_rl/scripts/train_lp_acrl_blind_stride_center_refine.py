"""Explicit V10/model400-to-V11 stride-center and stopped-recovery refinement.

Actor, critic and normalizers transfer bitwise, including fixed exploration.
Optimizer, LP and iteration start fresh at the source's actual adaptive LR.
Physics, exploration and all other PPO settings remain unchanged.
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
SOURCE_TASK = "Pochi-LP-ACRL-Blind-v10"
TARGET_TASK = "Pochi-LP-ACRL-Blind-v11"
SOURCE_CHECKPOINT_SHA256 = (
  "57d46865166d719541ebccdd4a56e7ba6096ca85021a597105a92c65dccad646"
)
BACKEND_PATCH = "pochi-warp-discrete-epa-epsilon-1e-7-v1"
RAM_ROOT = Path("/dev/shm")


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


def validate_checkpoint_pin(checkpoint: Path) -> str:
  digest = sha256(checkpoint)
  if digest != SOURCE_CHECKPOINT_SHA256:
    raise ValueError("V11 requires the exact preregistered Blind-v10 model400 bytes")
  return digest


def source_learning_rate(saved: dict) -> float:
  """Use the saved adaptive LR, never the parent's original factory LR."""
  state = (saved.get("infos") or {}).get("lp_acrl_wider_exploration_state") or {}
  groups = (saved.get("optimizer_state_dict") or {}).get("param_groups") or []
  rate = state.get("learning_rate")
  if (
    saved.get("iter") != 400
    or state.get("completed_updates") != 401
    or isinstance(rate, bool)
    or not isinstance(rate, (int, float))
    or not math.isfinite(rate)
    or rate != 7.593750000000004e-05
    or not groups
    or any(group.get("lr") != rate for group in groups)
  ):
    raise ValueError("Pinned source must have matching finite saved adaptive LR")
  return float(rate)


def validate_parent_sources(parent: dict, project: Path = PROJECT) -> dict[str, str]:
  """Check source family, all recorded runtime modules and physical assets."""
  if parent.get("task") != SOURCE_TASK:
    raise ValueError("Stride-center refinement requires a Blind-v10 parent")
  sources = {
    name: digest
    for name, digest in parent.get("source_sha256", {}).items()
    if name.startswith(("assets/", "src/pochi_rl/"))
  }
  required = {
    "assets/pochi/pochi.xml",
    "src/pochi_rl/mjlab/lp_acrl_blind_v10_env_cfg.py",
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
  """Require the unchanged V10 backend without any physics transition."""
  if parent.get("task") != SOURCE_TASK:
    raise ValueError("Stride-center refinement requires a Blind-v10 parent")
  if (
    backend.get("patch_id") != BACKEND_PATCH
    or backend.get("discrete_epa_epsilon") != 1e-7
    or (saved.get("infos") or {}).get("lp_acrl_contact_backend") != backend
    or parent.get("contact_backend") != backend
  ):
    raise ValueError("Source checkpoint, parent manifest and target backend must match")
  return {"source": backend, "target": backend, "unchanged": True}


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
  parser.add_argument("--gpu", type=int, default=1)
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
  checkpoint_sha256 = validate_checkpoint_pin(checkpoint)
  parent_path = checkpoint.parent / "run.json"
  parent_content = parent_path.read_bytes()
  parent = json.loads(parent_content)
  parent_sources = validate_parent_sources(parent)
  if args.seed != parent.get("seed"):
    raise ValueError("Refinement must preserve the registered source seed")
  output = args.output.resolve()
  if output.exists():
    raise FileExistsError(f"Refinement output must be new: {output}")
  guard_dir = require_ram_path(
    args.nan_guard_dir or RAM_ROOT / "pochi_lp_acrl_v11_nan_dumps" / output.name,
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
  from pochi_rl.mjlab.lp_acrl_blind_v11 import register_blind_v11
  from pochi_rl.mjlab.lp_acrl_refinement import output_budget_bytes
  from pochi_rl.mjlab.lp_acrl_stride_center_refinement import (
    LPACRLStrideCenterOnPolicyRunner,
    initialize_stride_center_refinement,
    validate_stride_center_checkpoint,
  )
  from pochi_rl.mjlab.lp_acrl_wider_exploration import (
    validate_wider_exploration_checkpoint,
  )
  from pochi_rl.mjlab.warp_contact_backend import (
    active_contact_backend_identity,
    contact_backend_source_paths,
    install_contact_backend,
  )

  parent_env_path = parent_path.parent / "params/env.yaml"
  parent_env_content = parent_env_path.read_bytes()
  parent_env_sha256 = hashlib.sha256(parent_env_content).hexdigest()
  saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
  if saved.get("iter") != 400:
    raise ValueError("Stride-center refinement requires the pinned Blind-v10 model400")
  validate_wider_exploration_checkpoint(saved)
  initial_learning_rate = source_learning_rate(saved)
  backend = install_contact_backend(
    cache_dir=RAM_ROOT / "pochi_lp_acrl_contact_backend"
  )
  transition = backend_transition(parent, saved, backend)
  backend_paths = contact_backend_source_paths()
  register_blind_v11()
  cfg = TrainConfig.from_task(TARGET_TASK)
  cfg.env.scene.num_envs = args.num_envs
  cfg.env.seed = cfg.agent.seed = args.seed
  cfg.agent.max_iterations = args.iterations
  cfg.agent.save_interval = args.save_interval
  if cfg.agent.algorithm.learning_rate != initial_learning_rate:
    raise ValueError("V11 initial LR must match the source actual adaptive LR")
  cfg.agent.resume = False
  cfg.agent.upload_model = False
  cfg.agent.run_name = "stride_center_refinement"
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
    PROJECT / "tests/test_lp_acrl_blind_v11.py",
    PROJECT / "tests/test_lp_acrl_stride_center_refinement.py",
    PROJECT / "tests/test_lp_acrl_stride_center_launcher.py",
    PROJECT / "tests/test_lp_acrl_stride_center_rewards.py",
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
  ) + math.ceil(checkpoint.stat().st_size * 1.25)  # Initial count0 checkpoint.
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
  (output / "parent_run.json").write_bytes(parent_content)
  (output / "parent_env.yaml").write_bytes(parent_env_content)
  manifest = {
    "paper": parent.get("paper"),
    "task": TARGET_TASK,
    "initialization": (
      "explicit model-only stride-center and stopped-recovery refinement; "
      "NOT from scratch"
    ),
    "source_checkpoint": str(checkpoint),
    "source_checkpoint_sha256": checkpoint_sha256,
    "parent_manifest": str(parent_path),
    "parent_manifest_sha256": hashlib.sha256(parent_content).hexdigest(),
    "parent_task": parent["task"],
    "parent_status": parent["status"],
    "parent_initialization": parent["initialization"],
    "source_iteration": int(saved["iter"]),
    "parent_env_sha256": parent_env_sha256,
    "source_completed_updates": 401,
    "local_initial_iteration": 0,
    "initial_checkpoint_iteration": -1,
    "initial_completed_updates": 0,
    "optimizer_initialization": "fresh Adam; source moments and step counts not loaded",
    "lp_initialization": (
      "fresh uniform curriculum and empty return/partial-episode buffers"
    ),
    "simulator_state_and_rng_restored": False,
    "source_actual_adaptive_learning_rate": initial_learning_rate,
    "allowed_environment_identity_difference": [
      "environment.rewards.moving_common_hip_pitch_deviation",
      "environment.rewards.moving_fore_aft_stride_center",
      "environment.rewards.standing_joint_velocity.func",
    ],
    "contact_backend": backend,
    "contact_backend_transition": transition,
    "contact_backend_source_archives": archived_backend,
    "reward_changes": {
      "moving_common_hip_pitch_deviation": {"operation": "remove"},
      "moving_fore_aft_stride_center": {
        "operation": "add",
        "weight": -36.0,
        "tau_s": 0.25,
        "definition": (
          "mean of top two squared per-foot EMA deviations of fore-aft "
          "foot-minus-whole-COM offsets from nominal geometry"
        ),
      },
      "standing_joint_velocity": {
        "operation": "multiply unchanged bounded cost by all-four-contact gate",
        "weight": -0.25,
        "existing_params_unchanged": True,
      },
      "all_other_rewards_unchanged": True,
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
    "ppo_configuration_changes": {
      "initial_learning_rate": {
        "parent_run_initial_learning_rate": parent.get("learning_rate"),
        "to_source_actual_adaptive_lr": initial_learning_rate,
      },
      "all_other_ppo_settings_unchanged": True,
    },
    "exploration_observation_symmetry_unchanged": True,
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
    runner = LPACRLStrideCenterOnPolicyRunner(wrapped, agent_cfg, str(output), "cuda:0")
    manifest["refinement_load"] = initialize_stride_center_refinement(
      runner, saved, parent
    )
    manifest["training_exploration"] = runner.exploration_identity()
    manifest["observation_corruption"] = runner.observation_corruption_identity()
    manifest["command_training_protocol"] = (
      runner.command_term.command_protocol_identity()
    )
    if sha256(checkpoint) != manifest["source_checkpoint_sha256"]:
      raise ValueError("Source checkpoint changed during model initialization")
    validate_parent_sources(parent)
    if sha256(parent_env_path) != parent_env_sha256:
      raise ValueError("Parent environment configuration changed before training")
    if active_contact_backend_identity() != backend:
      raise ValueError("Contact backend changed before training")
    if not math.isfinite(runner.alg.learning_rate) or runner.alg.optimizer.state:
      raise ValueError("Refinement must begin with a fresh finite optimizer")
    initial_path = output / "initial_state.pt"
    runner.save(str(initial_path))
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    if validate_stride_center_checkpoint(initial) != 0 or initial["iter"] != -1:
      raise ValueError("Initial refinement checkpoint must have count0 and iter-1")
    manifest["initial_checkpoint"] = {
      "path": str(initial_path),
      "sha256": sha256(initial_path),
      "completed_updates": 0,
      "iteration": -1,
    }
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
