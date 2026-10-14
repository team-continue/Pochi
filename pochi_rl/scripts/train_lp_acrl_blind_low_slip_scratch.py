"""Fresh random v9 training with annealed exploration and recorded provenance.

There is intentionally no load, resume, or initialization-checkpoint argument.
The v8 environment changes only contact_slip weight from -8 to -0.2.
Model, normalizers, optimizer and LP statistics begin from new state with the
unchanged v7 exploration schedule. This engineering run changes two rewards
relative to v7; it is not a single-factor comparison against v7. Contact
diagnostics and compilation stay in RAM.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
TASK = "Pochi-LP-ACRL-Blind-v9"
RAM_ROOT = Path("/dev/shm")
CHECKPOINT_BUDGET_BYTES = 4 * 1024**2


def sha256(path: Path) -> str:
  digest = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      digest.update(block)
  return digest.hexdigest()


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


def archive_robot_assets(project: Path, temporary_dir: Path) -> tuple[Path, dict]:
  """Create a verified lossless XML/mesh archive in RAM before budgeting output."""
  assets = sorted(
    path
    for path in (project / "assets/pochi").rglob("*")
    if path.is_file() and path.suffix in (".xml", ".obj")
  )
  if project / "assets/pochi/pochi.xml" not in assets or not any(
    path.suffix == ".obj" for path in assets
  ):
    raise ValueError("Scratch source must include robot XML and OBJ meshes")
  for path in assets:
    if not path.resolve().is_relative_to(project.resolve()):
      raise ValueError("Robot asset path escapes the project")
  hashes = {str(path.relative_to(project)): sha256(path) for path in assets}
  with tempfile.NamedTemporaryFile(
    dir=temporary_dir, prefix="pochi_v9_assets_", suffix=".tar.gz", delete=False
  ) as stream:
    archive = Path(stream.name)
  try:
    # Stable gzip headers permit verified byte-identical archive deduplication.
    with archive.open("wb") as raw:
      with gzip.GzipFile(
        fileobj=raw, mode="wb", filename="", mtime=0, compresslevel=6
      ) as compressed:
        with tarfile.open(fileobj=compressed, mode="w") as bundle:
          for path in assets:
            bundle.add(path, arcname=str(path.relative_to(project)), recursive=False)
    # Verify the actual archived bytes, not only the source before/after copying.
    with tarfile.open(archive, "r:gz") as bundle:
      for member in bundle.getmembers():
        if not member.isfile():
          raise ValueError("Robot archive must contain only regular source files")
        stream = bundle.extractfile(member)
        if stream is None:
          raise ValueError("Robot archive must contain only regular source files")
        digest = hashlib.sha256()
        for block in iter(lambda stream=stream: stream.read(1024 * 1024), b""):
          digest.update(block)
        if digest.hexdigest() != hashes[member.name]:
          raise ValueError("Robot asset changed while archiving")
    if any(sha256(project / name) != digest for name, digest in hashes.items()):
      raise ValueError("Robot asset changed after archiving")
  except BaseException:
    archive.unlink(missing_ok=True)
    raise
  return archive, hashes


def assert_fresh_scratch(runner) -> dict:
  """Check empty learning state and genuinely unused observation normalizers."""
  import torch

  from pochi_rl.mjlab.lp_acrl_refinement import _assert_fresh_training_state

  _assert_fresh_training_state(runner)
  if runner.alg.learning_rate != 1e-3 or any(
    group["lr"] != 1e-3 for group in runner.alg.optimizer.param_groups
  ):
    raise ValueError("Scratch requires initial adaptive learning rate1e-3")
  schedule = runner.alg.schedule_state()
  if schedule["completed_updates"] != 0 or schedule["next_moving_std"] != 0.35:
    raise ValueError("Scratch requires the initial exploration schedule")
  if not schedule["training_resume_allowed"]:
    raise ValueError("Scratch cannot begin from an inference-only actor")
  model_hashes = {}
  for name, model in (
    ("actor", runner.alg._raw_actor),
    ("critic", runner.alg._raw_critic),
  ):
    normalizer = model.obs_normalizer
    if (
      normalizer.count.item() != 0
      or normalizer._mean.any()
      or not torch.equal(normalizer._var, torch.ones_like(normalizer._var))
      or not torch.equal(normalizer._std, torch.ones_like(normalizer._std))
    ):
      raise ValueError("Scratch requires unused mean0/variance1 normalizers")
    digest = hashlib.sha256()
    for key, value in model.state_dict().items():
      if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError("Scratch model contains nonfinite initial state")
      digest.update(key.encode())
      digest.update(str(value.shape).encode())
      digest.update(str(value.dtype).encode())
      digest.update(value.detach().cpu().numpy().tobytes())
    model_hashes[name] = digest.hexdigest()
  return {
    "checkpoint_loaded": False,
    "initial_iteration": 0,
    "initial_optimizer_state_entries": 0,
    "initial_lp_stage": 0,
    "initial_lp_steps": 0,
    "initial_lp_episode_count": 0,
    "initial_actor_critic_normalizer_count": 0,
    "initial_model_state_sha256": model_hashes,
    "initial_exploration_schedule": schedule,
  }


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--gpu", type=int, default=3)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--iterations", type=int, default=1500)
  parser.add_argument("--save-interval", type=int, default=100)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--nan-guard-dir", type=Path)
  parser.add_argument("--smoke", action="store_true")
  args = parser.parse_args()
  if min(args.num_envs, args.iterations, args.save_interval) < 1 or args.gpu < 0:
    parser.error("Counts must be positive and GPU nonnegative")
  if args.smoke and (args.num_envs > 64 or args.iterations > 3):
    parser.error("Smoke requires at most64 environments and1–3 updates")
  output = args.output.resolve()
  if output.exists():
    raise FileExistsError(f"Scratch output must be new: {output}")
  guard_dir = require_ram_path(
    args.nan_guard_dir or RAM_ROOT / "pochi_lp_acrl_v9_nan_dumps" / output.name,
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
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.scripts.train import TrainConfig
  from mjlab.utils.os import dump_yaml
  from mjlab.utils.torch import configure_torch_backends

  import pochi_rl  # noqa: F401
  from pochi_rl.mjlab.lp_acrl_annealed import (
    LPACRLAnnealedOnPolicyRunner,
    schedule_identity,
  )
  from pochi_rl.mjlab.lp_acrl_blind_v9 import register_blind_v9
  from pochi_rl.mjlab.lp_acrl_refinement import output_budget_bytes
  from pochi_rl.mjlab.warp_contact_backend import (
    active_contact_backend_identity,
    contact_backend_source_paths,
    install_contact_backend,
  )

  backend = install_contact_backend(
    cache_dir=RAM_ROOT / "pochi_lp_acrl_contact_backend"
  )
  backend_paths = contact_backend_source_paths()
  register_blind_v9()
  cfg = TrainConfig.from_task(TASK)
  cfg.env.scene.num_envs = args.num_envs
  cfg.env.seed = cfg.agent.seed = args.seed
  cfg.agent.max_iterations = args.iterations
  cfg.agent.save_interval = args.save_interval
  cfg.agent.resume = False
  cfg.agent.upload_model = False
  cfg.agent.run_name = "low_slip_annealed_scratch"
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
    PROJECT / "tests/test_lp_acrl_blind_v9.py",
    PROJECT / "tests/test_lp_acrl_low_slip_scratch_launcher.py",
    PROJECT / "tests/test_lp_acrl_annealed.py",
  ]
  source_paths = [path for path in source_paths if path.exists()]
  archive, asset_hashes = archive_robot_assets(PROJECT, Path(os.environ["TMPDIR"]))
  try:
    source_bytes = (
      sum(p.stat().st_size for p in source_paths)
      + sum(p.stat().st_size for p in backend_paths.values())
      + archive.stat().st_size
    )
    budget = output_budget_bytes(
      CHECKPOINT_BUDGET_BYTES, args.iterations, args.save_interval, source_bytes
    )
    ancestor = output.parent
    while not ancestor.exists():
      ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    if free < budget:
      raise OSError(f"Scratch needs estimated {budget} disk bytes; only {free} free")
    output.mkdir(parents=True, exist_ok=False)
    guard_dir.mkdir(parents=True, exist_ok=False)
    hashes = dict(asset_hashes)
    for source in source_paths:
      relative = source.relative_to(PROJECT)
      hashes[str(relative)] = sha256(source)
      target = output / "source" / relative
      target.parent.mkdir(parents=True, exist_ok=True)
      shutil.copy2(source, target)
      if sha256(target) != hashes[str(relative)]:
        raise ValueError("Runtime source changed while snapshotting")
    archive_target = output / "robot_assets.tar.gz"
    shutil.copy2(archive, archive_target)
    if sha256(archive_target) != sha256(archive):
      raise ValueError("Robot archive changed while copying")
    archived_backend = archive_backend_sources(output, backend_paths, backend)
  finally:
    archive.unlink(missing_ok=True)
  manifest = {
    "paper": "https://arxiv.org/abs/2601.17428",
    "task": TASK,
    "initialization": "fresh random actor and critic; no checkpoint loaded",
    "checkpoint_loaded": False,
    "engineering_comparison": {
      "environment_base": "Pochi-LP-ACRL-Blind-v8",
      "environment_difference": {"rewards.contact_slip.weight": [-8.0, -0.2]},
      "algorithm_base": "Pochi-LP-ACRL-Blind-v7",
      "algorithm_unchanged": True,
      "single_factor_comparison_against_v7": False,
      "v7_reward_differences": [
        "replace moving_default_joint_deviation with common-hip-pitch cost",
        "contact_slip weight -8 to -0.2",
      ],
    },
    "seed": args.seed,
    "physical_gpu": args.gpu,
    "num_envs": args.num_envs,
    "iterations_requested": args.iterations,
    "save_interval": args.save_interval,
    "smoke_diagnostic": args.smoke,
    "source_sha256": hashes,
    "robot_asset_archive": {
      "path": "robot_assets.tar.gz",
      "sha256": sha256(archive_target),
      "bytes": archive_target.stat().st_size,
      "file_count": len(asset_hashes),
    },
    "contact_backend": backend,
    "contact_backend_source_archives": archived_backend,
    "exploration_schedule": schedule_identity(),
    "physics_dt": cfg.env.sim.mujoco.timestep,
    "decimation": cfg.env.decimation,
    "physics_solver": cfg.env.sim.mujoco.solver,
    "friction_cone": cfg.env.sim.mujoco.cone,
    "tracking_kernel_std": {
      name: cfg.env.rewards[name].params["std"]
      for name in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp")
    },
    "fall_penalty": cfg.env.rewards["termination"].weight,
    "training_symmetry": asdict(cfg.agent.algorithm).get("symmetry_cfg"),
    "learning_rate": cfg.agent.algorithm.learning_rate,
    "nan_guard_enabled": True,
    "nan_guard_history_frames": 16,
    "nan_guard_output_dir": str(guard_dir),
    "disk_budget_bytes": budget,
    "disk_free_before_bytes": free,
    "checkpoint_size_budget_bytes": CHECKPOINT_BUDGET_BYTES,
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
    runner = LPACRLAnnealedOnPolicyRunner(wrapped, agent_cfg, str(output), "cuda:0")
    manifest["scratch_initial_state"] = assert_fresh_scratch(runner)
    manifest["training_exploration"] = runner.exploration_identity()
    manifest["observation_corruption"] = runner.observation_corruption_identity()
    manifest["command_training_protocol"] = (
      runner.command_term.command_protocol_identity()
    )
    for name, digest in hashes.items():
      if sha256(PROJECT / name) != digest:
        raise ValueError(f"Runtime source or asset changed before training: {name}")
    if active_contact_backend_identity() != backend:
      raise ValueError("Contact backend changed before training")
    manifest["status"] = "running"
    write_manifest()
    print(
      json.dumps({"scratch_initial_state": manifest["scratch_initial_state"]}),
      flush=True,
    )
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
