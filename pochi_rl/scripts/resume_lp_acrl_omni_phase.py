"""Resume V18 training state on the sole authorized GPU, preserving old runs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--plan", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  plan = json.loads(args.plan.read_text())
  if plan["physical_gpu"] != 3:
    raise ValueError("Only physical GPU3 is authorized")
  checkpoint = Path(plan["source_checkpoint"])
  if digest(checkpoint) != plan["source_checkpoint_sha256"]:
    raise ValueError("Source checkpoint changed")
  for relative, expected in plan["source_sha256"].items():
    if digest(ROOT / relative) != expected:
      raise ValueError(f"Source or robot asset changed: {relative}")
  output = args.output.resolve()
  if output.exists():
    raise FileExistsError(output)
  if shutil.disk_usage(output.parent).free < 2 * 1024**3:
    raise OSError("Need2GiB artifact reserve")
  os.environ.update(
    CUDA_VISIBLE_DEVICES="3",
    MUJOCO_GL="egl",
    MUJOCO_EGL_DEVICE_ID="0",
    OMP_NUM_THREADS="4",
    OPENBLAS_NUM_THREADS="1",
    PYTHONUNBUFFERED="1",
    TMPDIR="/dev/shm/pochi_lp_acrl_compile_tmp",
    WARP_CACHE_PATH="/dev/shm/pochi_lp_acrl_warp_cache",
  )
  import mujoco
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.scripts.train import TrainConfig
  from mjlab.utils.os import dump_yaml
  from mjlab.utils.torch import configure_torch_backends
  from train_lp_acrl_blind_rough_no_touchdown import archive_backend_sources

  from pochi_rl.mjlab.lp_acrl_blind_v18 import register_blind_v18
  from pochi_rl.mjlab.lp_acrl_omni_runner import (
    STATE_KEY,
    OmniPhaseGaitRunner,
    validate_omni_checkpoint,
  )
  from pochi_rl.mjlab.lp_acrl_resume import _equal_tree
  from pochi_rl.mjlab.warp_contact_backend import (
    contact_backend_source_paths,
    install_contact_backend,
  )

  saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
  count = validate_omni_checkpoint(saved)
  if count != plan["source_completed_updates"]:
    raise ValueError("Resume count changed")
  if count + plan["additional_updates"] != plan["total_completed_updates"]:
    raise ValueError("Incorrect remaining update count")
  backend = install_contact_backend()
  if saved["infos"]["lp_acrl_contact_backend"] != backend:
    raise ValueError("Contact backend changed")
  configure_torch_backends()
  register_blind_v18()
  cfg = TrainConfig.from_task("Pochi-LP-ACRL-Blind-v18")
  cfg.env.seed = cfg.agent.seed = plan["training_seed"]
  cfg.env.scene.num_envs = plan["num_envs"]
  cfg.agent.max_iterations = plan["total_completed_updates"]
  cfg.agent.save_interval = plan["save_interval"]
  cfg.agent.upload_model = False
  guard = Path("/dev/shm/pochi_lp_acrl_v18_resume_nan") / output.name
  guard.mkdir(parents=True, exist_ok=False)
  cfg.env.sim.nan_guard.enabled = True
  cfg.env.sim.nan_guard.buffer_size = 16
  cfg.env.sim.nan_guard.max_envs_to_dump = 2
  cfg.env.sim.nan_guard.output_dir = str(guard)
  output.mkdir(exist_ok=False)
  shutil.copy2(args.plan, output / "continuation_plan.json")
  for relative in plan["source_sha256"]:
    target = output / "source" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / relative, target)
  archived = archive_backend_sources(output, contact_backend_source_paths(), backend)
  manifest = {
    "task": "Pochi-LP-ACRL-Blind-v18",
    "status": "initializing",
    "initialization": "Full V18 training-state resume; fresh seeded simulator",
    "source_checkpoint": str(checkpoint),
    "source_checkpoint_sha256": digest(checkpoint),
    "plan_sha256": digest(args.plan),
    "source_sha256": plan["source_sha256"],
    "contact_backend": backend,
    "contact_backend_source_archives": archived,
    "physical_gpu": 3,
    "seed": plan["training_seed"],
    "num_envs": plan["num_envs"],
    "source_completed_updates": count,
    "additional_updates": plan["additional_updates"],
    "total_completed_updates": plan["total_completed_updates"],
    "simulator_state_and_rng_restored": False,
    "started_at": datetime.now(UTC).isoformat(),
  }

  def write():
    (output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")

  write()
  env = None
  try:
    env = ManagerBasedRlEnv(cfg.env, device="cuda:0")
    model = guard / "compiled_model.mjb"
    mujoco.mj_saveModel(env.sim.mj_model, str(model))
    manifest["compiled_model"] = {"path": str(model), "sha256": digest(model)}
    dump_yaml(output / "params/env.yaml", asdict(cfg.env))
    dump_yaml(output / "params/agent.yaml", asdict(cfg.agent))
    wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
    runner = OmniPhaseGaitRunner(wrapped, asdict(cfg.agent), str(output), "cuda:0")
    runner.load(str(checkpoint), strict=True, map_location="cuda:0")
    if runner.current_learning_iteration != count:
      raise ValueError("Resume cursor differs")
    restored_path = output / "resume_state.pt"
    runner.save(str(restored_path))
    restored = torch.load(restored_path, map_location="cpu", weights_only=False)
    if validate_omni_checkpoint(restored) != count:
      raise ValueError("Restored checkpoint contract differs")
    for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
      if not _equal_tree(saved[key], restored[key]):
        raise ValueError(f"Resume changed {key}")
    for key in ("lp_acrl", STATE_KEY, "env_state"):
      if not _equal_tree(saved["infos"][key], restored["infos"][key]):
        raise ValueError(f"Resume changed {key}")
    for name, expected in plan["source_sha256"].items():
      if digest(ROOT / name) != expected:
        raise ValueError(f"Source changed during resume: {name}")
    manifest.update(
      status="running",
      resume_state_sha256=digest(restored_path),
      actor_critic_normalizers_adam_lr_lp_counter_equal_to_source=True,
    )
    write()
    print(
      "Full training-state restore audited; remaining updates:",
      plan["additional_updates"],
      flush=True,
    )
    runner.learn(plan["additional_updates"], init_at_random_ep_len=False)
    final = output / f"model_{plan['total_completed_updates'] - 1}.pt"
    actual_count = validate_omni_checkpoint(
      torch.load(final, map_location="cpu", weights_only=False)
    )
    if actual_count != plan["total_completed_updates"]:
      raise ValueError("Final update count differs")
    manifest.update(status="completed", completed_updates=actual_count)
  except BaseException as error:
    manifest.update(
      status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
      error=f"{type(error).__name__}: {error}",
    )
    raise
  finally:
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    write()
    if env is not None:
      env.close()


if __name__ == "__main__":
  main()
