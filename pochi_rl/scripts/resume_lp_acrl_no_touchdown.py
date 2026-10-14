"""Continue the exact V15 objective, including Adam, LR and LP statistics."""

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
  checkpoint = Path(plan["source_checkpoint"])
  if digest(checkpoint) != plan["source_checkpoint_sha256"]:
    raise ValueError("Changed source checkpoint")
  for relative, expected in plan["source_sha256"].items():
    if digest(ROOT / relative) != expected:
      raise ValueError(f"Changed source: {relative}")
  output = args.output.resolve()
  if output.exists():
    raise FileExistsError(output)
  if shutil.disk_usage(output.parent).free < 2 * 1024**3:
    raise OSError("Need 2 GiB reserve for new artifacts")
  os.environ.update(
    CUDA_VISIBLE_DEVICES=str(plan["physical_gpu"]), MUJOCO_GL="egl",
    MUJOCO_EGL_DEVICE_ID="0", OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1",
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

  from pochi_rl.mjlab.lp_acrl_blind_v15 import register_blind_v15
  from pochi_rl.mjlab.lp_acrl_no_touchdown_refinement import (
    LPACRLNoTouchdownOnPolicyRunner,
    validate_no_touchdown_checkpoint,
  )
  from pochi_rl.mjlab.lp_acrl_resume import _equal_tree
  from pochi_rl.mjlab.warp_contact_backend import install_contact_backend

  backend = install_contact_backend()
  saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
  count = validate_no_touchdown_checkpoint(saved)
  if saved["infos"]["lp_acrl_contact_backend"] != backend:
    raise ValueError("Contact backend changed")
  register_blind_v15()
  cfg = TrainConfig.from_task("Pochi-LP-ACRL-Blind-v15")
  cfg.env.scene.num_envs = plan["num_envs"]
  cfg.env.seed = cfg.agent.seed = plan["training_seed"]
  cfg.agent.save_interval = plan["save_interval"]
  cfg.agent.upload_model = False
  guard = Path("/dev/shm/pochi_lp_acrl_v15_resume_nan") / output.name
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
  manifest = {
    "task": "Pochi-LP-ACRL-Blind-v15", "status": "initializing",
    "initialization": "full V15 training-state resume; fresh seeded simulator",
    "source_checkpoint": str(checkpoint),
    "source_checkpoint_sha256": digest(checkpoint),
    "plan_sha256": digest(args.plan), "source_sha256": plan["source_sha256"],
    "contact_backend": backend, "physical_gpu": plan["physical_gpu"],
    "seed": plan["training_seed"], "num_envs": plan["num_envs"],
    "source_completed_updates": count,
    "additional_updates": plan["additional_updates"],
    "simulator_state_and_rng_restored": False,
    "started_at": datetime.now(UTC).isoformat(),
  }

  def write_manifest():
    (output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")

  write_manifest()
  configure_torch_backends()
  env = None
  try:
    env = ManagerBasedRlEnv(cfg=cfg.env, device="cuda:0")
    model = guard / "compiled_model.mjb"
    mujoco.mj_saveModel(env.sim.mj_model, str(model))
    manifest["compiled_model"] = {"path": str(model), "sha256": digest(model)}
    agent = asdict(cfg.agent)
    dump_yaml(output / "params/env.yaml", asdict(cfg.env))
    dump_yaml(output / "params/agent.yaml", agent)
    wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
    runner = LPACRLNoTouchdownOnPolicyRunner(wrapped, agent, str(output), "cuda:0")
    runner.load(str(checkpoint), strict=True, map_location="cuda:0")
    if runner.current_learning_iteration != count:
      raise ValueError("Resume cursor differs")
    initial_path = output / "resume_state.pt"
    runner.save(str(initial_path))
    initial = torch.load(initial_path, map_location="cpu", weights_only=False)
    for key in ("actor_state_dict", "critic_state_dict", "optimizer_state_dict"):
      if not _equal_tree(saved[key], initial[key]):
        raise ValueError(f"Resume changed {key}")
    for key in ("lp_acrl", "lp_acrl_no_touchdown_state", "env_state"):
      if not _equal_tree(saved["infos"][key], initial["infos"][key]):
        raise ValueError(f"Resume changed {key}")
    manifest.update(
      status="running", resume_state_sha256=digest(initial_path),
      actor_critic_normalizers_adam_lr_lp_counter_equal_to_source=True,
    )
    write_manifest()
    runner.learn(plan["additional_updates"], init_at_random_ep_len=False)
    manifest.update(status="completed", completed_updates=runner.completed_updates())
  except BaseException as error:
    manifest.update(
      status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
      error=f"{type(error).__name__}: {error}",
    )
    raise
  finally:
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    write_manifest()
    if env is not None:
      env.close()


if __name__ == "__main__":
  main()
