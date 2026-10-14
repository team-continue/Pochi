"""Independent broad-speed omni phase/lift refinement from accepted V17."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from train_lp_acrl_blind_rough_no_touchdown import (
  archive_backend_sources,
  sha256,
)

ROOT = Path(__file__).resolve().parents[1]


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--init-checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--gpu", type=int, default=3)
  parser.add_argument("--num-envs", type=int, default=4096)
  parser.add_argument("--iterations", type=int, default=2400)
  parser.add_argument("--save-interval", type=int, default=100)
  parser.add_argument("--seed", type=int, default=42)
  args = parser.parse_args()
  if min(args.num_envs, args.iterations, args.save_interval) < 1 or args.seed != 42:
    parser.error("Positive counts and seed42 are required")
  checkpoint = args.init_checkpoint.resolve(strict=True)
  checksum = sha256(checkpoint)
  if checksum != "697e212366d97625c1cfaf19e730e32d735f023c83aa1642a3a02086c97dabd1":
    raise ValueError("Require exact accepted V17/model1199 bytes")
  parent = json.loads((checkpoint.parent / "run.json").read_text())
  if parent.get("task") != "Pochi-LP-ACRL-Blind-v17":
    raise ValueError("Require an accepted V17 parent")
  parent_pins = parent["source_sha256"]
  for name, digest in parent_pins.items():
    if sha256(ROOT / name) != digest:
      raise ValueError(f"Parent source changed: {name}")
  output = args.output.resolve()
  if output.exists():
    raise FileExistsError("Preserve all earlier training runs")
  guard = Path("/dev/shm/pochi_lp_acrl_v18_nan_dumps") / output.name
  if guard.exists():
    raise FileExistsError("Use a new NaN guard directory")
  os.environ.update(
    CUDA_VISIBLE_DEVICES=str(args.gpu),
    MUJOCO_GL="egl",
    MUJOCO_EGL_DEVICE_ID="0",
    PYTHONUNBUFFERED="1",
    TMPDIR="/dev/shm/pochi_lp_acrl_compile_tmp",
    WARP_CACHE_PATH="/dev/shm/pochi_lp_acrl_warp_cache",
    OMP_NUM_THREADS="4",
    OPENBLAS_NUM_THREADS="1",
  )
  import mujoco
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.scripts.train import TrainConfig
  from mjlab.utils.os import dump_yaml
  from mjlab.utils.torch import configure_torch_backends

  from pochi_rl.mjlab.lp_acrl_blind_v18 import register_blind_v18
  from pochi_rl.mjlab.lp_acrl_omni_runner import (
    OmniPhaseGaitRunner,
    validate_omni_checkpoint,
  )
  from pochi_rl.mjlab.lp_acrl_phase_gait import PHASE_CONTRACT
  from pochi_rl.mjlab.warp_contact_backend import (
    contact_backend_source_paths,
    install_contact_backend,
  )

  saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
  backend = install_contact_backend()
  if backend != saved["infos"]["lp_acrl_contact_backend"]:
    raise ValueError("Source/target contact backend mismatch")
  configure_torch_backends()
  register_blind_v18()
  cfg = TrainConfig.from_task("Pochi-LP-ACRL-Blind-v18")
  cfg.env.seed = cfg.agent.seed = args.seed
  cfg.env.scene.num_envs = args.num_envs
  cfg.agent.max_iterations = args.iterations
  cfg.agent.save_interval = args.save_interval
  cfg.agent.upload_model = False
  cfg.env.sim.nan_guard.enabled = True
  cfg.env.sim.nan_guard.buffer_size = 16
  cfg.env.sim.nan_guard.max_envs_to_dump = 2
  cfg.env.sim.nan_guard.output_dir = str(guard)
  paths = [
    *sorted((ROOT / "src/pochi_rl").rglob("*.py")),
    Path(__file__),
    ROOT / "scripts/train_lp_acrl_blind_rough_no_touchdown.py",
    ROOT / "scripts/eval_lp_acrl_omni_phase.py",
    ROOT / "scripts/play_lp_acrl_omni_phase.py",
    ROOT / "tests/test_lp_acrl_omni_phase.py",
    ROOT / "pyproject.toml",
    ROOT / "uv.lock",
    *(ROOT / name for name in parent_pins if name.startswith("assets/")),
  ]
  paths = list(dict.fromkeys(p for p in paths if p.exists()))
  if shutil.disk_usage(output.parent).free < 1024**3:
    raise OSError("Require1GiB disk reserve")
  output.mkdir(parents=True, exist_ok=False)
  guard.mkdir(parents=True, exist_ok=False)
  pins = {}
  for source in paths:
    relative = source.relative_to(ROOT)
    target = output / "source" / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    pins[str(relative)] = sha256(source)
  archived = archive_backend_sources(output, contact_backend_source_paths(), backend)
  manifest = dict(
    task="Pochi-LP-ACRL-Blind-v18",
    initialization=(
      "Accepted50D body/phase weights with new-axis initialization; "
      "fresh optimizer and88-task curriculum"
    ),
    source_checkpoint=str(checkpoint),
    source_checkpoint_sha256=checksum,
    source_sha256=pins,
    contact_backend=backend,
    contact_backend_source_archives=archived,
    phase_guidance=PHASE_CONTRACT,
    command_ranges=dict(vx=(-0.8, 0.8), vy=(-0.5, 0.5), wz=(-1.8, 1.8)),
    command_sampling=(
      "88 signed tasks; pure axes + diagonal/arc/omni; "
      "half equal-mode coverage, half signed LP"
    ),
    reward_changes=dict(
      planar_huber_weight=-12.0,
      continuous_phase_costs=True,
      qualified_min_lift_m=0.04,
      touchdown_bonus=False,
    ),
    seed=args.seed,
    num_envs=args.num_envs,
    physical_gpu=args.gpu,
    iterations_requested=args.iterations,
    save_interval=args.save_interval,
    nan_guard_output_dir=str(guard),
    started_at=datetime.now(UTC).isoformat(),
    status="initializing",
  )

  def write():
    (output / "run.json").write_text(json.dumps(manifest, indent=2) + "\n")

  write()
  env = None
  try:
    env = ManagerBasedRlEnv(cfg.env, device="cuda:0")
    model = guard / "compiled_model.mjb"
    mujoco.mj_saveModel(env.sim.mj_model, str(model))
    manifest["compiled_model"] = dict(path=str(model), sha256=sha256(model))
    wrapped = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)
    runner = OmniPhaseGaitRunner(wrapped, asdict(cfg.agent), str(output), "cuda:0")
    manifest["migration"] = runner.initialize_from_phase(saved)
    dump_yaml(output / "params/env.yaml", asdict(cfg.env))
    dump_yaml(output / "params/agent.yaml", asdict(cfg.agent))
    runner.save(str(output / "initial_state.pt"))
    validate_omni_checkpoint(
      torch.load(output / "initial_state.pt", weights_only=False)
    )
    assert not runner.alg.optimizer.state
    assert sha256(checkpoint) == checksum
    for name, digest in pins.items():
      assert sha256(ROOT / name) == digest, name
    manifest["status"] = "running"
    write()
    print(json.dumps(manifest["migration"]), flush=True)
    runner.learn(args.iterations)
  except BaseException as exc:
    manifest.update(
      status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
      error=f"{type(exc).__name__}: {exc}",
    )
    raise
  else:
    manifest["status"] = "completed"
  finally:
    manifest["finished_at"] = datetime.now(UTC).isoformat()
    write()
    if env is not None:
      env.close()


if __name__ == "__main__":
  main()
