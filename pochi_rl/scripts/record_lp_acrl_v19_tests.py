"""Capture V19 replay states for the usual matrix video without modifying training."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

os.environ.update(
  CUDA_VISIBLE_DEVICES="3",
  MUJOCO_GL="egl",
  OMP_NUM_THREADS="4",
  OPENBLAS_NUM_THREADS="1",
  TMPDIR="/dev/shm/pochi_lp_acrl_compile_tmp",
  WARP_CACHE_PATH="/dev/shm/pochi_lp_acrl_warp_cache",
)

import eval_lp_acrl_omni_phase as base
import eval_lp_acrl_omni_transitions as transitions
import mujoco
import numpy as np
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer.model_sync import VIEWER_MODEL_FIELDS

from pochi_rl.mjlab.lp_acrl_blind_v19 import TASK_ID, register_blind_v19
from pochi_rl.mjlab.lp_acrl_speed_runner import (
  SpeedPhaseGaitRunner,
  validate_speed_checkpoint,
)
from pochi_rl.mjlab.warp_contact_backend import install_contact_backend


def digest(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
  with Path(path).open("x") as f:
    json.dump(value, f, indent=2, allow_nan=False, default=lambda x: x.item())


def capture(checkpoint, output, suite, profiles, copies, seed):
  base.PROFILE_SEGMENTS = profiles
  base.PROFILES = tuple(profiles)
  raw = {}
  original_summary = base.summarize_phase_profiles

  def summary(samples, names, dt, **kwargs):
    raw.update(samples=samples, names=names, dt=dt)
    return original_summary(samples, names, dt, **kwargs)

  base.summarize_phase_profiles = summary
  cfg = load_env_cfg(TASK_ID, play=True)
  cfg.seed = seed
  cfg.scene.num_envs = len(profiles) * copies
  cfg.auto_reset = False
  cfg.commands["base_velocity"].debug_vis = False
  agent = load_rl_cfg(TASK_ID)
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  qpos, qvel = [], []
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = SpeedPhaseGaitRunner(wrapped, asdict(agent), device="cuda:0")
    runner.load(
      str(checkpoint), load_cfg={"actor": True}, strict=True, map_location="cuda:0"
    )
    fresh = base.FreshContactStepWrapper(env, wrapped)

    class CaptureStep:
      def step(self, actions):
        result = fresh.step(actions)
        qpos.append(env.sim.data.qpos.detach().cpu().numpy().copy())
        qvel.append(env.sim.data.qvel.detach().cpu().numpy().copy())
        return result

    print(f"Capturing {suite}: {env.num_envs} trials, seed {seed}", flush=True)
    rows = base.evaluate_profiles(
      env, CaptureStep(), runner.get_inference_policy(device="cuda:0")
    )
    for row in rows:
      row["steady_passed"] = row["passed"]
      if suite == "in_motion":
        segments = profiles[row["profile"]]
        row["transitions"] = (
          [
            transitions.transition_metrics(
              raw["samples"],
              row["environment_index"],
              segments[j][0],
              segments[j - 1][2],
              segments[j][2],
              raw["dt"],
            )
            for j in range(2, 5)
          ]
          if len(segments) > 1
          else []
        )
        row["passed"] = row["passed"] and all(t["passed"] for t in row["transitions"])
    model_path = output / f"{suite}.mjb"
    mujoco.mj_saveModel(env.sim.mj_model, str(model_path), None)
    fields = {
      name: getattr(env.sim.model, name).detach().cpu().numpy().copy()
      for name in env.sim.expanded_fields & VIEWER_MODEL_FIELDS
    }
    np.savez_compressed(output / f"{suite}.visual_fields.npz", **fields)
    np.savez_compressed(
      output / f"{suite}.states.npz",
      qpos=np.stack(qpos),
      qvel=np.stack(qvel),
      **raw["samples"],
    )
    report = {
      "suite": suite,
      "seed": seed,
      "copies": copies,
      "profiles": profiles,
      "control_dt_s": raw["dt"],
      "names": raw["names"],
      "episodes": rows,
      "model": str(model_path),
      "model_sha256": digest(model_path),
      "states": str(output / f"{suite}.states.npz"),
      "states_sha256": digest(output / f"{suite}.states.npz"),
      "visual_fields": str(output / f"{suite}.visual_fields.npz"),
      "visual_fields_sha256": digest(output / f"{suite}.visual_fields.npz"),
    }
    dump(output / f"{suite}.json", report)
    print(
      f"Captured {suite}: {len(qpos)} states; "
      f"{sum(r['passed'] for r in rows)}/{len(rows)} passed",
      flush=True,
    )
  finally:
    base.summarize_phase_profiles = original_summary
    env.close()


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", required=True, type=Path)
  parser.add_argument("--output-dir", required=True, type=Path)
  parser.add_argument("--source-plan", required=True, type=Path)
  args = parser.parse_args()
  args.output_dir.mkdir(exist_ok=False, parents=True)
  pins = json.loads(args.source_plan.read_text())["source_sha256"]
  for path, value in pins.items():
    assert digest(path) == value, path
  suites = [
    ("original", dict(base.PROFILE_SEGMENTS), 4, 8456),
    ("in_motion", transitions.profiles(), 8, 12456),
  ]
  plan = {
    "protocol": "pochi-v19-all-tests-matrix-recording-v1",
    "task": TASK_ID,
    "physical_gpu": 3,
    "checkpoint": str(args.checkpoint),
    "checkpoint_sha256": digest(args.checkpoint),
    "visualization_replays_not_new_selection": True,
    "representative_rule": "First trial of each profile; never choose a better take",
    "cases": 73,
    "total_trials_captured": 440,
    "suites": [
      {"name": n, "profiles": p, "copies": c, "seed": s} for n, p, c, s in suites
    ],
    "core_source_sha256": pins,
    "capture_source_sha256": digest(__file__),
  }
  dump(args.output_dir / "recording_plan.json", plan)
  (args.output_dir / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
  backend = install_contact_backend()
  completed_updates = validate_speed_checkpoint(
    torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  )
  configure_torch_backends()
  register_blind_v19()
  for name, profiles, copies, seed in suites:
    capture(args.checkpoint, args.output_dir, name, profiles, copies, seed)
  assert digest(args.checkpoint) == plan["checkpoint_sha256"]
  for path, value in pins.items():
    assert digest(path) == value, path
  dump(
    args.output_dir / "capture_complete.json",
    {
      "completed": True,
      "completed_updates": completed_updates,
      "task": TASK_ID,
      "core_sources_unchanged": True,
      "checkpoint_unchanged": True,
      "contact_backend": backend,
    },
  )


if __name__ == "__main__":
  main()
