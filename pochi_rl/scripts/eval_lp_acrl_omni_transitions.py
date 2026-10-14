"""Additional V18 diagnosis: retain forward/backward speed while switching vy/wz."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np


def profiles():
  result = {"stand": ((0.0, 20.0, (0.0, 0.0, 0.0)),)}
  for vx in (0.25, 0.5, -0.25, -0.5):
    direction = "forward" if vx > 0 else "backward"
    prefix = f"{direction}_{abs(vx):g}"
    straight = (vx, 0.0, 0.0)
    result[f"{prefix}_baseline"] = ((0.0, 20.0, straight),)
    for mode in ("lateral", "yaw", "co_turn", "counter_turn"):
      for sign in (1, -1):
        vy = sign * 0.2 if mode != "yaw" else 0.0
        wz = sign * 0.6 if mode != "lateral" else 0.0
        if mode == "counter_turn":
          wz = -wz
        side = "left_first" if sign > 0 else "right_first"
        result[f"{prefix}_{mode}_{side}"] = (
          (0.0, 2.0, (0.0, 0.0, 0.0)),
          (2.0, 6.0, straight),
          (6.0, 10.0, (vx, vy, wz)),
          (10.0, 14.0, (vx, -vy, -wz)),
          (14.0, 18.0, straight),
          (18.0, 20.0, (0.0, 0.0, 0.0)),
        )
  return result


def transition_metrics(samples, index, start, before, after, dt):
  """Score retained vx through the first second and changed-axis response."""
  select = (samples["time_s"] > start + 1e-9) & (
    samples["time_s"] <= start + 1.0 + 1e-9
  )
  velocity = samples["velocity"][select, index]
  valid = samples["valid"][select, index]
  expected = round(1.0 / dt)
  if len(velocity) != expected or not valid.all():
    return {"start_s": start, "complete": False, "passed": False}
  command = np.asarray(after)
  changed = np.flatnonzero(np.abs(np.asarray(before) - command) > 1e-6)
  assert command[0] == before[0] and abs(command[0]) > 0
  tolerance = np.maximum([0.05, 0.05, 0.10], 0.15 * np.abs(command))
  forward_error = float(np.abs(velocity[:, 0] - command[0]).mean())
  retained = float(
    (np.sign(command[0]) * velocity[:, 0] >= 0.25 * abs(command[0])).mean()
  )
  window = round(0.5 / dt)
  settling = None
  for end in range(window, len(velocity) + 1):
    error = np.abs(velocity[end - window : end, changed] - command[changed]).mean(0)
    if bool((error <= tolerance[changed]).all()):
      settling = end * dt
      break
  checks = {
    "vx_error_first_second": bool(forward_error <= tolerance[0]),
    "retain_direction_at_least90pct": retained >= 0.90,
    "changed_axes_500ms_error_window_within1s": settling is not None,
    "tilt_peak_under20deg": float(samples["tilt_deg"][select, index].max()) <= 20.0,
    "no_nonfoot_contact_during_switch": not bool(
      samples["nonfoot_contact"][select, index].any()
    ),
    "no_whole_body_flight_during_switch": bool(
      samples["contact"][select, index].any(-1).all()
    ),
    "contact_slip_during_switch": float(samples["slip"][select, index].mean()) <= 0.10,
  }
  return {
    "start_s": start,
    "complete": True,
    "before": before,
    "after": after,
    "vx_error_first_second_m_s": forward_error,
    "retained_direction_fraction": retained,
    "changed_axes": changed.tolist(),
    "first_qualified_500ms_window_end_s_after_switch": settling,
    "checks": checks,
    "passed": all(checks.values()),
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--source-plan", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists() or args.output.with_suffix(".samples.npz").exists():
    parser.error("Preserve earlier outcomes")
  os.environ.update(
    CUDA_VISIBLE_DEVICES="3",
    MUJOCO_GL="egl",
    MUJOCO_EGL_DEVICE_ID="0",
    OMP_NUM_THREADS="4",
    OPENBLAS_NUM_THREADS="1",
    TMPDIR="/dev/shm/pochi_lp_acrl_compile_tmp",
    WARP_CACHE_PATH="/dev/shm/pochi_lp_acrl_warp_cache",
  )
  import eval_lp_acrl_omni_phase as base
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.torch import configure_torch_backends

  from pochi_rl.mjlab.lp_acrl_blind_v18 import TASK_ID, register_blind_v18
  from pochi_rl.mjlab.lp_acrl_omni_runner import (
    OmniPhaseGaitRunner,
    validate_omni_checkpoint,
  )
  from pochi_rl.mjlab.warp_contact_backend import install_contact_backend

  specs = profiles()
  copies = 8
  seed = 12456

  def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

  checkpoint_sha = sha(args.checkpoint)
  frozen_sources = json.loads(args.source_plan.read_text())["source_sha256"]
  for name, digest in frozen_sources.items():
    assert sha(name) == digest, f"Frozen source changed: {name}"
  plan = {
    "protocol": "pochi-omni-in-motion-switch-diagnostic-v1",
    "diagnostic_only_not_original_selection": True,
    "checkpoint": str(args.checkpoint.resolve()),
    "checkpoint_sha256": checkpoint_sha,
    "seed": seed,
    "physical_gpu": 3,
    "copies_each_profile": copies,
    "num_envs": len(specs) * copies,
    "profiles": specs,
    "coordinate_convention": (
      "body-frame vx/vy/wz; vx remains nonzero and constant "
      "from 2s through 18s in switch profiles"
    ),
    "steady_criteria": (
      "Unchanged V18 segment criteria; exclude 1s after dynamic segment start "
      "and 2s for pure-axis baselines"
    ),
    "transient_criteria": (
      "For switches 6/10/14s, score all first 1s: vx MAE<=max(.05,15%|vx|); "
      "retain signed vx>=25% target at least 90%; changed axes must meet "
      "original error bounds over some complete 500ms window ending<=1s. "
      "No fall/invalid, no nonfoot contact/flight, tilt<=20deg, slip<=.10m/s."
    ),
    "no_parameter_updates_or_action_override": True,
    "original_source_plan": str(args.source_plan.resolve()),
    "original_source_plan_sha256": sha(args.source_plan),
    "source_sha256": {
      **{str(Path(p).resolve()): h for p, h in frozen_sources.items()},
      str(args.source_plan.resolve()): sha(args.source_plan),
      str(Path(__file__).resolve()): sha(__file__),
      str(Path(base.__file__).resolve()): sha(base.__file__),
    },
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.with_suffix(".plan.json").open("x") as f:
    json.dump(plan, f, indent=2)
  # The frozen collector accepts configurable global profiles in this process.
  # Capture its raw samples at the existing summary boundary, without changing
  # collection, control, physics, policy output, or any frozen source file.
  base.PROFILE_SEGMENTS = specs
  base.PROFILES = tuple(specs)
  original_summary = base.summarize_phase_profiles
  captured = {}

  def summarize(samples, names, dt, **kwargs):
    captured.update(samples=samples, names=names, dt=dt)
    return original_summary(samples, names, dt, **kwargs)

  base.summarize_phase_profiles = summarize
  backend = install_contact_backend()
  count = validate_omni_checkpoint(
    torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  )
  configure_torch_backends()
  register_blind_v18()
  cfg = load_env_cfg(TASK_ID, play=True)
  cfg.seed = seed
  cfg.scene.num_envs = plan["num_envs"]
  cfg.auto_reset = False
  cfg.commands["base_velocity"].debug_vis = False
  agent = load_rl_cfg(TASK_ID)
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = OmniPhaseGaitRunner(wrapped, asdict(agent), device="cuda:0")
    runner.load(
      str(args.checkpoint), load_cfg={"actor": True}, strict=True, map_location="cuda:0"
    )
    rows = base.evaluate_profiles(
      env,
      base.FreshContactStepWrapper(env, wrapped),
      runner.get_inference_policy(device="cuda:0"),
    )
    for row in rows:
      segments = specs[row["profile"]]
      changes = (
        [
          transition_metrics(
            captured["samples"],
            row["environment_index"],
            segments[j][0],
            segments[j - 1][2],
            segments[j][2],
            captured["dt"],
          )
          for j in range(2, 5)
        ]
        if len(segments) > 1
        else []
      )
      row["steady_passed"] = row["passed"]
      row["transitions"] = changes
      row["passed"] = row["passed"] and all(e["passed"] for e in changes)
    summary = {}
    for name in specs:
      group = [r for r in rows if r["profile"] == name]
      summary[name] = {
        "episodes": len(group),
        "passed": sum(r["passed"] for r in group),
        "steady_passed": sum(r["steady_passed"] for r in group),
        "switches_passed": sum(
          all(t["passed"] for t in r["transitions"]) for r in group
        ),
        "falls": sum(r["fall"] for r in group),
        "invalid": sum(r["invalid_state"] for r in group),
      }
    samples_path = args.output.with_suffix(".samples.npz")
    np.savez_compressed(samples_path, **captured["samples"])
    assert sha(args.checkpoint) == checkpoint_sha
    for name, digest in plan["source_sha256"].items():
      assert sha(name) == digest
    report = {
      **plan,
      "completed_updates": count,
      "contact_backend": backend,
      "control_dt_s": env.step_dt,
      "samples": str(samples_path),
      "samples_sha256": sha(samples_path),
      "summary": summary,
      "episodes": rows,
    }
    with args.output.open("x") as f:
      json.dump(report, f, indent=2, allow_nan=False)
    print(json.dumps({"output": str(args.output), "summary": summary}), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
