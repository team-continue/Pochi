"""Diagnostic flat omni videos; never an acceptance or fallback selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

TASK = "Pochi-LP-ACRL-Blind-v18"
TASK_ID = 0
FPS = 25
PROTOCOL = "pochi-omni-phase-diagnostic-preview-v1"
SCENARIOS = {
  "translations": (
    ("initial_stop", 0.0, 2.0, (0.0, 0.0, 0.0)),
    ("forward", 2.0, 6.0, (0.5, 0.0, 0.0)),
    ("left", 6.0, 10.0, (0.0, 0.3, 0.0)),
    ("backward", 10.0, 14.0, (-0.5, 0.0, 0.0)),
    ("right", 14.0, 18.0, (0.0, -0.3, 0.0)),
    ("final_stop", 18.0, 20.0, (0.0, 0.0, 0.0)),
  ),
  "rotation_combined": (
    ("initial_stop", 0.0, 2.0, (0.0, 0.0, 0.0)),
    ("counterclockwise", 2.0, 6.0, (0.0, 0.0, 0.9)),
    ("clockwise", 6.0, 10.0, (0.0, 0.0, -0.9)),
    ("diagonal", 10.0, 14.0, (0.4, 0.25, 0.0)),
    ("arc", 14.0, 18.0, (0.5, 0.0, 0.9)),
    ("final_stop", 18.0, 20.0, (0.0, 0.0, 0.0)),
  ),
}
ACTIVE_SCENARIO = "translations"
PHASES = tuple((name, start, end) for name, start, end, _ in SCENARIOS[ACTIVE_SCENARIO])


def profile(elapsed_s):
  for name, start, end, command in SCENARIOS[ACTIVE_SCENARIO]:
    if start <= elapsed_s < end:
      return name, command
  raise ValueError("Outside the20s recording profile")


def _json_safe(value):
  """Preserve invalid terminal measurements as null, never nonstandard JSON NaN."""
  if isinstance(value, dict):
    return {key: _json_safe(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)):
    return [_json_safe(item) for item in value]
  if isinstance(value, float) and not math.isfinite(value):
    return None
  return value


def _fresh_observations(env):
  # compute() can return a cached observation containing the previous command.
  from tensordict import TensorDict

  return TensorDict(
    {
      name: env.observation_manager.compute_group(name, update_history=False)
      for name in env.cfg.observations
    },
    batch_size=[1],
  )


def _refresh_contact_reads(env) -> None:
  """Read final-forward contacts without stepping physics or sensor history."""
  for name in ("feet_ground_contact", "limb_ground_contact"):
    env.scene[name]._invalidate_cache()


def _faults(data, sensor, nonfoot, reward=None) -> list[str]:
  from pochi_rl.lp_acrl.physics import physics_fault_masks

  faults = [name for name, mask in physics_fault_masks(data).items() if bool(mask[0])]
  for name, value in (
    ("foot_contact", sensor.data.found),
    ("foot_force", sensor.data.force),
    ("nonfoot_contact", nonfoot.data.found),
    ("foot_position", data.site_pos_w),
    ("foot_velocity", data.site_lin_vel_w),
    ("reward", reward),
  ):
    if value is not None and not bool(torch.isfinite(value).all()):
      faults.append(f"nonfinite/{name}")
  return faults


def _snapshot(env, sites, order) -> dict:
  data = env.scene["robot"].data
  sensor = env.scene["feet_ground_contact"]

  def values(tensor):
    return tensor[0].detach().cpu().tolist()

  error = data.joint_pos.double() - data.default_joint_pos.double()
  return _json_safe(
    {
      "contact": values(sensor.data.found[:, order] > 0),
      "foot_force_w_N": values(sensor.data.force[:, order]),
      "foot_vertical_load_abs_N": values(sensor.data.force[:, order, 2].abs()),
      "foot_position_w_m": values(data.site_pos_w[:, sites]),
      "foot_velocity_w_m_s": values(data.site_lin_vel_w[:, sites]),
      "joint_default_rmse_rad": float(error.square().mean().sqrt()),
      "joint_position_rad": values(data.joint_pos),
      "joint_velocity_rad_s": values(data.joint_vel),
      "body_position_w_m": values(data.root_link_pos_w),
      "body_quaternion_wxyz": values(data.root_link_quat_w),
      "body_linear_velocity_b_m_s": values(data.root_link_lin_vel_b),
      "body_angular_velocity_b_rad_s": values(data.root_link_ang_vel_b),
      "nonfoot_contact": bool((env.scene["limb_ground_contact"].data.found > 0).any()),
    }
  )


def summarize(records: list[dict], dt: float) -> list[dict]:
  """Summarize each phase after an explicit1s settling interval; no pass grade."""
  rows = []
  for name, start, end in PHASES:
    chosen = [
      record
      for record in records
      if record["phase"] == name
      and record["valid"]
      and record["time_after_s"] > start + 1.0 + 1e-9
    ]
    row = {
      "phase": name,
      "start_s": start,
      "end_s": end,
      "excluded_transient_s": 1.0,
      "valid_samples": len(chosen),
      "expected_samples": round((end - start - 1.0) / dt),
      "metrics": None,
    }
    if chosen:

      def array(key, rows=chosen):
        return np.asarray([record[key] for record in rows])

      contact = array("contact")
      force = array("foot_vertical_load_abs_N")
      velocity = array("body_linear_velocity_b_m_s")
      angular = array("body_angular_velocity_b_rad_s")
      position = array("body_position_w_m")
      foot_speed = np.linalg.norm(array("foot_velocity_w_m_s")[:, :, :2], axis=-1)
      q_rmse = array("joint_default_rmse_rad")
      duty = contact.mean(0)
      row["metrics"] = {
        "joint_default_rmse_rad": float(np.sqrt(np.mean(q_rmse**2))),
        "joint_velocity_rms_rad_s": float(
          np.sqrt(np.mean(array("joint_velocity_rad_s") ** 2))
        ),
        "all_four_contact_fraction": float(contact.all(-1).mean()),
        "foot_contact_fraction": duty.tolist(),
        "foot_contact_duty_spread": float(duty.max() - duty.min()),
        "foot_vertical_load_abs_mean_N": force.mean(0).tolist(),
        "foot_loaded_fraction_gt_1N": (force > 1.0).mean(0).tolist(),
        "body_linear_velocity_b_mean_m_s": velocity.mean(0).tolist(),
        "body_angular_velocity_b_mean_rad_s": angular.mean(0).tolist(),
        "horizontal_speed_mean_m_s": float(
          np.linalg.norm(velocity[:, :2], axis=-1).mean()
        ),
        "yaw_rate_absolute_mean_rad_s": float(np.abs(angular[:, 2]).mean()),
        "horizontal_displacement_m": float(
          np.linalg.norm(position[-1, :2] - position[0, :2])
        ),
        "horizontal_drift_max_m": float(
          np.linalg.norm(position[:, :2] - position[0, :2], axis=-1).max()
        ),
        "contact_slip_mean_m_s": float(
          ((foot_speed * contact).sum(-1) / contact.sum(-1).clip(min=1)).mean()
        ),
        "nonfoot_contact_fraction": float(array("nonfoot_contact").mean()),
      }
    row["complete"] = row["valid_samples"] == row["expected_samples"]
    rows.append(row)
  return rows


@torch.inference_mode()
def record_demo(env, wrapped, policy, *, writer=None, max_steps=None) -> dict:
  """Record one first episode; optional max_steps is for bounded integration tests."""
  from pochi_rl.robot import FOOT_GEOMS, FOOT_SITES

  if env.num_envs != 1 or env.cfg.auto_reset:
    raise ValueError("Recording requires one environment with auto_reset=False")
  if not np.isclose(env.max_episode_length * env.step_dt, 20.0):
    raise ValueError("The checkpoint must retain its20s episode horizon")
  every = round(1.0 / (FPS * env.step_dt))
  if every < 1 or not np.isclose(every * env.step_dt * FPS, 1.0):
    raise ValueError("Control rate must permit exactly25 video frames per second")
  limit = env.max_episode_length if max_steps is None else max_steps
  if not 1 <= limit <= env.max_episode_length:
    raise ValueError("max_steps must lie within the checkpoint episode horizon")

  term = env.command_manager.get_term("base_velocity")
  term.set_evaluation(task_ids=TASK_ID, command=(0.0, 0.0, 0.0))
  env.reset()
  _refresh_contact_reads(env)
  robot = env.scene["robot"]
  sites, _ = robot.find_sites(FOOT_SITES, preserve_order=True)
  sensor = env.scene["feet_ground_contact"]
  nonfoot = env.scene["limb_ground_contact"]
  names = [name.rsplit("/", 1)[-1] for name in sensor.primary_names]
  order = [names.index(name) for name in FOOT_GEOMS]
  if sensor.data.force is None or sensor.data.found.shape != (1, 4):
    raise ValueError("Recording requires four foot contact and net-force readings")
  initial = _snapshot(env, sites, order)
  initial_faults = _faults(robot.data, sensor, nonfoot)
  records, frame_times = [], []
  reason = "invalid_initial_state" if initial_faults else "profile_complete"
  failure = initial_faults
  if not initial_faults:
    for step in range(limit):
      before = step * env.step_dt
      phase, values = profile(before)
      command = torch.tensor([values], device=env.device)
      term.vel_command_b.copy_(command)
      term.vel_command_w.copy_(command)
      term.is_standing_env[:] = command.abs().amax(dim=1) <= 1e-6
      obs = _fresh_observations(env)
      if not all(bool(torch.isfinite(value).all()) for value in obs.values()):
        reason, failure = "invalid_observation", ["nonfinite/policy_observation"]
        break
      if any(value.shape != (1, 50) for value in obs.values()):
        raise ValueError("Blind demo requires exactly50 observations per group")
      actions = policy(obs)
      if actions.shape != (1, 12):
        raise ValueError("Blind demo requires12 policy actions")
      if not bool(torch.isfinite(actions).all()):
        reason, failure = "invalid_action", ["nonfinite/policy_action"]
        break
      # Standard RSL wrapper applies its usual±4 clipping; no action replacement.
      _, reward, done, _ = wrapped.step(actions)
      _refresh_contact_reads(env)
      faults = _faults(robot.data, sensor, nonfoot, reward)
      terminated = bool(env.reset_terminated[0])
      truncated = bool(env.reset_time_outs[0])
      after = (step + 1) * env.step_dt
      record = {
        "step": step + 1,
        "time_before_s": before,
        "time_after_s": after,
        "phase": phase,
        "command_vx_vy_wz": list(values),
        "valid": not faults and not terminated,
        "terminated": terminated,
        "truncated": truncated,
        "physics_faults": faults,
        **_snapshot(env, sites, order),
      }
      records.append(record)
      # Render finite terminal falls, but never ask MuJoCo to render invalid state.
      if (
        writer is not None and not faults and ((step + 1) % every == 0 or bool(done[0]))
      ):
        frame = env.render()
        if frame is None or frame.shape[:2] != (540, 960):
          raise RuntimeError("Renderer must return960x540 RGB frames")
        writer.append_data(frame)
        frame_times.append(after)
      if faults:
        reason, failure = "invalid_state", faults
        break
      if terminated:
        reason = "fall"
        break
      if bool(done[0]):
        reason = (
          "profile_complete" if step + 1 == env.max_episode_length else "early_timeout"
        )
        break
    else:
      if limit < env.max_episode_length:
        reason = "bounded_check"

  return {
    "protocol": PROTOCOL,
    "task_id": TASK_ID,
    "terrain": "native infinite plane",
    "terrain_level": 0,
    "step_height_m": 0.0,
    "terrain_origin_w_m": env.scene.env_origins[0].detach().cpu().tolist(),
    "control_dt_s": env.step_dt,
    "requested_duration_s": 20.0,
    "recorded_duration_s": len(records) * env.step_dt,
    "complete": reason == "profile_complete" and len(records) == env.max_episode_length,
    "stop_reason": reason,
    "failure_reasons": failure,
    "fall": any(record["terminated"] for record in records),
    "invalid": bool(failure),
    "foot_order": ["FL", "FR", "RL", "RR"],
    "default_joint_position_rad": robot.data.default_joint_pos[0]
    .detach()
    .cpu()
    .tolist(),
    "initial_state": initial,
    "video_fps": FPS,
    "video_frame_times_s": frame_times,
    "metric_conventions": {
      "sampling": (
        "50Hz after each control step and its final forward/sense; "
        "terminal states before any reset"
      ),
      "contact_alignment": (
        "Feet and limb contact read caches invalidated after reset and each "
        "wrapped.step, before faults/snapshots. No extra physics, sensor update "
        "or history advancement. Earlier untagged recordings retain legacy "
        "cached contact sampling."
      ),
      "foot_load": "Absolute world-Z net contact force; contact flags also retained",
      "valid": "Finite, within shared physics limits, and not a fall transition",
      "settling": (
        "First 1 s of each phase excluded only from summary; trace/video retain it"
      ),
      "purpose": (
        "Single first-take flat omni visualization; "
        "separate registered batch acceptance"
      ),
    },
    "segments": summarize(records, env.step_dt),
    "records": records,
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--scenario", choices=tuple(SCENARIOS), required=True)
  parser.add_argument("--checkpoint", required=True, type=Path)
  parser.add_argument("--output", required=True, type=Path)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seed", type=int, choices=[10125, 10126], required=True)
  args = parser.parse_args()
  if (args.scenario, args.seed) not in (
    ("translations", 10125),
    ("rotation_combined", 10126),
  ):
    parser.error("Require the registered scenario/seed pair")
  global ACTIVE_SCENARIO, PHASES
  ACTIVE_SCENARIO = args.scenario
  PHASES = tuple(
    (name, start, end) for name, start, end, _ in SCENARIOS[ACTIVE_SCENARIO]
  )
  if args.output.exists() or args.output.with_suffix(".json").exists():
    parser.error("Preserve earlier takes")
  import imageio.v2 as imageio
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.torch import configure_torch_backends

  from pochi_rl.mjlab.lp_acrl_blind_v18 import register_blind_v18
  from pochi_rl.mjlab.lp_acrl_omni_runner import (
    OmniPhaseGaitRunner,
    validate_omni_checkpoint,
  )
  from pochi_rl.mjlab.warp_contact_backend import install_contact_backend

  os.environ["MUJOCO_GL"] = "egl"
  backend = install_contact_backend()
  saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  validate_omni_checkpoint(saved)
  configure_torch_backends()
  register_blind_v18()
  cfg = load_env_cfg(TASK, play=True)
  cfg.scene.num_envs = 1
  cfg.seed = args.seed
  cfg.auto_reset = False
  cfg.commands["base_velocity"].debug_vis = False
  cfg.viewer.width, cfg.viewer.height = 960, 540
  cfg.viewer.max_extra_envs = 0
  cfg.viewer.distance = 1.6
  cfg.viewer.elevation = -20
  cfg.viewer.azimuth = 130
  agent = load_rl_cfg(TASK)
  env = ManagerBasedRlEnv(cfg, device=args.device, render_mode="rgb_array")
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = OmniPhaseGaitRunner(wrapped, asdict(agent), device=args.device)
    runner.load(
      str(args.checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with imageio.get_writer(
      str(args.output), fps=FPS, codec="libx264", quality=7, macro_block_size=2
    ) as writer:
      report = record_demo(env, wrapped, policy, writer=writer)
    report.update(
      scenario=args.scenario,
      scenario_segments=SCENARIOS[args.scenario],
      task=TASK,
      checkpoint=str(args.checkpoint),
      checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
      seed=args.seed,
      device=args.device,
      video=str(args.output),
      height_scan=False,
      observation_dim=50,
      action_dim=12,
      contact_backend=backend,
      central_payload_kg=3.0,
      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    )
    args.output.with_suffix(".json").write_text(
      json.dumps(_json_safe(report), indent=2) + "\n"
    )
    print(
      json.dumps(
        {
          "output": str(args.output),
          "recording": report["recording_status"]
          if "recording_status" in report
          else report.get("outcome"),
        },
        indent=2,
      )
    )
  finally:
    env.close()


if __name__ == "__main__":
  main()
