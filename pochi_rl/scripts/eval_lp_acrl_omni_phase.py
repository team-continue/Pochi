"""Registered1.5Hz/foot-lift behavioral checks, with raw contact/height traces."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from eval_lp_acrl_behavior_plane import FreshContactStepWrapper, frozen

# Body-frame commands; same directions are tested at several distinct speeds.
PROFILE_SEGMENTS = {"stand": ((0.0, 20.0, (0.0, 0.0, 0.0)),)}
for axis, label, speeds in (
  (0, "x", (0.15, 0.25, 0.5, 0.8)),
  (1, "y", (0.15, 0.3, 0.5)),
  (2, "yaw", (0.4, 1.0, 1.8)),
):
  for sign in (-1, 1):
    for speed in speeds:
      command = [0.0, 0.0, 0.0]
      command[axis] = sign * speed
      PROFILE_SEGMENTS[f"{label}_{sign * speed:+g}"] = ((0.0, 20.0, tuple(command)),)
for sx in (-1, 1):
  for sy in (-1, 1):
    PROFILE_SEGMENTS[f"diagonal_{sx}_{sy}"] = ((0.0, 20.0, (sx * 0.35, sy * 0.2, 0.0)),)
    PROFILE_SEGMENTS[f"arc_{sx}_{sy}"] = ((0.0, 20.0, (sx * 0.4, 0.0, sy * 0.9)),)
    PROFILE_SEGMENTS[f"omni_{sx}_{sy}"] = (
      (0.0, 20.0, (sx * 0.3, sy * 0.2, sx * sy * 0.6)),
    )
for name, axis, value in (("stop_start_lateral", 1, 0.25), ("stop_start_yaw", 2, 0.6)):
  positive = [0.0, 0.0, 0.0]
  positive[axis] = value
  negative = tuple(-v for v in positive)
  PROFILE_SEGMENTS[name] = (
    (0.0, 5.0, (0.0, 0.0, 0.0)),
    (5.0, 10.0, tuple(positive)),
    (10.0, 15.0, (0.0, 0.0, 0.0)),
    (15.0, 20.0, negative),
  )
PROFILE_SEGMENTS["direction_switch"] = (
  (0.0, 4.0, (0.4, 0.0, 0.0)),
  (4.0, 8.0, (0.0, 0.3, 0.0)),
  (8.0, 12.0, (-0.4, 0.0, 0.0)),
  (12.0, 16.0, (0.0, -0.3, 0.0)),
  (16.0, 20.0, (0.0, 0.0, 0.0)),
)
PROFILES = tuple(PROFILE_SEGMENTS)
fresh_command_observations = frozen.fresh_command_observations


def profile_command(profile, elapsed):
  for start, end, command in PROFILE_SEGMENTS[profile]:
    if start <= elapsed < end:
      return command
  raise ValueError("Outside the20s registered profile")


def summarize_omni_segment(samples, index, start, end, command, dt, settle_s):
  segment = frozen.summarize_segment(
    samples, index, start, end, command, dt, settle_s=settle_s
  )
  if segment["metrics"] is None:
    return segment
  if not any(command):
    return segment  # Existing exact-zero standing criteria are retained.
  valid = (
    (samples["time_s"] > start + settle_s + 1e-9)
    & (samples["time_s"] <= end + 1e-9)
    & samples["valid"][:, index]
  )
  contacts = samples["contact"][valid, index]
  height = samples["foot_z"][valid, index]
  velocity = samples["velocity"][valid, index]
  metrics = segment["metrics"]
  errors = np.abs(velocity - np.asarray(command)).mean(0)
  metrics["command_error_mean_abs_vx_vy_wz"] = errors.tolist()
  tolerance = np.maximum([0.05, 0.05, 0.10], 0.15 * np.abs(command))
  metrics["command_error_tolerance_vx_vy_wz"] = tolerance.tolist()
  duty = np.asarray(metrics["foot_contact_fraction"])
  checks = {
    "each_axis_command_error": bool((errors <= tolerance).all()),
    "contact_slip_mean_m_s": metrics["contact_slip_mean_m_s"] <= 0.10,
    "tilt_p95_deg": metrics["tilt_p95_deg"] <= 15.0,
    "nonfoot_contact_fraction": metrics["nonfoot_contact_fraction"] <= 0.01,
    "contact_duty_balance": metrics["foot_contact_duty_spread"] <= 0.30,
    "all_legs_contact_and_swing": bool(((duty >= 0.15) & (duty <= 0.85)).all()),
    "all_legs_footfalls": min(metrics["footfalls_per_leg"]) >= 2,
  }
  frequency = np.asarray(metrics["footfalls_per_leg"]) / (
    segment["expected_samples"] * dt
  )
  cycles, qualified = [], []
  for foot in range(4):
    contact = contacts[:, foot]
    starts = np.flatnonzero(contact[:-1] & ~contact[1:]) + 1
    peaks = []
    for lift in starts:
      landing = np.flatnonzero(contact[lift:])
      if not len(landing):
        continue
      last = lift + landing[0]
      if (last - lift) * dt + 1e-9 >= 0.06:
        peaks.append(float(height[lift:last, foot].max() - height[lift - 1, foot]))
    cycles.append(peaks)
    qualified.append(float(np.mean(np.asarray(peaks) >= 0.04 - 1e-6)) if peaks else 0.0)
  metrics.update(
    cadence_per_leg_hz=frequency.tolist(),
    observed_swing_peak_lifts_m=cycles,
    qualified_lift_fraction_per_leg=qualified,
    whole_body_airborne_fraction=float((~contacts.any(-1)).mean()),
  )
  checks.update(
    each_leg_cadence_1p2_to_1p8_hz=bool(
      ((frequency >= 1.2) & (frequency <= 1.8)).all()
    ),
    each_leg_at_least80pct_swings_lift4cm=bool((np.asarray(qualified) >= 0.8).all()),
    each_leg_at_least2_observed_swings=all(len(v) >= 2 for v in cycles),
    no_observed_whole_body_flight=bool(contacts.any(-1).all()),
  )
  segment.update(checks=checks, passed=segment["complete"] and all(checks.values()))
  return segment


def summarize_phase_profiles(samples, profiles, dt, *, falls, invalid, horizon_steps):
  if not np.isclose(horizon_steps * dt, 20.0):
    raise ValueError("Only the registered20s protocol is supported")
  rows = []
  for index, profile in enumerate(profiles):
    segments = [
      summarize_omni_segment(
        samples,
        index,
        start,
        end,
        command,
        dt,
        2.0 if len(PROFILE_SEGMENTS[profile]) == 1 else 1.0,
      )
      for start, end, command in PROFILE_SEGMENTS[profile]
    ]
    rows.append(
      dict(
        environment_index=index,
        profile=profile,
        fall=bool(falls[index]),
        invalid_state=bool(invalid[index]),
        segments=segments,
        passed=not bool(falls[index] or invalid[index])
        and all(s["passed"] for s in segments),
      )
    )
  return rows


@torch.inference_mode()
def evaluate_profiles(env, wrapped, policy):
  from pochi_rl.lp_acrl.physics import invalid_physics_mask
  from pochi_rl.robot import FOOT_GEOMS, FOOT_SITES

  if env.cfg.auto_reset:
    raise ValueError("Behavior evaluation requires auto_reset=False")
  if env.num_envs % len(PROFILES):
    raise ValueError("num_envs must be a multiple of the registered profile count")
  if not np.isclose(env.max_episode_length * env.step_dt, 20.0):
    raise ValueError("Behavior profiles currently require the20s checkpoint horizon")
  profiles = [PROFILES[i % len(PROFILES)] for i in range(env.num_envs)]
  term = env.command_manager.get_term("base_velocity")
  term.set_evaluation(task_ids=0, command=(0.0, 0.0, 0.0))
  env.reset()
  robot = env.scene["robot"]
  sites, _ = robot.find_sites(FOOT_SITES, preserve_order=True)
  sensor = env.scene["feet_ground_contact"]
  names = [name.rsplit("/", 1)[-1] for name in sensor.primary_names]
  order = [names.index(name) for name in FOOT_GEOMS]
  nonfoot = env.scene["limb_ground_contact"]
  active = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
  falls = torch.zeros_like(active)
  invalid = torch.zeros_like(active)
  history = []
  for step in range(env.max_episode_length):
    command = torch.tensor(
      [profile_command(profile, step * env.step_dt) for profile in profiles],
      device=env.device,
      dtype=torch.float32,
    )
    term.vel_command_b[:] = command
    term.vel_command_w[:] = command
    term.is_standing_env[:] = command.abs().amax(-1) <= 1e-6
    observations = fresh_command_observations(env)
    action = policy(observations)
    action[~active] = 0
    _, reward, done, _ = wrapped.step(action)
    terminated = env.reset_terminated.bool()
    physical_valid = ~invalid_physics_mask(robot.data) & torch.isfinite(reward)
    contact_raw = sensor.data.found
    nonfoot_raw = nonfoot.data.found
    foot_force = sensor.data.force[:, order, 2].abs()
    physical_valid &= torch.isfinite(contact_raw).reshape(env.num_envs, -1).all(-1)
    physical_valid &= torch.isfinite(nonfoot_raw).reshape(env.num_envs, -1).all(-1)
    physical_valid &= torch.isfinite(foot_force).all(-1)
    contact = contact_raw[:, order] > 0
    slip_per_foot = robot.data.site_lin_vel_w[:, sites, :2].norm(dim=-1)
    slip = (slip_per_foot * contact).sum(-1) / contact.sum(-1).clamp_min(1)
    physical_valid &= torch.isfinite(slip)
    valid = active & physical_valid & ~terminated
    velocity = torch.stack(
      (
        robot.data.root_link_lin_vel_b[:, 0],
        robot.data.root_link_lin_vel_b[:, 1],
        robot.data.root_link_ang_vel_b[:, 2],
      ),
      dim=-1,
    )
    quat = robot.data.root_link_quat_w
    cosine = (1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2)).clamp(-1, 1)
    record = {
      "valid": valid,
      "velocity": velocity,
      "position": robot.data.root_link_pos_w,
      "contact": contact,
      "foot_vertical_force": foot_force,
      "nonfoot_contact": (nonfoot_raw > 0).reshape(env.num_envs, -1).any(-1),
      "slip": slip,
      "tilt_deg": torch.rad2deg(torch.acos(cosine)),
      "joint_error": robot.data.joint_pos - robot.data.default_joint_pos,
      "joint_velocity": robot.data.joint_vel,
      "foot_z": robot.data.site_pos_w[:, sites, 2],
    }
    history.append(
      {key: value.detach().cpu().numpy().copy() for key, value in record.items()}
    )
    falls |= active & terminated
    invalid |= active & ~physical_valid
    finished = done.bool() | ~physical_valid
    active &= ~finished
    if not bool(active.any()):
      break
    if bool(finished.any()):
      # Read terminal states first; all later padding episodes remain excluded.
      env.reset(env_ids=finished.nonzero(as_tuple=False).flatten())
  samples = {key: np.stack([record[key] for record in history]) for key in history[0]}
  samples["time_s"] = np.arange(1, len(history) + 1) * env.step_dt
  return summarize_phase_profiles(
    samples,
    profiles,
    env.step_dt,
    falls=falls.cpu().numpy(),
    invalid=invalid.cpu().numpy(),
    horizon_steps=env.max_episode_length,
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--seed", type=int, default=8456)
  parser.add_argument("--num-envs", type=int, default=len(PROFILES) * 4)
  args = parser.parse_args()
  if (args.seed, args.num_envs) not in [
    (8456, len(PROFILES) * 4),
    (8789, len(PROFILES) * 8),
  ]:
    parser.error("Registered screen8456 or holdout8789 only")
  if args.output.exists():
    parser.error("Preserve previous outcomes")
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
  from pochi_rl.robot import PAYLOAD_MASS_KG

  backend = install_contact_backend()
  checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
  saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  count = validate_omni_checkpoint(saved)
  configure_torch_backends()
  register_blind_v18()
  cfg = load_env_cfg(TASK_ID, play=True)
  cfg.seed = args.seed
  cfg.scene.num_envs = args.num_envs
  cfg.auto_reset = False
  cfg.commands["base_velocity"].debug_vis = False
  agent = load_rl_cfg(TASK_ID)
  env = ManagerBasedRlEnv(cfg, device=args.device)
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = OmniPhaseGaitRunner(wrapped, asdict(agent), device=args.device)
    runner.load(
      str(args.checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=args.device,
    )
    rows = evaluate_profiles(
      env,
      FreshContactStepWrapper(env, wrapped),
      runner.get_inference_policy(device=args.device),
    )
    summary = {
      p: {
        "episodes": sum(e["profile"] == p for e in rows),
        "passed": sum(e["profile"] == p and e["passed"] for e in rows),
        "falls": sum(e["profile"] == p and e["fall"] for e in rows),
        "invalid": sum(e["profile"] == p and e["invalid_state"] for e in rows),
      }
      for p in PROFILES
    }
    report = dict(
      protocol="pochi-omni-phase1p5hz-lift5cm-v1",
      task=TASK_ID,
      checkpoint=str(args.checkpoint.resolve()),
      checkpoint_sha256=checkpoint_sha,
      completed_updates=count,
      seed=args.seed,
      num_envs=args.num_envs,
      control_dt_s=env.step_dt,
      contact_backend=backend,
      body_observation_dims=48,
      phase_observation_dims=2,
      heightscan=False,
      payload_kg=PAYLOAD_MASS_KG,
      profile_segments=PROFILE_SEGMENTS,
      tracking_tolerance=(
        "Each mean absolute axis error "
        "<=max(0.05m/s or0.10rad/s,15% command magnitude)"
      ),
      convention=(
        "One leg cadence1.2-1.8Hz; >=80% observed complete swings lift>=4cm "
        "relative to last preceding contact; target5cm. Native infinite plane, "
        "first episodes, fresh post-control contacts. No whole-body flight. "
        "Frozen stop/walk criteria additionally required."
      ),
      source_sha256={
        Path(__file__).name: hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        Path(frozen.__file__).name: hashlib.sha256(
          Path(frozen.__file__).read_bytes()
        ).hexdigest(),
      },
      summary=summary,
      episodes=rows,
    )
    assert hashlib.sha256(args.checkpoint.read_bytes()).hexdigest() == checkpoint_sha
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "summary": summary}), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
