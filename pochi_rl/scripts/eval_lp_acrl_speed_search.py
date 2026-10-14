"""Speed-limit screen with unchanged gait criteria and motor measurements."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from eval_lp_acrl_behavior_plane import FreshContactStepWrapper, frozen
from eval_lp_acrl_omni_phase import PROFILE_SEGMENTS as RETAINED_PROFILES
from eval_lp_acrl_omni_phase import summarize_omni_segment
from eval_lp_acrl_omni_transitions import transition_metrics

PROFILE_SEGMENTS = dict(RETAINED_PROFILES)
for axis, label, speeds in (
  (0, "x", (1.0, 1.25, 1.5, 2.0, 2.5, 3.0, 4.0)),
  (1, "y", (0.75, 1.0, 1.25, 1.5)),
  (2, "yaw", (2.5, 3.0, 4.0)),
):
  for sign in (-1, 1):
    for speed in speeds:
      command = [0.0, 0.0, 0.0]
      command[axis] = sign * speed
      PROFILE_SEGMENTS[f"{label}_{sign * speed:+g}"] = ((0.0, 20.0, tuple(command)),)
for vx in (-1.5, -1.0, -0.5, 0.5, 1.0, 1.5):
  for sy in (-1, 1):
    for sw in (-1, 1):
      PROFILE_SEGMENTS[f"moving_{vx:+g}_vy{sy:+d}_yaw{sw:+d}"] = (
        (0.0, 2.0, (0.0, 0.0, 0.0)),
        (2.0, 6.0, (vx, 0.0, 0.0)),
        (6.0, 10.0, (vx, sy * 0.3, sw * 0.8)),
        (10.0, 14.0, (vx, -sy * 0.3, -sw * 0.8)),
        (14.0, 18.0, (vx, 0.0, 0.0)),
        (18.0, 20.0, (0.0, 0.0, 0.0)),
      )
PROFILES = tuple(PROFILE_SEGMENTS)
fresh_command_observations = frozen.fresh_command_observations


def motor_snapshot(robot, action):
  """Last substep's torque limits using the velocity cached by the motor.

  A near-bound output indicates utilization, not proven physical impossibility.
  These are 50Hz samples, not peak/thermal or hardware validation.
  """
  from pochi_rl.robot import RS02_NO_LOAD_SPEED_RAD_S, RS02_PEAK_TORQUE_NM

  torque = robot.data.qfrc_actuator
  velocity = torch.zeros_like(torque)
  for actuator in robot.actuators:
    velocity[:, actuator.target_ids] = actuator._joint_vel_clipped
  clipped = velocity.clamp(-2 * RS02_NO_LOAD_SPEED_RAD_S, 2 * RS02_NO_LOAD_SPEED_RAD_S)
  upper = (RS02_PEAK_TORQUE_NM * (1 - clipped / RS02_NO_LOAD_SPEED_RAD_S)).clamp_max(
    RS02_PEAK_TORQUE_NM
  )
  lower = (RS02_PEAK_TORQUE_NM * (-1 - clipped / RS02_NO_LOAD_SPEED_RAD_S)).clamp_min(
    -RS02_PEAK_TORQUE_NM
  )
  limits = robot.data.joint_pos_limits
  position = robot.data.joint_pos
  margin = torch.minimum(position - limits[:, :, 0], limits[:, :, 1] - position)
  return {
    "action_raw": action,
    "motor_torque_nm": torque,
    "motor_cached_velocity_rad_s": velocity,
    "motor_upper_nm": upper,
    "motor_lower_nm": lower,
    "joint_limit_margin_rad": margin,
  }


def summarize_mechanics(samples, index):
  valid = samples["valid"][:, index] & (samples["time_s"] > 2.0)
  if not valid.any():
    return None

  def values(key):
    return samples[key][valid, index]

  torque = values("motor_torque_nm")
  high, low = values("motor_upper_nm"), values("motor_lower_nm")
  near_bound = (np.abs(torque - high) <= 0.05 * np.maximum(np.abs(high), 1.0)) | (
    np.abs(torque - low) <= 0.05 * np.maximum(np.abs(low), 1.0)
  )
  return {
    "sampled_motor_near_limit_fraction_per_joint": near_bound.mean(0).tolist(),
    "sampled_motor_abs_torque_p95_nm_per_joint": np.percentile(
      np.abs(torque), 95, axis=0
    ).tolist(),
    "sampled_joint_abs_speed_p95_rad_s_per_joint": np.percentile(
      np.abs(values("joint_velocity")), 95, axis=0
    ).tolist(),
    "sampled_joint_limit_within_0p05rad_fraction_per_joint": (
      values("joint_limit_margin_rad") <= 0.05
    )
    .mean(0)
    .tolist(),
    "raw_action_exceeds_clip4_fraction_per_joint": (np.abs(values("action_raw")) >= 4)
    .mean(0)
    .tolist(),
    "interpretation": (
      "50Hz last-substep samples; utilization only, "
      "not a proof of physical maximum or thermal safety"
    ),
  }


def profile_command(profile, elapsed):
  for start, end, command in PROFILE_SEGMENTS[profile]:
    if start <= elapsed < end:
      return command
  raise ValueError("Outside the20s registered profile")


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
    transitions = []
    if profile.startswith("moving_"):
      specs = PROFILE_SEGMENTS[profile]
      transitions = [
        transition_metrics(
          samples, index, specs[i][0], specs[i - 1][2], specs[i][2], dt
        )
        for i in (2, 3, 4)
      ]
    rows.append(
      dict(
        environment_index=index,
        profile=profile,
        fall=bool(falls[index]),
        invalid_state=bool(invalid[index]),
        mechanics=summarize_mechanics(samples, index),
        switches=transitions,
        segments=segments,
        passed=not bool(falls[index] or invalid[index])
        and all(s["passed"] for s in segments)
        and all(s["passed"] for s in transitions),
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
      **motor_snapshot(robot, action),
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
  return samples, summarize_phase_profiles(
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
  parser.add_argument("--seed", type=int, default=20456)
  parser.add_argument("--num-envs", type=int, default=len(PROFILES) * 4)
  parser.add_argument(
    "--source-v18",
    action="store_true",
    help="Evaluate frozen V18 as the extrapolation baseline",
  )
  args = parser.parse_args()
  if (args.seed, args.num_envs) not in [
    (20456, len(PROFILES) * 4),
    (21456, len(PROFILES) * 8),
  ]:
    parser.error("New development screen20456 or independent holdout21456 only")
  if args.output.exists() or args.output.with_suffix(".samples.npz").exists():
    parser.error("Preserve previous outcomes")
  if os.environ.get("CUDA_VISIBLE_DEVICES") != "3" or args.device != "cuda:0":
    raise ValueError("Set CUDA_VISIBLE_DEVICES=3 and use cuda:0 for GPU3 only")
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.torch import configure_torch_backends

  if args.source_v18:
    from pochi_rl.mjlab.lp_acrl_blind_v18 import TASK_ID
    from pochi_rl.mjlab.lp_acrl_blind_v18 import register_blind_v18 as register
    from pochi_rl.mjlab.lp_acrl_omni_runner import (
      OmniPhaseGaitRunner as Runner,
    )
    from pochi_rl.mjlab.lp_acrl_omni_runner import (
      validate_omni_checkpoint as validate,
    )
  else:
    from pochi_rl.mjlab.lp_acrl_blind_v19 import TASK_ID
    from pochi_rl.mjlab.lp_acrl_blind_v19 import register_blind_v19 as register
    from pochi_rl.mjlab.lp_acrl_speed_runner import (
      SpeedPhaseGaitRunner as Runner,
    )
    from pochi_rl.mjlab.lp_acrl_speed_runner import (
      validate_speed_checkpoint as validate,
    )
  from pochi_rl.mjlab.warp_contact_backend import install_contact_backend
  from pochi_rl.robot import PAYLOAD_MASS_KG

  backend = install_contact_backend()
  checkpoint_sha = hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
  saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  count = validate(saved)
  root = Path(__file__).resolve().parents[1]
  parent_plan = Path(
    "/data/k_suzuki/pochi_lp_acrl_artifacts/2026-10-08/blind_v18_omni_gpu3_continue_diagnostics/continuation_plan.json"
  )
  pins = json.loads(parent_plan.read_text())["source_sha256"]
  if not args.source_v18:
    pins = {
      **pins,
      **json.loads((args.checkpoint.parent / "run.json").read_text())["source_sha256"],
    }
  pins.setdefault(
    str(Path(__file__).resolve().relative_to(root)),
    hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  )
  for name, expected in pins.items():
    assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
  args.output.parent.mkdir(parents=True, exist_ok=True)
  plan = {
    "checkpoint_sha256": checkpoint_sha,
    "seed": args.seed,
    "num_envs": args.num_envs,
    "profiles": PROFILE_SEGMENTS,
    "source_sha256": pins,
    "physical_gpu": 3,
    "first_episode_only": True,
    "criteria": (
      "Frozen V18 gait/stop/axis tolerances + original first-second transition metrics"
    ),
    "development_only": args.seed == 20456,
    "held_out_seed": 21456,
  }
  with args.output.with_suffix(".plan.json").open("x") as f:
    json.dump(plan, f, indent=2)
  configure_torch_backends()
  register()
  cfg = load_env_cfg(TASK_ID, play=True)
  cfg.seed = args.seed
  cfg.scene.num_envs = args.num_envs
  cfg.auto_reset = False
  cfg.commands["base_velocity"].debug_vis = False
  agent = load_rl_cfg(TASK_ID)
  env = ManagerBasedRlEnv(cfg, device=args.device)
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = Runner(wrapped, asdict(agent), device=args.device)
    runner.load(
      str(args.checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=args.device,
    )
    samples, rows = evaluate_profiles(
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
      protocol="pochi-v19-speed-search-development-v1",
      source_v18_extrapolation=args.source_v18,
      retained_profile_count=len(RETAINED_PROFILES),
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
        "Each mean absolute axis error <=max(0.05m/s or0.10rad/s,15% command magnitude)"
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
    sample_path = args.output.with_suffix(".samples.npz")
    if sample_path.exists():
      raise FileExistsError(sample_path)
    np.savez_compressed(sample_path, **samples)
    report["raw_samples"] = str(sample_path)
    for name, expected in pins.items():
      assert hashlib.sha256((root / name).read_bytes()).hexdigest() == expected, name
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "summary": summary}), flush=True)
  finally:
    env.close()


if __name__ == "__main__":
  main()
