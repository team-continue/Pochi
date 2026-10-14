"""Diagnose standing, walking and stop/start on explicitly flattened LP terrain.

This is a behavioral diagnostic, not the paper's task-catalog evaluation.
Physics, robot, action and observation identities remain checkpoint-exact.
Only terrain subfamilies and unreported rewards may differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

PROFILES = ("stand", "forward", "backward", "stop_start")
FOOT_ORDER = ("FL", "FR", "RL", "RR")
MIN_AIR_TIME_S = 0.04
STAND_LIMITS = {
  "joint_default_rmse_rad": 0.12,
  "all_four_contact_fraction_min": 0.95,
  "horizontal_speed_mean_m_s": 0.03,
  "yaw_rate_absolute_mean_rad_s": 0.05,
  "horizontal_drift_max_m": 0.10,
  "contact_slip_mean_m_s": 0.02,
  "tilt_p95_deg": 10.0,
  "joint_velocity_rms_rad_s": 0.15,
  "nonfoot_contact_fraction": 0.01,
}
WALK_LIMITS = {
  "forward_error_mean_m_s": 0.05,
  "lateral_speed_absolute_mean_m_s": 0.05,
  "yaw_rate_absolute_mean_rad_s": 0.10,
  "foot_contact_duty_min": 0.15,
  "foot_contact_duty_max": 0.85,
  "foot_contact_duty_spread_max": 0.30,
  "footfalls_per_leg_min": 2,
  "contact_slip_mean_m_s": 0.10,
  "tilt_p95_deg": 15.0,
  "nonfoot_contact_fraction": 0.01,
}


def profile_command(profile: str, elapsed_s: float) -> tuple[float, float, float]:
  if profile == "stand":
    speed = 0.0
  elif profile == "forward":
    speed = 0.25
  elif profile == "backward":
    speed = -0.25
  elif profile == "stop_start":
    speed = 0.25 if int(elapsed_s // 5) % 2 else 0.0
  else:
    raise ValueError(f"Unknown behavior profile: {profile}")
  return speed, 0.0, 0.0


def validate_behavior_identity(saved: dict, current: dict) -> dict:
  """Permit flat subterrain replacement and reward exclusion, nothing else."""
  left, right = deepcopy(saved), deepcopy(current)
  old_env, new_env = left["environment"], right["environment"]
  rewards = {
    "saved": old_env.pop("rewards"),
    "diagnostic": new_env.pop("rewards"),
  }
  terrain = {}
  for label, environment in (("saved", old_env), ("diagnostic", new_env)):
    generator = environment["terrain_generator"]
    terrain[label] = None if generator is None else generator.pop("sub_terrains")
  if terrain["diagnostic"] is not None:
    if terrain["saved"] is None or set(terrain["saved"]) != set(terrain["diagnostic"]):
      raise ValueError("Diagnostic terrain must preserve the saved family names")
    for name, family in terrain["diagnostic"].items():
      if family["type"] != "mjlab.terrains.primitive_terrains.BoxFlatTerrainCfg":
        raise ValueError("Only BoxFlatTerrainCfg may replace diagnostic terrain")
      original = terrain["saved"][name]
      for key in ("size", "proportion"):
        if family[key] != original[key]:
          raise ValueError(f"Diagnostic terrain changed {key}")
  if left != right:
    changed = sorted(
      key for key in set(old_env) | set(new_env) if old_env.get(key) != new_env.get(key)
    )
    raise ValueError(
      "Checkpoint identity mismatch outside diagnostic terrain/rewards: "
      + (", ".join(changed) or "task/command semantics")
    )
  return {
    "terrain_subfamilies": terrain,
    "rewards": rewards,
    "reason": (
      "Replace every terrain tile with a flat box for behavior isolation. "
      "Reward identity is excluded because no returns or rewards are graded. "
      "All other recorded task, physics, motor, observation and action identities "
      "must match. Actor/normalizer state loads strictly; checkpoint is unmodified."
    ),
  }


def contact_bouts(contact: np.ndarray, dt: float) -> dict:
  """Count observed landings; retain boundary censoring for bout durations."""
  contact = np.asarray(contact, dtype=bool)
  if not len(contact):
    return {"footfalls": 0, "air": [], "contact": []}
  starts = np.r_[0, np.flatnonzero(contact[1:] != contact[:-1]) + 1]
  ends = np.r_[starts[1:], len(contact)]
  result = {"footfalls": 0, "air": [], "contact": []}
  for start, end in zip(starts, ends, strict=True):
    duration = (end - start) * dt
    state = bool(contact[start])
    record = {
      "duration_s": float(duration),
      "left_censored": bool(start == 0),
      "right_censored": bool(end == len(contact)),
    }
    result["contact" if state else "air"].append(record)
    if not state and end < len(contact) and duration + 1e-9 >= MIN_AIR_TIME_S:
      result["footfalls"] += 1
  return result


def summarize_segment(samples, index, start_s, end_s, command, dt, *, settle_s):
  time = samples["time_s"]
  window = (time > start_s + settle_s + 1e-9) & (time <= end_s + 1e-9)
  valid = window & samples["valid"][:, index]
  expected = int(round((end_s - start_s - settle_s) / dt))
  count = int(valid.sum())
  result = {
    "start_s": start_s,
    "end_s": end_s,
    "excluded_transient_s": settle_s,
    "command_vx_vy_wz": list(command),
    "expected_samples": expected,
    "valid_samples": count,
    "complete": count == expected,
  }
  if not count:
    return {**result, "passed": False, "metrics": None}

  def values(name):
    return samples[name][valid, index]

  velocity = values("velocity")
  contacts = values("contact")
  duty = contacts.mean(axis=0)
  position = values("position")
  # Drift begins at the first post-transient sample, never at a later reset.
  drift = np.linalg.norm(position[:, :2] - position[0, :2], axis=-1)
  bouts = [contact_bouts(contacts[:, foot], dt) for foot in range(4)]
  falls = [bout["footfalls"] for bout in bouts]
  metrics = {
    "actual_mean_vx_vy_wz": velocity.mean(axis=0).tolist(),
    "forward_error_mean_m_s": float(np.abs(velocity[:, 0] - command[0]).mean()),
    "horizontal_speed_mean_m_s": float(np.linalg.norm(velocity[:, :2], axis=-1).mean()),
    "lateral_speed_absolute_mean_m_s": float(np.abs(velocity[:, 1]).mean()),
    "yaw_rate_absolute_mean_rad_s": float(np.abs(velocity[:, 2]).mean()),
    "horizontal_drift_max_m": float(drift.max()),
    "horizontal_displacement_m": float(drift[-1]),
    "base_height_mean_m": float(position[:, 2].mean()),
    "joint_default_rmse_rad": float(np.sqrt(np.mean(values("joint_error") ** 2))),
    "joint_velocity_rms_rad_s": float(np.sqrt(np.mean(values("joint_velocity") ** 2))),
    "tilt_p95_deg": float(np.quantile(values("tilt_deg"), 0.95)),
    "tilt_max_deg": float(values("tilt_deg").max()),
    "foot_contact_fraction": duty.tolist(),
    "foot_vertical_force_abs_mean_N": values("foot_vertical_force").mean(0).tolist(),
    "foot_loaded_contact_fraction_gt_1N": (values("foot_vertical_force") > 1.0)
    .mean(0)
    .tolist(),
    "foot_contact_duty_spread": float(duty.max() - duty.min()),
    "all_four_contact_fraction": float(contacts.all(axis=-1).mean()),
    "nonfoot_contact_fraction": float(values("nonfoot_contact").mean()),
    "contact_slip_mean_m_s": float(values("slip").mean()),
    "footfalls_per_leg": falls,
    "per_leg_bouts": dict(zip(FOOT_ORDER, bouts, strict=True)),
  }
  if command[0] == 0:
    checks = {
      key: metrics[key] <= value
      for key, value in STAND_LIMITS.items()
      if key != "all_four_contact_fraction_min"
    }
    checks["all_four_contact"] = (
      metrics["all_four_contact_fraction"]
      >= STAND_LIMITS["all_four_contact_fraction_min"]
    )
  else:
    checks = {
      key: metrics[key] <= value for key, value in WALK_LIMITS.items() if key in metrics
    }
    checks.update(
      {
        "all_legs_contact_and_swing": bool(
          (duty >= WALK_LIMITS["foot_contact_duty_min"]).all()
          and (duty <= WALK_LIMITS["foot_contact_duty_max"]).all()
        ),
        "contact_duty_balance": metrics["foot_contact_duty_spread"]
        <= WALK_LIMITS["foot_contact_duty_spread_max"],
        "all_legs_footfalls": min(falls) >= WALK_LIMITS["footfalls_per_leg_min"],
      }
    )
  return {
    **result,
    "metrics": metrics,
    "checks": checks,
    "passed": result["complete"] and all(checks.values()),
  }


def summarize_profiles(samples, profiles, dt, *, falls, invalid, horizon_steps):
  duration = horizon_steps * dt
  rows = []
  for index, profile in enumerate(profiles):
    boundaries = (
      [0.0, 5.0, 10.0, 15.0, duration] if profile == "stop_start" else [0.0, duration]
    )
    segments = [
      summarize_segment(
        samples,
        index,
        start,
        end,
        profile_command(profile, start),
        dt,
        settle_s=1.0 if profile == "stop_start" else 2.0,
      )
      for start, end in zip(boundaries[:-1], boundaries[1:], strict=True)
    ]
    rows.append(
      {
        "environment_index": index,
        "profile": profile,
        "fall": bool(falls[index]),
        "invalid_state": bool(invalid[index]),
        "segments": segments,
        "passed": not bool(falls[index] or invalid[index])
        and all(segment["passed"] for segment in segments),
      }
    )
  return rows


def fresh_command_observations(env):
  """Public per-group compute avoids the manager's stale-command cache."""
  from tensordict import TensorDict

  return TensorDict(
    {
      name: env.observation_manager.compute_group(name, update_history=False)
      for name in env.cfg.observations
    },
    batch_size=[env.num_envs],
  )


@torch.inference_mode()
def evaluate_profiles(env, wrapped, policy):
  from pochi_rl.lp_acrl.physics import invalid_physics_mask
  from pochi_rl.robot import FOOT_GEOMS, FOOT_SITES

  if env.cfg.auto_reset:
    raise ValueError("Behavior evaluation requires auto_reset=False")
  if env.num_envs % len(PROFILES):
    raise ValueError("num_envs must be a positive multiple of four")
  if not np.isclose(env.max_episode_length * env.step_dt, 20.0):
    raise ValueError("Behavior profiles currently require the20s checkpoint horizon")
  profiles = [PROFILES[i % 4] for i in range(env.num_envs)]
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
  return summarize_profiles(
    samples,
    profiles,
    env.step_dt,
    falls=falls.cpu().numpy(),
    invalid=invalid.cpu().numpy(),
    horizon_steps=env.max_episode_length,
  )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default="Pochi-LP-ACRL-Blind-v0")
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--device", default="cuda:2")
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--seed", type=int, default=456)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.num_envs < 4 or args.num_envs % 4:
    parser.error("--num-envs must be a positive multiple of four")
  if not args.checkpoint.is_file():
    parser.error("--checkpoint must name an existing file")
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.terrains.primitive_terrains import BoxFlatTerrainCfg
  from mjlab.utils.torch import configure_torch_backends
  from rsl_rl.runners import OnPolicyRunner

  import pochi_rl  # noqa: F401
  from pochi_rl.robot import PAYLOAD_MASS_KG, PAYLOAD_POS

  configure_torch_backends()
  saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  infos = saved.get("infos", {})
  if infos.get("lp_acrl_checkpoint_version") != 1:
    parser.error("Expected an independent LP-ACRL checkpoint")
  identity = infos["lp_acrl"]["task_space"]
  physical = identity["environment"]
  cfg = load_env_cfg(args.task, play=True)
  for key, value in physical["physics"].items():
    if key != "type":
      setattr(cfg.sim.mujoco, key, value)
  cfg.decimation = physical["decimation"]
  saved_sensors = {sensor["name"]: sensor for sensor in physical["sensors"]}
  for sensor in cfg.scene.sensors:
    if sensor.name in saved_sensors and "history_length" in saved_sensors[sensor.name]:
      sensor.history_length = saved_sensors[sensor.name]["history_length"]
  generator = cfg.scene.terrain.terrain_generator
  if generator is not None:
    generator.sub_terrains = {
      name: BoxFlatTerrainCfg(size=family.size, proportion=family.proportion)
      for name, family in generator.sub_terrains.items()
    }
  cfg.seed = args.seed
  cfg.scene.num_envs = args.num_envs
  cfg.auto_reset = False
  cfg.commands["base_velocity"].debug_vis = False
  agent = load_rl_cfg(args.task)
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    term = env.command_manager.get_term("base_velocity")
    overrides = validate_behavior_identity(identity, term.checkpoint_identity())
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = load_runner_cls(args.task)(wrapped, asdict(agent), device=args.device)
    # The recorded narrow identity check above replaces only the custom runner's
    # equality check; this is the native strict actor+normalizer loading API.
    OnPolicyRunner.load(
      runner,
      str(args.checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=args.device,
    )
    rows = evaluate_profiles(
      env, wrapped, runner.get_inference_policy(device=args.device)
    )
    robot = env.scene["robot"]
    masses = env.sim.model.body_mass
    bodies = robot.indexing.body_ids.long()
    base_local, _ = robot.find_bodies(("base_link",), preserve_order=True)
    base_global = bodies[base_local[0]]
    total_mass = masses[:, bodies].sum(-1).detach().cpu().tolist()
    base_mass = masses[:, base_global].detach().cpu().tolist()
    summary = {
      profile: {
        "episodes": sum(row["profile"] == profile for row in rows),
        "passed": sum(row["profile"] == profile and row["passed"] for row in rows),
        "falls": sum(row["profile"] == profile and row["fall"] for row in rows),
        "invalid": sum(
          row["profile"] == profile and row["invalid_state"] for row in rows
        ),
      }
      for profile in PROFILES
    }
    report = {
      "task": args.task,
      "checkpoint": str(args.checkpoint.resolve()),
      "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
      "seed": args.seed,
      "num_envs": env.num_envs,
      "device": args.device,
      "control_dt_s": env.step_dt,
      "horizon_s": 20.0,
      "identity_overrides": overrides,
      "observation_terms": {
        group: list(group_cfg.terms) for group, group_cfg in cfg.observations.items()
      },
      "height_scan_present": any(
        "height" in name.lower()
        for group in cfg.observations.values()
        for name in group.terms
      ),
      "payload": {
        "central_ballast_already_in_robot_asset_kg": PAYLOAD_MASS_KG,
        "position_in_base_frame_m": PAYLOAD_POS,
        "realized_robot_mass_kg": total_mass,
        "realized_base_mass_kg": base_mass,
      },
      "metric_conventions": {
        "purpose": "Flat-terrain behavior diagnostics; not paper scores",
        "profile_commands": (
          "stand 0; forward +.25; backward -.25; "
          "stop/start 0,+.25,0,+.25 in 5s blocks; lateral/yaw 0"
        ),
        "transient_exclusion_s": {
          "constant_profiles": 2.0,
          "stop_start_each_segment": 1.0,
        },
        "standing_limits": STAND_LIMITS,
        "walking_limits": WALK_LIMITS,
        "criteria_origin": (
          "Explicit Pochi engineering acceptance conventions; "
          "not paper criteria or hardware guarantees"
        ),
        "sampling": (
          "50Hz post-control-step, before manual terminal reset; first episode "
          "only; invalid and fall samples excluded"
        ),
        "contact": (
          "Contact sensor found flag; brief sub-control-step contacts may be missed"
        ),
        "footfall_minimum_observed_air_time_s": MIN_AIR_TIME_S,
        "bouts": (
          "Observed run lengths; boundary bouts marked censored; "
          "no inference about unobserved swing phases"
        ),
        "drift": "Distance from first valid post-transient sample in each segment",
        "foot_order": FOOT_ORDER,
        "rewards": (
          "Reward values are not reported or graded; "
          "nonfinite reward still invalidates a transition"
        ),
      },
      "summary": summary,
      "episodes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))
  finally:
    env.close()


if __name__ == "__main__":
  main()
