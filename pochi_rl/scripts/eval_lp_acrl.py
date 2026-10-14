"""Evaluate an independent LP-ACRL checkpoint with equal episodes per task.

Examples:
  uv run python scripts/eval_lp_acrl.py --task Pochi-LP-ACRL-Flat-v0 \
    --checkpoint logs/rsl_rl/pochi_lp_acrl_flat/<run>/model_2999.pt
  uv run python scripts/eval_lp_acrl.py --task Pochi-LP-ACRL-Rough-v0 \
    --checkpoint <checkpoint> --task-ids 0 --video /tmp/lp_acrl.mp4

The default evaluates the entire task catalog, independently of training's
sampling distribution. A task subset is explicitly labeled as screening.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

FORWARD_ERROR_FLOOR = 0.05
YAW_ERROR_FLOOR = 0.10
FOOTFALL_AIR_TIME_S = 0.10
MAX_BASE_CONTACT_FRACTION = 0.05
CENTRAL_PLATFORM_WIDTH_M = 1.2
CENTRAL_PLATFORM_TERRAINS = (
  "ascending_stairs",
  "descending_stairs",
  "uphill",
  "downhill",
)


@torch.inference_mode()
def evaluate_batch(env, wrapped, policy, task_ids, *, video_path=None):
  """Measure only each environment's first episode, including timeout state.

  Automatic reset must be disabled. Fall transitions are excluded from the
  non-falling tracking interval; timeout transitions are included. Done envs
  are reset explicitly after metrics are read, then excluded from all sums.
  Keep batch resets in inference mode because sensors allocate cached tensors
  during inference-mode steps and reuse them in later batches.
  """
  from pochi_rl.lp_acrl.metrics import epte_sp, percentage_tracking_error
  from pochi_rl.lp_acrl.physics import invalid_physics_mask
  from pochi_rl.robot.pochi_constants import FOOT_SITES

  if env.cfg.auto_reset:
    raise ValueError("Evaluation requires auto_reset=False for terminal metrics")
  term = env.command_manager.get_term("base_velocity")
  if len(task_ids) != env.num_envs:
    raise ValueError("Provide exactly one fixed task ID per environment")
  term.set_evaluation(task_ids=task_ids)
  env.reset()
  obs = wrapped.get_observations()
  command = term.command.clone()
  robot = env.scene["robot"]
  site_ids, _ = robot.find_sites(FOOT_SITES, preserve_order=True)
  contact_sensor = env.scene["feet_ground_contact"]
  nonfoot_sensor = env.scene["limb_ground_contact"]
  nonfoot_names = nonfoot_sensor.primary_names
  base_indices = [
    index
    for index, name in enumerate(nonfoot_names)
    if name.rsplit("/", 1)[-1] == "base_collision"
  ]
  if len(base_indices) != 1:
    raise ValueError("Expected one named base_collision contact sensor primary")
  base_index = base_indices[0]
  horizon = int(env.max_episode_length)
  n = env.num_envs
  device = env.device
  terrain_indices = term.catalog.task_indices()[task_ids.cpu(), 2].tolist()
  terrain_names = [term.catalog.terrains[index] for index in terrain_indices]
  platform_applicable = torch.tensor(
    [name in CENTRAL_PLATFORM_TERRAINS for name in terrain_names], device=device
  )
  # Keep the first episode's tile origin even while inactive worlds reset.
  platform_origins = env.scene.env_origins[:, :2].clone()
  active = torch.ones(n, device=device, dtype=torch.bool)
  alive_steps = torch.zeros(n, device=device, dtype=torch.long)
  episode_steps = torch.zeros_like(alive_steps)
  falls = torch.zeros_like(active)
  timeouts = torch.zeros_like(active)
  invalid = torch.zeros_like(active)
  reward_sum = torch.zeros(n, device=device)
  error_sum = torch.zeros(n, 2, device=device)
  velocity_sum = torch.zeros(n, 3, device=device)
  absolute_velocity_sum = torch.zeros_like(velocity_sum)
  contact_sum = torch.zeros(n, 4, device=device)
  support_sum = torch.zeros(n, device=device)
  nonfoot_sum = torch.zeros(n, len(nonfoot_names), device=device)
  any_nonfoot_sum = torch.zeros(n, device=device)
  slip_sum = torch.zeros(n, device=device)
  footfalls = torch.zeros(n, 4, device=device, dtype=torch.long)
  air_steps = torch.zeros_like(footfalls)
  outside_platform_steps = torch.zeros_like(alive_steps)
  start = robot.data.root_link_pos_w[:, :2].clone()
  end = start.clone()
  writer = None
  frame_every = max(1, round(0.04 / env.step_dt))
  if video_path is not None:
    import imageio.v2 as imageio

    video_path.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(str(video_path), fps=1.0 / (frame_every * env.step_dt))
  try:
    with torch.inference_mode():
      for step in range(horizon):
        actions = policy(obs)
        actions[~active] = 0.0
        obs, reward, done, _ = wrapped.step(actions)
        terminated = env.reset_terminated.clone().bool()
        truncated = env.reset_time_outs.clone().bool()
        done = done.bool()
        velocity = torch.stack(
          (
            robot.data.root_link_lin_vel_b[:, 0],
            robot.data.root_link_lin_vel_b[:, 1],
            robot.data.root_link_ang_vel_b[:, 2],
          ),
          dim=-1,
        )
        valid = torch.isfinite(reward) & ~invalid_physics_mask(robot.data)
        # Invalid states fail the episode. Mask metric inputs, without touching
        # simulator state, so even huge finite values cannot overflow the report.
        velocity = torch.where(valid[:, None], velocity, 0.0)
        errors = torch.stack(
          (
            percentage_tracking_error(
              velocity[:, 0],
              command[:, 0],
              denominator_floor=FORWARD_ERROR_FLOOR,
            ),
            percentage_tracking_error(
              velocity[:, 2],
              command[:, 2],
              denominator_floor=YAW_ERROR_FLOOR,
            ),
          ),
          dim=-1,
        )
        contact = contact_sensor.data.found > 0
        if contact.shape != (n, 4):
          raise RuntimeError(f"Expected four foot contacts, got {contact.shape}")
        # ContactSensor.primary_names is the documented slot-to-geometry map.
        # Reduce arbitrary per-primary slots without assuming the base is last.
        nonfoot = (
          (nonfoot_sensor.data.found > 0).reshape(n, len(nonfoot_names), -1).any(-1)
        )
        landed = contact & (air_steps * env.step_dt >= FOOTFALL_AIR_TIME_S)
        air_steps = torch.where(contact, 0, air_steps + 1)
        slip = robot.data.site_lin_vel_w[:, site_ids, :2].norm(dim=-1)
        slip = (slip * contact).sum(-1) / contact.sum(-1).clamp_min(1)
        valid &= torch.isfinite(slip)
        valid &= torch.isfinite(contact_sensor.data.found).reshape(n, -1).all(-1)
        valid &= torch.isfinite(nonfoot_sensor.data.found).reshape(n, -1).all(-1)
        newly_invalid = active & ~valid
        invalid |= newly_invalid
        alive = active & ~terminated & valid
        outside_platform = (
          (robot.data.root_link_pos_w[:, :2] - platform_origins).abs()
          > CENTRAL_PLATFORM_WIDTH_M / 2
        ).any(-1)
        outside_platform_steps += (
          alive & platform_applicable & outside_platform
        ).long()
        slip = torch.nan_to_num(slip, nan=0.0, posinf=0.0, neginf=0.0)
        nonfoot_sum += nonfoot.float() * alive[:, None]
        any_nonfoot_sum += nonfoot.any(-1).float() * alive
        error_sum += errors * alive[:, None]
        velocity_sum += velocity * alive[:, None]
        absolute_velocity_sum += velocity.abs() * alive[:, None]
        contact_sum += contact.float() * alive[:, None]
        support_sum += (contact.sum(-1) >= 3).float() * alive
        slip_sum += slip * alive
        footfalls += landed.long() * alive[:, None]
        alive_steps += alive.long()
        episode_steps += active.long()
        reward_sum += torch.where(active & valid, reward, 0.0)
        end[active] = robot.data.root_link_pos_w[active, :2]
        falls |= active & terminated
        timeouts |= active & truncated
        if (
          writer is not None and bool(active[0] & valid[0]) and step % frame_every == 0
        ):
          writer.append_data(env.render())
        finished = done | newly_invalid
        active &= ~finished
        if not bool(active.any()):
          break
        if bool(finished.any()):
          # Invalid physics explicitly fails its episode. Discard it before
          # the next policy call; never repair or sanitize simulator state.
          # Following episodes run only as inactive simulation padding.
          env.reset(env_ids=finished.nonzero(as_tuple=False).flatten())
          obs = wrapped.get_observations()
          air_steps[finished] = 0
  finally:
    if writer is not None:
      writer.close()

  count = alive_steps.clamp_min(1)
  errors = epte_sp(error_sum / count[:, None], alive_steps[:, None], horizon)
  mean_velocity = velocity_sum / count[:, None]
  mean_abs_velocity = absolute_velocity_sum / count[:, None]
  paper_success = (alive_steps > 900) & (errors < 0.30).all(-1) & ~invalid
  # These guards are additional Pochi diagnostics, not part of paper Eq. (8).
  forward_motion = (mean_abs_velocity[:, 0] > 0.005) & (
    mean_abs_velocity[:, 0] >= 0.25 * command[:, 0].abs()
  )
  yaw_motion = (mean_abs_velocity[:, 2] > 0.03) & (
    mean_abs_velocity[:, 2] >= 0.25 * command[:, 2].abs()
  )
  gait_observed = (footfalls.sum(-1) >= 2) & ((footfalls > 0).sum(-1) >= 2)
  motion_observed = forward_motion | yaw_motion
  base_fraction = nonfoot_sum[:, base_index] / count
  base_clear = base_fraction <= MAX_BASE_CONTACT_FRACTION
  accepted = paper_success & motion_observed & gait_observed & base_clear
  seconds = (alive_steps * env.step_dt).clamp_min(env.step_dt)
  results = []
  for i, task_id in enumerate(task_ids.tolist()):
    displacement = float((end[i] - start[i]).norm())
    if bool(invalid[i]) and not math.isfinite(displacement):
      displacement = None
    platform_measured = bool(platform_applicable[i] & (alive_steps[i] > 0))
    results.append(
      {
        "task_id": task_id,
        "terrain": terrain_names[i],
        "command_vx_vy_wz": command[i].tolist(),
        "actual_mean_vx_vy_wz": mean_velocity[i].tolist(),
        "actual_mean_absolute_vx_vy_wz": mean_abs_velocity[i].tolist(),
        "epte_sp_forward": float(errors[i, 0]),
        "epte_sp_yaw": float(errors[i, 1]),
        "alive_steps": int(alive_steps[i]),
        "episode_steps": int(episode_steps[i]),
        "fall": bool(falls[i]),
        "timeout": bool(timeouts[i]),
        "horizon_reached": bool(episode_steps[i] >= horizon),
        "invalid_state": bool(invalid[i]),
        "episodic_reward": float(reward_sum[i]),
        "displacement_xy_m": displacement,
        "central_platform_applicable": bool(platform_applicable[i]),
        "central_platform_outside_base_fraction": (
          float(outside_platform_steps[i] / count[i]) if platform_measured else None
        ),
        "ever_left_platform": (
          bool(outside_platform_steps[i] > 0) if platform_measured else None
        ),
        "foot_contact_fraction": (contact_sum[i] / count[i]).tolist(),
        "three_or_more_feet_fraction": float(support_sum[i] / count[i]),
        "contact_slip_m_s": float(slip_sum[i] / count[i]),
        "any_nonfoot_contact_fraction": float(any_nonfoot_sum[i] / count[i]),
        "base_contact_fraction": float(base_fraction[i]),
        "nonfoot_contact_fraction_by_geom": {
          name: float(nonfoot_sum[i, j] / count[i])
          for j, name in enumerate(nonfoot_names)
        },
        "base_support_guard_passed": bool(base_clear[i]),
        "footfalls_per_leg": footfalls[i].tolist(),
        "footfalls_per_second": float(footfalls[i].sum() / seconds[i]),
        "paper_style_success": bool(paper_success[i]),
        "motion_observed": bool(motion_observed[i]),
        "gait_observed": bool(gait_observed[i]),
        "accepted_locomotion": bool(accepted[i]),
      }
    )
  return results


def summarize(results):
  n = len(results)
  platform_rows = [
    row
    for row in results
    if row.get("central_platform_outside_base_fraction") is not None
  ]
  return {
    "episodes": n,
    "falls": sum(row["fall"] for row in results),
    "timeouts": sum(row["timeout"] for row in results),
    "invalid_episodes": sum(row["invalid_state"] for row in results),
    "central_platform_applicable_episodes": sum(
      row.get("central_platform_applicable", False) for row in results
    ),
    "central_platform_measured_episodes": len(platform_rows),
    "mean_outside_central_platform_base_fraction": (
      sum(row["central_platform_outside_base_fraction"] for row in platform_rows)
      / len(platform_rows)
      if platform_rows
      else None
    ),
    "ever_left_platform_rate": (
      sum(row["ever_left_platform"] for row in platform_rows) / len(platform_rows)
      if platform_rows
      else None
    ),
    "horizon_reached_episodes": sum(row["horizon_reached"] for row in results),
    "base_support_guard_rejected_episodes": sum(
      not row["base_support_guard_passed"] for row in results
    ),
    "mean_any_nonfoot_contact_fraction": sum(
      row["any_nonfoot_contact_fraction"] for row in results
    )
    / n,
    "mean_base_contact_fraction": sum(row["base_contact_fraction"] for row in results)
    / n,
    "paper_style_success_rate": sum(row["paper_style_success"] for row in results) / n,
    "accepted_locomotion_rate": sum(row["accepted_locomotion"] for row in results) / n,
    "mean_epte_sp_forward": sum(row["epte_sp_forward"] for row in results) / n,
    "mean_epte_sp_yaw": sum(row["epte_sp_yaw"] for row in results) / n,
    "mean_alive_steps": sum(row["alive_steps"] for row in results) / n,
    "mean_actual_vx_m_s": sum(row["actual_mean_vx_vy_wz"][0] for row in results) / n,
    "mean_contact_slip_m_s": sum(row["contact_slip_m_s"] for row in results) / n,
    "mean_footfalls_per_second": sum(row["footfalls_per_second"] for row in results)
    / n,
  }


def summarize_by_terrain(results):
  """Keep platform coverage visible beside each terrain family's success rate."""
  return [
    {"terrain": terrain, **summarize([r for r in results if r["terrain"] == terrain])}
    for terrain in dict.fromkeys(row["terrain"] for row in results)
  ]


def restore_checkpoint_rewards(cfg, saved_rewards):
  """Restore supported launcher options and refuse silently changed returns."""
  from pochi_rl.mjlab.lp_acrl_commands import _config_identity

  # Older runs predate the training-only, zero-valued physics assertion. Their
  # reward identity must stay exact; evaluation validates state independently.
  if "physics_validity" not in saved_rewards:
    cfg.rewards.pop("physics_validity", None)
  for name in ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"):
    params = saved_rewards[name]["params"]
    if "std" in params:
      cfg.rewards[name].params["std"] = params["std"]
    else:
      cfg.rewards[name].params.pop("std", None)
  if "termination" in saved_rewards:
    from mjlab.envs import mdp
    from mjlab.managers.reward_manager import RewardTermCfg

    cfg.rewards["termination"] = RewardTermCfg(
      func=mdp.is_terminated, weight=saved_rewards["termination"]["weight"]
    )
  if _config_identity(cfg.rewards) != saved_rewards:
    raise ValueError(
      "Checkpoint reward configuration contains unsupported differences; "
      "cannot report comparable episodic returns. Only tracking std and "
      "termination-penalty overrides and the absent legacy physics assertion "
      "are restored by this evaluator."
    )


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--task",
    choices=(
      "Pochi-LP-ACRL-Flat-v0",
      "Pochi-LP-ACRL-Rough-v0",
      "Pochi-LP-ACRL-Blind-v0",
      "Pochi-LP-ACRL-Blind-v1",
    ),
    default="Pochi-LP-ACRL-Rough-v0",
  )
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--device", default="cuda:2")
  parser.add_argument("--episodes-per-task", type=int, default=4)
  parser.add_argument("--num-envs", type=int, default=600)
  parser.add_argument("--seed", type=int, default=20261006)
  parser.add_argument(
    "--terrain-seed", type=int, help="Override procedural terrain for held-out testing"
  )
  parser.add_argument("--output", type=Path)
  parser.add_argument("--task-ids", type=int, nargs="+", help="Subset for screening")
  parser.add_argument("--video", type=Path, help="MP4; requires one --task-ids value")
  args = parser.parse_args()
  if args.episodes_per_task < 1 or args.num_envs < 1:
    parser.error("--episodes-per-task and --num-envs must be positive")
  if not args.checkpoint.is_file():
    parser.error(f"Checkpoint does not exist: {args.checkpoint}")
  if args.task_ids is not None and (
    len(set(args.task_ids)) != len(args.task_ids) or min(args.task_ids) < 0
  ):
    parser.error("--task-ids must be distinct nonnegative integers")
  if args.video and (args.task_ids is None or len(args.task_ids) != 1):
    parser.error("--video requires exactly one --task-ids value")

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends

  import pochi_rl  # noqa: F401
  from pochi_rl.lp_acrl import FLAT_TASK_CATALOG, ROUGH_TASK_CATALOG
  from pochi_rl.lp_acrl.physics import PHYSICS_LIMITS

  configure_torch_backends()
  cfg = load_env_cfg(args.task, play=True)
  # These two task-scale adaptations are configurable by the fresh-run
  # launcher. Recover their exact saved representation before identity checks.
  saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  metadata = saved.get("infos") or {}
  if metadata.get("lp_acrl_checkpoint_version") != 1:
    parser.error("Expected an independent LP-ACRL checkpoint")
  saved_environment = metadata["lp_acrl"]["task_space"]["environment"]
  saved_rewards = saved_environment["rewards"]
  for name, value in saved_environment["physics"].items():
    if name != "type":
      setattr(cfg.sim.mujoco, name, value)
  cfg.decimation = saved_environment["decimation"]
  saved_sensors = {sensor["name"]: sensor for sensor in saved_environment["sensors"]}
  for sensor in cfg.scene.sensors:
    saved_sensor = saved_sensors[sensor.name]
    if "history_length" in saved_sensor:
      sensor.history_length = saved_sensor["history_length"]
  try:
    restore_checkpoint_rewards(cfg, saved_rewards)
  except ValueError as exc:
    parser.error(str(exc))
  cfg.seed = args.seed
  terrain_generator = cfg.scene.terrain.terrain_generator
  if args.terrain_seed is not None:
    if terrain_generator is None:
      parser.error("--terrain-seed only applies to the rough terrain task")
    terrain_generator.seed = args.terrain_seed
  cfg.auto_reset = False
  cfg.commands["base_velocity"].debug_vis = False
  agent = load_rl_cfg(args.task)
  catalog = (
    ROUGH_TASK_CATALOG if cfg.commands["base_velocity"].rough else FLAT_TASK_CATALOG
  )
  selected_ids = args.task_ids or list(range(catalog.num_tasks))
  if max(selected_ids) >= catalog.num_tasks:
    parser.error(f"Task IDs must be below {catalog.num_tasks}")
  jobs = [task_id for _ in range(args.episodes_per_task) for task_id in selected_ids]
  cfg.scene.num_envs = min(args.num_envs, len(jobs))
  cfg.viewer.width = 960
  cfg.viewer.height = 540
  cfg.viewer.max_extra_envs = 0
  cfg.viewer.elevation = -20.0
  cfg.viewer.azimuth = 130.0
  env = ManagerBasedRlEnv(
    cfg=cfg, device=args.device, render_mode="rgb_array" if args.video else None
  )
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = load_runner_cls(args.task)(wrapped, asdict(agent), device=args.device)
    runner.load(
      str(args.checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=args.device,
    )
    policy = runner.get_inference_policy(device=args.device)
    results = []
    num_batches = math.ceil(len(jobs) / env.num_envs)
    for batch, offset in enumerate(range(0, len(jobs), env.num_envs)):
      batch_ids = jobs[offset : offset + env.num_envs]
      valid = len(batch_ids)
      padded_ids = batch_ids + [batch_ids[-1]] * (env.num_envs - valid)
      task_ids = torch.tensor(padded_ids, device=env.device, dtype=torch.long)
      rows = evaluate_batch(
        env,
        wrapped,
        policy,
        task_ids,
        video_path=args.video if batch == 0 else None,
      )[:valid]
      for local, row in enumerate(rows):
        row["episode_index"] = (offset + local) // len(selected_ids)
      results.extend(rows)
      print(
        json.dumps(
          {
            "batch": batch + 1,
            "batches": num_batches,
            "completed_episodes": len(results),
            **summarize(rows),
          }
        ),
        flush=True,
      )
    indices = catalog.task_indices().tolist()
    bounds = catalog.task_bounds().tolist()
    by_task = []
    for task_id in selected_ids:
      rows = [row for row in results if row["task_id"] == task_id]
      by_task.append(
        {
          "task_id": task_id,
          "indices_forward_yaw_terrain_level": indices[task_id],
          "terrain": catalog.terrains[indices[task_id][2]],
          "forward_bounds_m_s": bounds[task_id][0],
          "yaw_magnitude_bounds_rad_s": bounds[task_id][1],
          **summarize(rows),
        }
      )
    report = {
      "task": args.task,
      "checkpoint": str(args.checkpoint.resolve()),
      "checkpoint_validation": "LP-ACRL metadata and catalog; strict actor load",
      "seed": args.seed,
      "terrain_seed": None if terrain_generator is None else terrain_generator.seed,
      "terrain_seed_overridden": args.terrain_seed is not None,
      "device": args.device,
      "num_envs": env.num_envs,
      "episodes_per_task": args.episodes_per_task,
      "catalog_size": catalog.num_tasks,
      "selected_task_ids": selected_ids,
      "all_tasks_evaluated": len(selected_ids) == catalog.num_tasks,
      "sampling": "Equal first episodes per selected task; training sampler frozen",
      "control_dt_s": env.step_dt,
      "horizon_steps": int(env.max_episode_length),
      "metric_conventions": {
        "paper": "https://arxiv.org/abs/2601.17428",
        "epte_sp": "(mean_relative_error * alive_steps + K - alive_steps) / K",
        "error": "abs(actual-command)/max(abs(command), floor); no clipping",
        "forward_denominator_floor_m_s": FORWARD_ERROR_FLOOR,
        "yaw_denominator_floor_rad_s": YAW_ERROR_FLOOR,
        "zero_yaw_flat": "abs(actual_yaw_rate) / 0.10; implementation convention",
        "sampling_time": "Post-step, before reset; fall transition excluded",
        "invalid_state_handling": (
          "Nonfinite pose, velocity, joints, reward or contact metrics, or "
          "physics exceeding the generous limits below, fail the episode. "
          "Reset only after recording failure. Undefined displacement is null; "
          "statistics and reward exclude invalid transitions."
        ),
        "invalid_physics_limits": PHYSICS_LIMITS,
        "paper_style_success": "alive_steps > 900 and forward/yaw EPTE-SP < 0.30",
        "footfall_minimum_air_time_s": FOOTFALL_AIR_TIME_S,
        "motion_guard": (
          "mean absolute vx > .005 m/s or yaw > .03 rad/s, and that axis's "
          "mean absolute speed >= 25% of command magnitude"
        ),
        "gait_guard": "At least two footfalls across at least two feet",
        "base_support_guard": (
          "base_collision contact in at most 5% of non-falling control samples"
        ),
        "max_base_contact_fraction": MAX_BASE_CONTACT_FRACTION,
        "contact_sampling": (
          "Post-control-step found flags, before reset; brief contacts between "
          "control samples may be missed. Geometry order uses sensor.primary_names"
        ),
        "central_platform_coverage": {
          "nominal_width_m": CENTRAL_PLATFORM_WIDTH_M,
          "applicable_terrains": CENTRAL_PLATFORM_TERRAINS,
          "outside_condition": "max(abs(base_xy - episode_tile_origin_xy)) > 0.6 m",
          "sampling": "Valid non-falling first-episode steps, including timeout",
          "not_applicable": "Random rough and flat terrain: null coverage/ever-left",
          "zero_valid_steps": "Null coverage/ever-left; no base samples observed",
          "summary": "Mean of measured episode fractions, grouped by terrain",
          "interpretation": (
            "Base-position proxy only; not proof that every foot traversed an "
            "obstacle. Does not change paper success or locomotion grading."
          ),
        },
        "accepted_locomotion": (
          "paper_style_success AND motion_guard AND gait_guard AND base_support_guard"
        ),
        "guards_are_pochi_additions": True,
      },
      "summary": summarize(results),
      "by_task": by_task,
      "by_terrain": summarize_by_terrain(results),
      "episodes": results,
    }
    output = args.output or args.checkpoint.with_suffix(".evaluation.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(output), **report["summary"]}, indent=2))
  finally:
    env.close()


if __name__ == "__main__":
  main()
