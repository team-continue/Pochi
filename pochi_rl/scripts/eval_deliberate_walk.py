"""Compare speed, support, slipping and cadence at identical commands.

Example:
  uv run python scripts/eval_deliberate_walk.py --task Pochi-Deliberate-Walk-v0 \
    --checkpoint logs/rsl_rl/pochi_deliberate_walk/<run>/model_1499.pt
"""

from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch


def evaluate(
  env, wrapped, policy, speed, seconds, trace_path=None, video_path=None, reset=True
):
  term = env.command_manager.get_term("base_velocity")
  term.cfg.ranges.lin_vel_x = (speed, speed)
  term.cfg.ranges.lin_vel_y = (0.0, 0.0)
  term.cfg.ranges.ang_vel_z = (0.0, 0.0)
  term.cfg.heading_command = False
  term.cfg.rel_standing_envs = 0.0
  term.cfg.rel_forward_envs = 0.0
  term.cfg.rel_world_envs = 0.0
  term.cfg.init_velocity_prob = 0.0
  robot = env.scene["robot"]
  from pochi_rl.robot.pochi_constants import FOOT_SITES

  site_ids, _ = robot.find_sites(FOOT_SITES, preserve_order=True)
  sensor = env.scene["feet_ground_contact"]
  samples = {k: [] for k in ("vx", "joint", "support", "slip", "touchdowns")}
  falls = 0
  falls_after_settling = 0
  fall_events = []
  trace = []
  writer = None
  if video_path:
    import imageio.v2 as imageio

    writer = imageio.get_writer(str(video_path), fps=25)
  steps = round(seconds / env.step_dt)
  warmup = round(2.8 / env.step_dt)
  with writer if writer is not None else nullcontext(), torch.inference_mode():
    if reset:
      env.reset()
    else:
      term.reset(torch.arange(env.num_envs, device=env.device))
    obs = wrapped.get_observations()
    air_steps = torch.zeros(env.num_envs, 4, device=env.device)
    start = robot.data.root_link_pos_w.clone()
    for step in range(steps + warmup):
      obs, _, done, _ = wrapped.step(policy(obs))
      count = int(done.sum())
      falls += count
      if count:
        fall_events.append({"time_s": round(step * env.step_dt, 3), "count": count})
      if step >= warmup:
        falls_after_settling += count
      contact = sensor.data.found > 0
      # Count a footfall only after >= 0.10 s airborne; ignore contact chatter.
      landed = contact & (air_steps * env.step_dt >= 0.10)
      air_steps = torch.where(contact, 0.0, air_steps + 1)
      air_steps[done.bool()] = 0.0
      if step == warmup:
        start = robot.data.root_link_pos_w.clone()
      if step >= warmup:
        if trace_path:
          trace.append(
            torch.cat(
              (
                contact[0].float(),
                robot.data.site_pos_w[0, site_ids, 2] - env.scene.env_origins[0, 2],
                robot.data.root_link_lin_vel_b[0],
              )
            )
            .cpu()
            .numpy()
          )
        if writer is not None and (step - warmup) % 2 == 0:
          writer.append_data(env.render())
        samples["vx"].append(robot.data.root_link_lin_vel_b[:, 0].clone())
        samples["joint"].append(robot.data.joint_vel.abs().clone())
        samples["support"].append(contact.sum(-1).clone())
        slip = robot.data.site_lin_vel_w[:, site_ids, :2].norm(dim=-1)
        samples["slip"].append((slip * contact).sum(-1) / contact.sum(-1).clamp(min=1))
        samples["touchdowns"].append(landed)
    drift = (robot.data.root_link_pos_w - start)[:, :2].norm(dim=-1).mean()
  if trace_path:
    np.savez_compressed(
      trace_path,
      samples=np.asarray(trace),
      dt=env.step_dt,
      columns="contact_FL FR RL RR, height_FL FR RL RR, vx vy vz",
    )
  data = {k: torch.stack(v).float() for k, v in samples.items()}
  return {
    "command_vx_m_s": speed,
    "actual_vx_m_s": float(data["vx"].mean()),
    "joint_speed_mean_rad_s": float(data["joint"].mean()),
    "joint_speed_p95_rad_s": float(data["joint"].quantile(0.95)),
    "three_or_more_feet_fraction": float((data["support"] >= 3).float().mean()),
    "four_feet_fraction": float((data["support"] == 4).float().mean()),
    "contact_slip_m_s": float(data["slip"].mean()),
    "footfalls_per_second": float(data["touchdowns"].sum((0, 2)).mean() / seconds),
    "footfalls_per_second_per_leg": (
      data["touchdowns"].sum(0).mean(0) / seconds
    ).tolist(),
    "falls": falls,
    "falls_after_settling": falls_after_settling,
    "fall_events": fall_events,
    "displacement_m": float(drift),
    "seconds_per_env": seconds,
    "settling_seconds": warmup * env.step_dt,
    "num_envs": env.num_envs,
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default="Pochi-Deliberate-Walk-v0")
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--device", default="cuda:2")
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--seconds", type=float, default=14.0)
  parser.add_argument("--speeds", type=float, nargs="+", default=[0.06, 0.1, 0.14, 0.0])
  parser.add_argument("--output", type=Path)
  parser.add_argument(
    "--continuous",
    action="store_true",
    help="Switch commands without resetting the robot",
  )
  parser.add_argument("--trace", type=Path, help="First command, first env, NPZ trace")
  parser.add_argument("--video", type=Path, help="First command, MP4 video")
  args = parser.parse_args()
  if args.num_envs < 1 or args.seconds <= 0:
    parser.error("--num-envs and --seconds must be positive")
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends

  import pochi_rl  # noqa: F401

  configure_torch_backends()
  cfg = load_env_cfg(args.task, play=True)
  cfg.seed = 42
  cfg.scene.num_envs = args.num_envs
  cfg.viewer.width = 960
  cfg.viewer.height = 540
  cfg.viewer.max_extra_envs = 0
  cfg.viewer.elevation = -20.0
  cfg.viewer.azimuth = 130.0
  cfg.commands["base_velocity"].debug_vis = False
  agent = load_rl_cfg(args.task)
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
    for index, speed in enumerate(args.speeds):
      result = evaluate(
        env,
        wrapped,
        policy,
        speed,
        args.seconds,
        args.trace if index == 0 else None,
        args.video if index == 0 else None,
        reset=index == 0 or not args.continuous,
      )
      results.append(result)
      print(json.dumps(result, indent=2), flush=True)
    report = {"task": args.task, "checkpoint": str(args.checkpoint), "results": results}
    if args.output:
      args.output.write_text(json.dumps(report, indent=2) + "\n")
  finally:
    env.close()


if __name__ == "__main__":
  main()
