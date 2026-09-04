"""Measure what a trained stand-up policy actually does.

Rolls a checkpoint out in the play environment and reports the numbers that
matter for taking it near hardware: how fast the joints turn, how much torque
they pull, whether the robot gets up, and whether it does it on its feet.  The
same figures for the scripted controller are printed alongside, since that is
the behaviour the policy was shaped to reproduce.

  uv run python scripts/eval_standup_policy.py --checkpoint <ckpt.pt>
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from pochi_rl.control.standup import reference_height
from pochi_rl.robot import NOMINAL_BASE_HEIGHT
from pochi_rl.task_spec import POCHI_STANDUP_SPEC as S

TASK_ID = "Pochi-StandUp-Flat-v0"


def parse_args() -> argparse.Namespace:
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--checkpoint", type=Path, required=True)
  p.add_argument("--num-envs", type=int, default=64)
  # Stop short of the episode boundary: the env auto-resets there, and a
  # reading taken after that is of a robot lying back down on purpose.
  p.add_argument(
    "--seconds",
    type=float,
    default=S.reference.settle_s + S.reference.rise_s + 3.0,
  )
  p.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
  return p.parse_args()


def _percentile(values: list[float], q: float) -> float:
  return float(np.percentile(np.asarray(values), q))


def roll_out(args: argparse.Namespace) -> dict[str, float]:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

  import pochi_rl  # noqa: F401

  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.scene.num_envs = args.num_envs
  agent_cfg = load_rl_cfg(TASK_ID)
  raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

  runner_cls = load_runner_cls(TASK_ID)
  assert runner_cls is not None
  runner = runner_cls(env, asdict(agent_cfg), device=args.device)
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(device=args.device)

  robot = raw_env.scene["robot"]
  obs, _ = env.reset()
  dt = S.control.sim_dt * S.control.decimation
  start_height = robot.data.root_link_pos_w[:, 2].mean().item()
  speeds: list[float] = []
  torques: list[float] = []
  heights: list[float] = []
  errors: list[float] = []
  rise_time = float("nan")
  with torch.inference_mode():
    for step in range(int(args.seconds / dt)):
      obs, *_ = env.step(policy(obs))
      speeds.append(robot.data.joint_vel.abs().max().item())
      torques.append(robot.data.actuator_force.abs().max().item())
      h = robot.data.root_link_pos_w[:, 2].mean().item()
      heights.append(h)
      errors.append(
        abs(h - reference_height(step * dt, S.reference.settle_s, S.reference.rise_s))
      )
      if np.isnan(rise_time) and h > NOMINAL_BASE_HEIGHT - 0.02:
        rise_time = step * dt
  env.close()

  return {
    "start_height": start_height,
    "peak_speed": max(speeds),
    "p99_speed": _percentile(speeds, 99),
    "peak_torque": max(torques),
    "final_height": heights[-1],
    "min_height_after_rise": min(heights[len(heights) // 2 :]),
    "rise_time_s": rise_time,
    "max_ramp_error": max(errors),
  }


def scripted_reference(seconds: float) -> dict[str, float]:
  from pochi_rl.control.mujoco_driver import StandUpSim

  sim = StandUpSim()
  dt = sim.model.opt.timestep
  start_height = sim.base_height
  speeds, torques, heights, errors = [], [], [], []
  rise_time = float("nan")
  for step in range(int(seconds / dt)):
    sim.step()
    speeds.append(float(np.abs(sim.data.qvel[6:]).max()))
    torques.append(float(np.abs(sim.data.actuator_force).max()))
    heights.append(sim.base_height)
    errors.append(
      abs(
        sim.base_height
        - reference_height(step * dt, S.reference.settle_s, S.reference.rise_s)
      )
    )
    if np.isnan(rise_time) and sim.base_height > NOMINAL_BASE_HEIGHT - 0.02:
      rise_time = step * dt
  return {
    "start_height": start_height,
    "peak_speed": max(speeds),
    "p99_speed": _percentile(speeds, 99),
    "peak_torque": max(torques),
    "final_height": heights[-1],
    "min_height_after_rise": min(heights[len(heights) // 2 :]),
    "rise_time_s": rise_time,
    "max_ramp_error": max(errors),
  }


def main() -> int:
  args = parse_args()
  learned = roll_out(args)
  scripted = scripted_reference(args.seconds)

  rows = [
    ("start base height [m]", "start_height"),
    ("peak joint speed [rad/s]", "peak_speed"),
    ("99th pct joint speed [rad/s]", "p99_speed"),
    ("peak joint torque [N.m]", "peak_torque"),
    ("time to standing [s]", "rise_time_s"),
    ("final base height [m]", "final_height"),
    ("lowest height after rise [m]", "min_height_after_rise"),
    ("max deviation from ramp [m]", "max_ramp_error"),
  ]
  print(f"\n{'':32} {'learned':>10} {'scripted':>10}")
  for label, key in rows:
    print(f"{label:32} {learned[key]:10.3f} {scripted[key]:10.3f}")
  print(
    f"\nsafety envelope: motor no-load speed {S.safety.motor_speed_limit} rad/s, "
    f"torque ceiling {S.safety.motor_effort_limit} N.m, "
    f"reward charges above {S.safety.soft_speed_limit} rad/s"
  )
  print(f"nominal stance height: {NOMINAL_BASE_HEIGHT} m")
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
