"""Measure how fast the legs move, and whether the robot stands still.

Two things a reward curve will not tell you: whether the gait is deliberate or
frantic, and whether "stop" actually means stop.  This drives a checkpoint at a
fixed command and reports both.

  uv run python scripts/report_gait.py --checkpoint <model.pt>
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

TASK_ID = "Pochi-Velocity-Flat-v0"


def _rollout(env, robot, policy, command, steps: int):
  """Pin the velocity command, then measure joint motion and base drift.

  The command term rewrites ``vel_command_b`` every step (heading control,
  standing envs), so the command is pinned through its config and a reset
  rather than by writing the tensor.
  """
  term = env.unwrapped.command_manager.get_term("base_velocity")
  cfg = term.cfg
  vx, vy, wz = command
  cfg.ranges.lin_vel_x = (vx, vx)
  cfg.ranges.lin_vel_y = (vy, vy)
  cfg.ranges.ang_vel_z = (wz, wz)
  cfg.heading_command = False
  cfg.rel_standing_envs = 0.0
  cfg.rel_forward_envs = 0.0
  cfg.rel_world_envs = 0.0
  cfg.init_velocity_prob = 0.0

  speeds, achieved, start, drift = [], [], None, None
  settle = steps // 5
  # The reset touches buffers created under inference mode by a previous
  # rollout, so it has to run inside inference mode as well.
  with torch.inference_mode():
    env.unwrapped.reset()
    obs = env.get_observations()
    for step in range(steps):
      obs, _, _, _ = env.step(policy(obs))
      if step == settle:
        start = robot.data.root_link_pos_w.clone()
      if step >= settle:
        speeds.append(robot.data.joint_vel.abs().clone())
        achieved.append(robot.data.root_link_lin_vel_b[:, 0].clone())
        drift = (robot.data.root_link_pos_w - start).norm(dim=-1)
  actual = term.command.mean(dim=0)
  return torch.stack(speeds), drift, actual, torch.stack(achieved)


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--steps", type=int, default=500)
  parser.add_argument(
    "--device", default="cuda:0" if torch.cuda.is_available() else "cpu"
  )
  args = parser.parse_args()

  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends

  import pochi_rl  # noqa: F401

  configure_torch_backends()
  env_cfg = load_env_cfg(TASK_ID, play=True)
  env_cfg.scene.num_envs = args.num_envs
  agent_cfg = load_rl_cfg(TASK_ID)

  env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
  robot = env.scene["robot"]
  wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
  runner = (load_runner_cls(TASK_ID) or MjlabOnPolicyRunner)(
    wrapped, asdict(agent_cfg), device=args.device
  )
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(device=args.device)

  print(f"\n{args.checkpoint}\n")
  step_dt = env.step_dt

  walk = None
  for speed in (0.2, 0.6, 1.0):
    s, _, cmd, vx = _rollout(wrapped, robot, policy, (speed, 0.0, 0.0), args.steps)
    if speed == 0.6:
      walk = s
    print(f"=== walking, command {[round(float(v), 2) for v in cmd]} ===")
    print(
      f"  joint speed  mean {s.mean():6.2f}   p95 {s.quantile(0.95):6.2f}"
      f"   max {s.max():6.2f}  rad/s"
    )
    # Joint speed alone hides a robot that simply refuses to move, so check
    # that the commanded velocity is actually achieved.
    print(
      f"  achieved vx  mean {vx.mean():6.2f} m/s"
      f"   (command {speed:.1f}, error {abs(float(vx.mean()) - speed):.2f})"
    )
  assert walk is not None

  stand, drift, cmd_s, _ = _rollout(wrapped, robot, policy, (0.0, 0.0, 0.0), args.steps)
  held = (args.steps - args.steps // 5) * step_dt
  print(f"\n=== standing, command {[round(float(v), 2) for v in cmd_s]} ===")
  print(
    f"  joint speed  mean {stand.mean():6.2f}   p95 {stand.quantile(0.95):6.2f}"
    f"   max {stand.max():6.2f}  rad/s"
  )
  assert drift is not None
  print(
    f"  base drift over {held:.1f}s: mean {drift.mean() * 100:5.1f} cm"
    f"   max {drift.max() * 100:5.1f} cm"
  )
  ratio = float(stand.mean() / walk.mean().clamp(min=1e-6))
  print(f"\n  standing joint speed is {ratio:.1%} of walking")
  if ratio < 0.15:
    print("  -> stands still")
  else:
    print("  -> still marching in place")

  env.close()


if __name__ == "__main__":
  main()
