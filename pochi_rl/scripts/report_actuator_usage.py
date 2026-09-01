"""Report how hard a trained policy works the RS02 actuators.

The sim clamps torque to the RS02 torque-speed curve, so a policy can only ever
produce a *feasible* gait.  It can still produce an *unsustainable* one: the
motor's 7 N.m continuous rating is a thermal limit, not a mechanical one, and
nothing in the env stops a gait from sitting above it forever.  This script
measures the duty cycle so that stays a deliberate decision rather than a
surprise on hardware.

  uv run python scripts/report_actuator_usage.py --checkpoint <model.pt>
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from pochi_rl.robot import (
  JOINT_NAMES,
  RS02_NO_LOAD_SPEED_RAD_S,
  RS02_PEAK_TORQUE_NM,
  RS02_RATED_TORQUE_NM,
)

TASK_ID = "Pochi-Velocity-Flat-v0"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--num-envs", type=int, default=64)
  parser.add_argument("--steps", type=int, default=1000)
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

  runner_cls = load_runner_cls(TASK_ID) or MjlabOnPolicyRunner
  runner = runner_cls(wrapped, asdict(agent_cfg), device=args.device)
  runner.load(
    str(args.checkpoint),
    load_cfg={"actor": True},
    strict=True,
    map_location=args.device,
  )
  policy = runner.get_inference_policy(device=args.device)

  sq_sum = torch.zeros(12, device=args.device)
  peak_torque = torch.zeros(12, device=args.device)
  peak_speed = torch.zeros(12, device=args.device)
  at_limit = torch.zeros(12, device=args.device)
  samples = 0

  obs = wrapped.get_observations()
  with torch.inference_mode():
    for _ in range(args.steps):
      obs, _, _, _ = wrapped.step(policy(obs))

      torque = robot.data.actuator_force
      vel = robot.data.joint_vel
      # Same signed envelope the actuator enforces; a sample sitting on it means
      # the policy asked for more torque than the motor could give.
      ratio = (
        vel.clamp(-2.0 * RS02_NO_LOAD_SPEED_RAD_S, 2.0 * RS02_NO_LOAD_SPEED_RAD_S)
        / RS02_NO_LOAD_SPEED_RAD_S
      )
      upper = (RS02_PEAK_TORQUE_NM * (1.0 - ratio)).clamp(max=RS02_PEAK_TORQUE_NM)
      lower = (RS02_PEAK_TORQUE_NM * (-1.0 - ratio)).clamp(min=-RS02_PEAK_TORQUE_NM)

      sq_sum += torque.square().sum(dim=0)
      peak_torque = torch.maximum(peak_torque, torque.abs().amax(dim=0))
      peak_speed = torch.maximum(peak_speed, vel.abs().amax(dim=0))
      at_limit += ((torque >= upper - 1e-2) | (torque <= lower + 1e-2)).sum(dim=0)
      samples += torque.shape[0]

  rms = (sq_sum / samples).sqrt().cpu().numpy()
  peak_torque = peak_torque.cpu().numpy()
  peak_speed = peak_speed.cpu().numpy()
  clamped = (at_limit / samples).cpu().numpy()
  env.close()

  print(f"\n{args.checkpoint}")
  print(f"{samples} samples over {args.steps} steps x {args.num_envs} envs\n")
  print(
    f"{'joint':<16}{'RMS N.m':>9}{'peak N.m':>10}{'peak rad/s':>12}{'on clamp':>10}"
  )
  print("-" * 57)
  for i, name in enumerate(JOINT_NAMES):
    print(
      f"{name:<16}{rms[i]:>9.2f}{peak_torque[i]:>10.2f}"
      f"{peak_speed[i]:>12.2f}{clamped[i]:>9.1%}"
    )
  print("-" * 57)
  print(
    f"{'worst':<16}{rms.max():>9.2f}{peak_torque.max():>10.2f}"
    f"{peak_speed.max():>12.2f}{clamped.max():>9.1%}"
  )
  print(
    f"\nlimits: peak {RS02_PEAK_TORQUE_NM} N.m, continuous {RS02_RATED_TORQUE_NM} N.m,"
    f" no-load {RS02_NO_LOAD_SPEED_RAD_S} rad/s"
  )

  hot = [JOINT_NAMES[i] for i in np.flatnonzero(rms > RS02_RATED_TORQUE_NM)]
  if hot:
    print(
      f"\nWARNING: {', '.join(hot)} exceed the {RS02_RATED_TORQUE_NM} N.m continuous\n"
      "rating. The gait is mechanically valid but would overheat the motors."
    )
  else:
    print(
      f"\nOK: every joint stays under the {RS02_RATED_TORQUE_NM} N.m continuous rating."
    )


if __name__ == "__main__":
  main()
