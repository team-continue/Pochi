"""V13 focuses the frozen V12 controller and rewards on signed flat walking."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from pochi_rl.mjlab.lp_acrl_blind_v12_env_cfg import (
  SOURCE_LEARNING_RATE,
  pochi_lp_acrl_blind_v12_env_cfg,
  pochi_lp_acrl_blind_v12_ppo_runner_cfg,
)


def pochi_lp_acrl_blind_v13_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Five signed forward bins, exact zero lateral/yaw, native infinite plane.

  Deriving from V12 retains its 400 Hz physics, 48D observations and 23 rewards.
  The existing terrain-edge timeout is retained at 11 m from each reset origin.
  A native terrain plane leaves scene.spec_fn available to the frozen evaluator.
  """
  cfg = pochi_lp_acrl_blind_v12_env_cfg(play=play)
  command = cfg.commands["base_velocity"]
  command.rough = False
  command.ranges.lin_vel_y = (0.0, 0.0)
  command.ranges.ang_vel_z = (0.0, 0.0)
  assert cfg.scene.terrain is not None
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  assert cfg.scene.spec_fn is None
  return cfg


def pochi_lp_acrl_blind_v13_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = pochi_lp_acrl_blind_v12_ppo_runner_cfg()
  assert cfg.algorithm.learning_rate == SOURCE_LEARNING_RATE
  return cfg
