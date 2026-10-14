"""Flat omni-directional extension of the accepted V17 phase/lift gait."""

from mjlab.managers import RewardTermCfg

from pochi_rl.mjlab.lp_acrl_blind_v17_env_cfg import (
  pochi_lp_acrl_blind_v17_env_cfg,
  pochi_lp_acrl_blind_v17_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_omni_commands import (
  MAX_COMMAND,
  OmniPhaseVelocityCommandCfg,
  planar_tracking_huber_error,
)


def pochi_lp_acrl_blind_v18_env_cfg(play=False):
  cfg = pochi_lp_acrl_blind_v17_env_cfg(play=play)
  old = cfg.commands["base_velocity"]
  command = OmniPhaseVelocityCommandCfg(**vars(old))
  command.standing_episode_fraction = 0.2
  command.transition_episode_fraction = 0.2
  command.transition_straight_probability = 0.0
  command.zero_lateral_probability = command.zero_yaw_probability = 0.0
  command.ranges.lin_vel_x = (-MAX_COMMAND[0], MAX_COMMAND[0])
  command.ranges.lin_vel_y = (-MAX_COMMAND[1], MAX_COMMAND[1])
  command.ranges.ang_vel_z = (-MAX_COMMAND[2], MAX_COMMAND[2])
  cfg.commands["base_velocity"] = command
  # Infinite plane: fast20s rollouts can legitimately move past the old11m edge.
  cfg.terminations.pop("task_terrain_edge")
  cfg.rewards["planar_tracking_huber"] = RewardTermCfg(
    func=planar_tracking_huber_error, weight=-12.0, params={"delta": 0.25}
  )
  return cfg


def pochi_lp_acrl_blind_v18_ppo_runner_cfg():
  cfg = pochi_lp_acrl_blind_v17_ppo_runner_cfg()
  cfg.algorithm.learning_rate = 7.593750000000004e-05
  return cfg
