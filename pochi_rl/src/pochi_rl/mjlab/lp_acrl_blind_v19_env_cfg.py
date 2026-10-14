"""Independent high-speed search; preserve V18 physics, gait and rewards."""

from pochi_rl.mjlab.lp_acrl_blind_v18_env_cfg import (
  pochi_lp_acrl_blind_v18_env_cfg,
  pochi_lp_acrl_blind_v18_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_speed_commands import (
  MAX_COMMAND,
  SpeedPhaseVelocityCommandCfg,
)


def pochi_lp_acrl_blind_v19_env_cfg(play=False):
  cfg = pochi_lp_acrl_blind_v18_env_cfg(play=play)
  command = SpeedPhaseVelocityCommandCfg(**vars(cfg.commands["base_velocity"]))
  command.ranges.lin_vel_x = (-MAX_COMMAND[0], MAX_COMMAND[0])
  command.ranges.lin_vel_y = (-MAX_COMMAND[1], MAX_COMMAND[1])
  command.ranges.ang_vel_z = (-MAX_COMMAND[2], MAX_COMMAND[2])
  cfg.commands["base_velocity"] = command
  return cfg


def pochi_lp_acrl_blind_v19_ppo_runner_cfg():
  return pochi_lp_acrl_blind_v18_ppo_runner_cfg()
