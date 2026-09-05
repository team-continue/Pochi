"""A low-speed crawl: one foot swings, three support the body."""

from dataclasses import fields

from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.tasks.velocity import mdp

from pochi_rl.mjlab import deliberate_walk as gait
from pochi_rl.mjlab import rewards
from pochi_rl.mjlab.agents.rsl_rl_ppo_cfg import pochi_flat_ppo_runner_cfg
from pochi_rl.mjlab.velocity_flat_env_cfg import pochi_velocity_flat_env_cfg


def pochi_deliberate_walk_env_cfg(play: bool = False):
  cfg = pochi_velocity_flat_env_cfg(play=play)
  cfg.curriculum = {}
  command = cfg.commands["base_velocity"]
  command.ranges.lin_vel_x = (0.06, 0.14) if not play else (0.1, 0.1)
  command.ranges.lin_vel_y = (-0.025, 0.025) if not play else (0.0, 0.0)
  command.ranges.ang_vel_z = (-0.15, 0.15) if not play else (0.0, 0.0)
  command.heading_command = False
  command.ranges.heading = None
  command.rel_heading_envs = 0.0
  command.rel_standing_envs = 0.15 if not play else 0.0
  command.resampling_time_range = (8.4, 14.0)
  cfg.commands["base_velocity"] = gait.DeliberateVelocityCommandCfg(
    **{field.name: getattr(command, field.name) for field in fields(command)}
  )
  cfg.events.pop("push_robot", None)
  cfg.events["reset_root"].params["velocity_range"] = {}
  for group in cfg.observations.values():
    group.terms["gait_phase"] = ObservationTermCfg(func=gait.phase_observation)
  cfg.rewards["track_lin_vel_xy_exp"].weight = 3.0
  cfg.rewards["track_lin_vel_xy_exp"].params["std"] = 0.12
  cfg.rewards["track_ang_vel_z_exp"].params["std"] = 0.25
  cfg.rewards["joint_vel_l2"].weight = -0.008
  cfg.rewards["action_rate_l2"].weight = -0.04
  cfg.rewards["ang_vel_xy_l2"].weight = -0.15
  cfg.rewards["stand_still"].params["command_threshold"] = gait.SPEC.command_threshold
  cfg.rewards.pop("feet_air_time")
  cfg.rewards.update(
    contact_schedule=RewardTermCfg(
      func=gait.contact_schedule,
      weight=-1.5,
      params={"sensor_name": "feet_ground_contact"},
    ),
    insufficient_support=RewardTermCfg(
      func=gait.insufficient_support,
      weight=-2.0,
      params={"sensor_name": "feet_ground_contact"},
    ),
    foot_height=RewardTermCfg(
      func=gait.foot_height_error,
      weight=-12.0,
      params={"asset_cfg": gait.FEET},
    ),
    feet_slip=RewardTermCfg(
      func=mdp.feet_slip,
      weight=-2.0,
      params={
        "sensor_name": "feet_ground_contact",
        "command_name": "base_velocity",
        "command_threshold": 0.0,
        "asset_cfg": gait.FEET,
      },
    ),
    upright=RewardTermCfg(func=rewards.upright, weight=0.5, params={"std": 0.3}),
  )
  return cfg


def pochi_deliberate_walk_ppo_runner_cfg():
  cfg = pochi_flat_ppo_runner_cfg()
  cfg.experiment_name = "pochi_deliberate_walk"
  cfg.max_iterations = 1500
  cfg.algorithm.gamma = 0.995
  return cfg
