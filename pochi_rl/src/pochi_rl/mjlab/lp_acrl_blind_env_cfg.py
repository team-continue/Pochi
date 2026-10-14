"""Height-scan-free LP-ACRL with explicit standing and balanced locomotion.

This is a separate Pochi adaptation. Original LP-ACRL environments/checkpoints
keep their observation, reward and terrain contracts unchanged.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pochi_rl.mjlab.lp_acrl_env_cfg import pochi_lp_acrl_rough_env_cfg
from pochi_rl.robot.pochi_constants import FOOT_SITES


def pochi_lp_acrl_blind_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """600 terrain tasks, 48 proprioceptive observations, explicit stop practice."""
  from pochi_rl.mjlab import lp_acrl_balanced_rewards as balanced

  cfg = pochi_lp_acrl_rough_env_cfg(play=play)
  cfg.scene.sensors = tuple(
    sensor for sensor in cfg.scene.sensors if sensor.name != "terrain_scan"
  )
  for group in cfg.observations.values():
    group.terms.pop("height_scan", None)

  command = cfg.commands["base_velocity"]
  command.standing_episode_fraction = 0.20
  command.transition_episode_fraction = 0.10
  command.transition_interval_s = (4.0, 6.0)
  command.zero_yaw_probability = 0.75
  command.zero_lateral_probability = 0.50
  command.transition_straight_probability = 0.70
  command.transition_forward_range = (0.10, 0.40)
  command.manual_control_on_play = True

  # The previous 10cm heightfield grid could exceed Warp's per-pair50-contact
  # candidate limit for the torso. At20cm, its bounding-sphere candidate bound
  # falls from72 to32 triangles; heights and task families remain unchanged.
  generator = cfg.scene.terrain.terrain_generator
  assert generator is not None
  for family in generator.sub_terrains.values():
    terrain = family.terrain
    if hasattr(terrain, "horizontal_scale"):
      terrain.horizontal_scale = 0.20

  cfg.rewards["track_lin_vel_xy_exp"].params["std"] = 0.15
  cfg.rewards["track_ang_vel_z_exp"].params["std"] = 0.35
  # Negative rewards for normal short touchdowns favored avoiding footfalls.
  cfg.rewards.pop("feet_air_time")
  cfg.rewards["undesired_contacts"].weight = -1.0
  cfg.rewards["termination"] = RewardTermCfg(func=mdp.is_terminated, weight=-200.0)
  feet = SceneEntityCfg("robot", site_names=FOOT_SITES, preserve_order=True)
  cfg.rewards.update(
    {
      "standing_posture": RewardTermCfg(
        func=balanced.standing_posture, weight=0.5
      ),
      "standing_joint_velocity": RewardTermCfg(
        func=balanced.standing_joint_velocity, weight=-0.02
      ),
      "standing_missing_contacts": RewardTermCfg(
        func=balanced.standing_missing_contacts, weight=-1.0
      ),
      "contact_slip": RewardTermCfg(
        func=balanced.contact_slip, weight=-0.2, params={"asset_cfg": feet}
      ),
      "upright_error": RewardTermCfg(func=balanced.upright_error, weight=-0.3),
      "support_height_error": RewardTermCfg(
        func=balanced.support_height_error, weight=-1.0, params={"asset_cfg": feet}
      ),
      "balanced_touchdown": RewardTermCfg(
        func=balanced.balanced_touchdown, weight=2.0, params={"asset_cfg": feet}
      ),
      "prolonged_foot_contact": RewardTermCfg(
        func=balanced.prolonged_foot_contact, weight=-0.6
      ),
    }
  )
  return cfg
