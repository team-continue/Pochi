"""Second blind reward version, preserving the first experiment unchanged."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pochi_rl.mjlab.lp_acrl_blind_env_cfg import pochi_lp_acrl_blind_env_cfg
from pochi_rl.robot.pochi_constants import FOOT_SITES


def pochi_lp_acrl_blind_v1_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Reward real foot cycles and commanded translation, with quiet standing."""
  from pochi_rl.mjlab import lp_acrl_blind_v1_rewards as refined

  cfg = pochi_lp_acrl_blind_env_cfg(play=play)
  cfg.rewards.pop("prolonged_foot_contact")
  cfg.rewards["standing_joint_velocity"] = RewardTermCfg(
    func=refined.standing_joint_velocity_bounded, weight=-0.25
  )
  cfg.rewards["valid_foot_cycle_age"] = RewardTermCfg(
    func=refined.valid_foot_cycle_age,
    weight=-0.6,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=FOOT_SITES, preserve_order=True
      )
    },
  )
  cfg.rewards["signed_translation_progress"] = RewardTermCfg(
    func=refined.signed_translation_progress, weight=0.5
  )
  return cfg
