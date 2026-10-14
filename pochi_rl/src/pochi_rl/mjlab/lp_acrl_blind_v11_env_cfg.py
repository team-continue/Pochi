"""V11 replaces the moving pose proxy and gates stopped velocity damping."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg, SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from pochi_rl.mjlab.lp_acrl_blind_v10_env_cfg import (
  pochi_lp_acrl_blind_v10_env_cfg,
  pochi_lp_acrl_blind_v10_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_stride_center_rewards import (
  NOMINAL_FOOT_COM_X,
  moving_top_two_stride_center,
  standing_joint_velocity_bounded_all_contact,
)
from pochi_rl.robot import FOOT_SITES

# Exact Adam learning rate saved in the pinned V10/model400 checkpoint.
SOURCE_LEARNING_RATE = 7.593750000000004e-05


def stride_center_reward_cfg() -> RewardTermCfg:
  return RewardTermCfg(
    func=moving_top_two_stride_center,
    weight=-36.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES, preserve_order=True),
      "tau_s": 0.25,
      "nominal_offsets_m": NOMINAL_FOOT_COM_X,
    },
  )


def pochi_lp_acrl_blind_v11_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = pochi_lp_acrl_blind_v10_env_cfg(play=play)
  del cfg.rewards["moving_common_hip_pitch_deviation"]
  cfg.rewards["moving_fore_aft_stride_center"] = stride_center_reward_cfg()
  cfg.rewards[
    "standing_joint_velocity"
  ].func = standing_joint_velocity_bounded_all_contact
  return cfg


def pochi_lp_acrl_blind_v11_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = pochi_lp_acrl_blind_v10_ppo_runner_cfg()
  cfg.algorithm.learning_rate = SOURCE_LEARNING_RATE
  return cfg
