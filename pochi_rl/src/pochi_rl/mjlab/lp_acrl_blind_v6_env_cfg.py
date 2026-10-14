"""Blind-v6 adds one moving default-pose penalty to the frozen v5 task."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg, SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from pochi_rl.mjlab.lp_acrl_blind_v5_env_cfg import (
  pochi_lp_acrl_blind_v5_env_cfg,
  pochi_lp_acrl_blind_v5_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_moving_posture_rewards import moving_default_joint_deviation
from pochi_rl.robot import JOINT_NAMES


def moving_posture_reward_cfg() -> RewardTermCfg:
  return RewardTermCfg(
    func=moving_default_joint_deviation,
    weight=-4.0,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=JOINT_NAMES, preserve_order=True
      ),
    },
  )


def pochi_lp_acrl_blind_v6_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = pochi_lp_acrl_blind_v5_env_cfg(play=play)
  cfg.rewards["moving_default_joint_deviation"] = moving_posture_reward_cfg()
  return cfg


def pochi_lp_acrl_blind_v6_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return pochi_lp_acrl_blind_v5_ppo_runner_cfg()
