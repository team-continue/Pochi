"""Blind-v8 adds one moving common-hip-pitch penalty to the frozen v5 task."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg, SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from pochi_rl.mjlab.lp_acrl_blind_v5_env_cfg import (
  pochi_lp_acrl_blind_v5_env_cfg,
  pochi_lp_acrl_blind_v5_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_common_hip_rewards import (
  HIP_PITCH_NAMES,
  moving_common_hip_pitch_deviation,
)


def common_hip_reward_cfg() -> RewardTermCfg:
  return RewardTermCfg(
    func=moving_common_hip_pitch_deviation,
    weight=-4.0,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", joint_names=HIP_PITCH_NAMES, preserve_order=True
      ),
    },
  )


def pochi_lp_acrl_blind_v8_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = pochi_lp_acrl_blind_v5_env_cfg(play=play)
  cfg.rewards["moving_common_hip_pitch_deviation"] = common_hip_reward_cfg()
  return cfg


def pochi_lp_acrl_blind_v8_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return pochi_lp_acrl_blind_v5_ppo_runner_cfg()
