"""V12 adds a nonsaturating yaw-error cost to the frozen V11 environment."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from pochi_rl.mjlab.lp_acrl_blind_v11_env_cfg import (
  pochi_lp_acrl_blind_v11_env_cfg,
  pochi_lp_acrl_blind_v11_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_yaw_stability_rewards import yaw_tracking_huber_error

# Exact Adam learning rate saved in the pinned V11/model300 checkpoint.
SOURCE_LEARNING_RATE = 2.2500000000000008e-05


def yaw_stability_reward_cfg() -> RewardTermCfg:
  return RewardTermCfg(
    func=yaw_tracking_huber_error, weight=-8.0, params={"delta": 0.3}
  )


def pochi_lp_acrl_blind_v12_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = pochi_lp_acrl_blind_v11_env_cfg(play=play)
  cfg.rewards["yaw_tracking_huber"] = yaw_stability_reward_cfg()
  return cfg


def pochi_lp_acrl_blind_v12_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = pochi_lp_acrl_blind_v11_ppo_runner_cfg()
  cfg.algorithm.learning_rate = SOURCE_LEARNING_RATE
  return cfg
