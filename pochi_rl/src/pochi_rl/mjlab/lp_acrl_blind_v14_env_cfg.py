"""V14 adds the frozen raw-contact duty balance term to the complete V13 task."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from pochi_rl.mjlab.lp_acrl_blind_v3_rewards import moving_contact_duty_variance
from pochi_rl.mjlab.lp_acrl_blind_v13_env_cfg import (
  pochi_lp_acrl_blind_v13_env_cfg,
  pochi_lp_acrl_blind_v13_ppo_runner_cfg,
)

# Actual adaptive Adam LR saved in the pinned V13/model200 checkpoint.
SOURCE_LEARNING_RATE = 7.593750000000004e-05


def duty_balance_reward_cfg() -> RewardTermCfg:
  """Reuse the exact V3 term, including its one-second contact-duty EMA."""
  return RewardTermCfg(
    func=moving_contact_duty_variance,
    weight=-8.0,
    params={"tau_s": 1.0},
  )


def pochi_lp_acrl_blind_v14_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = pochi_lp_acrl_blind_v13_env_cfg(play=play)
  cfg.rewards["moving_contact_duty_variance"] = duty_balance_reward_cfg()
  return cfg


def pochi_lp_acrl_blind_v14_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = pochi_lp_acrl_blind_v13_ppo_runner_cfg()
  cfg.algorithm.learning_rate = SOURCE_LEARNING_RATE
  return cfg
