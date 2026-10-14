"""Fresh v9 engineering run: v8 rewards with lower slip cost and v7 PPO."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from pochi_rl.mjlab.lp_acrl_blind_v7_env_cfg import (
  pochi_lp_acrl_blind_v7_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_blind_v8_env_cfg import pochi_lp_acrl_blind_v8_env_cfg


def pochi_lp_acrl_blind_v9_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = pochi_lp_acrl_blind_v8_env_cfg(play=play)
  cfg.rewards["contact_slip"].weight = -0.2
  return cfg


def pochi_lp_acrl_blind_v9_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  return pochi_lp_acrl_blind_v7_ppo_runner_cfg()
