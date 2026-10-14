"""Fresh v7 scratch policy: v6 environment with scheduled moving exploration."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from pochi_rl.mjlab.lp_acrl_blind_v6_env_cfg import (
  pochi_lp_acrl_blind_v6_env_cfg,
  pochi_lp_acrl_blind_v6_ppo_runner_cfg,
)


def pochi_lp_acrl_blind_v7_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return pochi_lp_acrl_blind_v6_env_cfg(play=play)


def pochi_lp_acrl_blind_v7_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = pochi_lp_acrl_blind_v6_ppo_runner_cfg()
  cfg.actor.distribution_cfg.update(
    init_std=0.35, std_range=(0.03, 0.35), learn_std=False
  )
  cfg.algorithm.class_name = "pochi_rl.mjlab.lp_acrl_annealed:LPACRLAnnealedPPO"
  cfg.algorithm.learning_rate = 1e-3
  cfg.max_iterations = 1500
  return cfg
