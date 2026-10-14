"""V10 retains V8's full environment and widens only moving exploration."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from pochi_rl.mjlab.lp_acrl_blind_v8_env_cfg import (
  pochi_lp_acrl_blind_v8_env_cfg,
  pochi_lp_acrl_blind_v8_ppo_runner_cfg,
)


def pochi_lp_acrl_blind_v10_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  return pochi_lp_acrl_blind_v8_env_cfg(play=play)


def pochi_lp_acrl_blind_v10_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = pochi_lp_acrl_blind_v8_ppo_runner_cfg()
  cfg.actor.distribution_cfg.update(
    init_std=0.15, std_range=(0.15, 0.15), learn_std=False
  )
  return cfg
