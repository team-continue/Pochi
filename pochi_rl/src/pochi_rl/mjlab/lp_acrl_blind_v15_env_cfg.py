"""V15 removes only the per-touchdown bonus from the frozen V13 flat task."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from pochi_rl.mjlab.lp_acrl_blind_v13_env_cfg import (
  pochi_lp_acrl_blind_v13_env_cfg,
  pochi_lp_acrl_blind_v13_ppo_runner_cfg,
)

# Actual adaptive Adam LR saved in the pinned V13/model700 checkpoint.
SOURCE_LEARNING_RATE = 5.0625000000000024e-05


def pochi_lp_acrl_blind_v15_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = pochi_lp_acrl_blind_v13_env_cfg(play=play)
  del cfg.rewards["balanced_touchdown"]
  return cfg


def pochi_lp_acrl_blind_v15_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = pochi_lp_acrl_blind_v13_ppo_runner_cfg()
  cfg.algorithm.learning_rate = SOURCE_LEARNING_RATE
  return cfg
