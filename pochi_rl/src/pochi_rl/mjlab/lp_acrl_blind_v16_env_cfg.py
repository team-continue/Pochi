"""V16 restores original rough600 tasks without restoring touchdown bonus."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from pochi_rl.mjlab.lp_acrl_blind_v12_env_cfg import pochi_lp_acrl_blind_v12_env_cfg
from pochi_rl.mjlab.lp_acrl_blind_v15_env_cfg import (
  pochi_lp_acrl_blind_v15_ppo_runner_cfg,
)

SOURCE_LEARNING_RATE = 7.593750000000004e-05


def pochi_lp_acrl_blind_v16_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = pochi_lp_acrl_blind_v12_env_cfg(play=play)
  del cfg.rewards["balanced_touchdown"]
  return cfg


def pochi_lp_acrl_blind_v16_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = pochi_lp_acrl_blind_v15_ppo_runner_cfg()
  cfg.algorithm.learning_rate = SOURCE_LEARNING_RATE
  return cfg
