"""Isolated low-noise v5 refinement; contact backend and task geometry stay v4."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from pochi_rl.mjlab.lp_acrl_blind_v4_env_cfg import (
  pochi_lp_acrl_blind_v4_env_cfg,
  pochi_lp_acrl_blind_v4_ppo_runner_cfg,
)


def pochi_lp_acrl_blind_v5_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  cfg = pochi_lp_acrl_blind_v4_env_cfg(play=play)
  cfg.observations["actor"].enable_corruption = False
  cfg.rewards["track_ang_vel_z_exp"].params["std"] = 0.15
  cfg.rewards["standing_posture"].weight = 1.5
  cfg.rewards["standing_posture"].params["std"] = 0.15
  cfg.rewards["standing_missing_contacts"].weight = -2.0
  return cfg


def pochi_lp_acrl_blind_v5_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  cfg = pochi_lp_acrl_blind_v4_ppo_runner_cfg()
  cfg.actor.distribution_cfg.update(
    init_std=0.03, learn_std=False, std_range=(0.03, 0.03)
  )
  cfg.algorithm.learning_rate = 1.0e-4
  cfg.algorithm.symmetry_cfg["use_data_augmentation"] = False
  return cfg
