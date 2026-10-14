"""Isolated duty-balance and slip adaptation of the unchanged blind v2 task."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.reward_manager import RewardTermCfg

from pochi_rl.mjlab.lp_acrl_blind_v2 import pochi_lp_acrl_blind_v2_env_cfg
from pochi_rl.mjlab.lp_acrl_blind_v3_rewards import moving_contact_duty_variance


def pochi_lp_acrl_blind_v3_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Change only the moving duty-variance cost and existing contact-slip weight."""
  cfg = pochi_lp_acrl_blind_v2_env_cfg(play=play)
  cfg.rewards["contact_slip"].weight = -8.0
  cfg.rewards["moving_contact_duty_variance"] = RewardTermCfg(
    func=moving_contact_duty_variance,
    weight=-8.0,
    params={"tau_s": 1.0},
  )
  return cfg
