"""Explicit v5 registration; backend installation remains a launcher operation."""

from __future__ import annotations

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from pochi_rl.mjlab.lp_acrl_blind_v5_env_cfg import (
  pochi_lp_acrl_blind_v5_env_cfg,
  pochi_lp_acrl_blind_v5_ppo_runner_cfg,
)

BLIND_V5_TASK_ID = "Pochi-LP-ACRL-Blind-v5"


def register_blind_v5() -> str:
  if BLIND_V5_TASK_ID not in list_tasks():
    from pochi_rl.mjlab.lp_acrl_low_noise import LPACRLLowNoiseOnPolicyRunner

    register_mjlab_task(
      task_id=BLIND_V5_TASK_ID,
      env_cfg=pochi_lp_acrl_blind_v5_env_cfg(),
      play_env_cfg=pochi_lp_acrl_blind_v5_env_cfg(play=True),
      rl_cfg=pochi_lp_acrl_blind_v5_ppo_runner_cfg(),
      runner_cls=LPACRLLowNoiseOnPolicyRunner,
    )
  return BLIND_V5_TASK_ID
