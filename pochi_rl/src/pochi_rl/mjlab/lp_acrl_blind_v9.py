"""Explicit-only v9 registration; fresh state comes from its scratch launcher."""

from __future__ import annotations

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from pochi_rl.mjlab.lp_acrl_blind_v9_env_cfg import (
  pochi_lp_acrl_blind_v9_env_cfg,
  pochi_lp_acrl_blind_v9_ppo_runner_cfg,
)

BLIND_V9_TASK_ID = "Pochi-LP-ACRL-Blind-v9"


def register_blind_v9() -> str:
  if BLIND_V9_TASK_ID not in list_tasks():
    from pochi_rl.mjlab.lp_acrl_annealed import LPACRLAnnealedOnPolicyRunner

    register_mjlab_task(
      task_id=BLIND_V9_TASK_ID,
      env_cfg=pochi_lp_acrl_blind_v9_env_cfg(),
      play_env_cfg=pochi_lp_acrl_blind_v9_env_cfg(play=True),
      rl_cfg=pochi_lp_acrl_blind_v9_ppo_runner_cfg(),
      runner_cls=LPACRLAnnealedOnPolicyRunner,
    )
  return BLIND_V9_TASK_ID
