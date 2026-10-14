"""Local-only v6 registration; no backend or global task mutation on import."""

from __future__ import annotations

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from pochi_rl.mjlab.lp_acrl_blind_v6_env_cfg import (
  pochi_lp_acrl_blind_v6_env_cfg,
  pochi_lp_acrl_blind_v6_ppo_runner_cfg,
)

BLIND_V6_TASK_ID = "Pochi-LP-ACRL-Blind-v6"


def register_blind_v6() -> str:
  if BLIND_V6_TASK_ID not in list_tasks():
    from pochi_rl.mjlab.lp_acrl_low_noise import LPACRLLowNoiseOnPolicyRunner

    register_mjlab_task(
      task_id=BLIND_V6_TASK_ID,
      env_cfg=pochi_lp_acrl_blind_v6_env_cfg(),
      play_env_cfg=pochi_lp_acrl_blind_v6_env_cfg(play=True),
      rl_cfg=pochi_lp_acrl_blind_v6_ppo_runner_cfg(),
      runner_cls=LPACRLLowNoiseOnPolicyRunner,
    )
  return BLIND_V6_TASK_ID
