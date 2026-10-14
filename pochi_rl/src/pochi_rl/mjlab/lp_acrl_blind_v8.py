"""Local-only v8 registration; no backend or global task mutation on import."""

from __future__ import annotations

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from pochi_rl.mjlab.lp_acrl_blind_v8_env_cfg import (
  pochi_lp_acrl_blind_v8_env_cfg,
  pochi_lp_acrl_blind_v8_ppo_runner_cfg,
)

BLIND_V8_TASK_ID = "Pochi-LP-ACRL-Blind-v8"


def register_blind_v8() -> str:
  if BLIND_V8_TASK_ID not in list_tasks():
    from pochi_rl.mjlab.lp_acrl_low_noise import LPACRLLowNoiseOnPolicyRunner

    register_mjlab_task(
      task_id=BLIND_V8_TASK_ID,
      env_cfg=pochi_lp_acrl_blind_v8_env_cfg(),
      play_env_cfg=pochi_lp_acrl_blind_v8_env_cfg(play=True),
      rl_cfg=pochi_lp_acrl_blind_v8_ppo_runner_cfg(),
      runner_cls=LPACRLLowNoiseOnPolicyRunner,
    )
  return BLIND_V8_TASK_ID
