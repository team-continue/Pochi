"""Local V12 registration; its explicit model-only initializer starts training."""

from __future__ import annotations

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from pochi_rl.mjlab.lp_acrl_blind_v12_env_cfg import (
  pochi_lp_acrl_blind_v12_env_cfg,
  pochi_lp_acrl_blind_v12_ppo_runner_cfg,
)

BLIND_V12_TASK_ID = "Pochi-LP-ACRL-Blind-v12"


def register_blind_v12() -> str:
  if BLIND_V12_TASK_ID not in list_tasks():
    from pochi_rl.mjlab.lp_acrl_yaw_stability_refinement import (
      LPACRLYawStabilityOnPolicyRunner,
    )

    register_mjlab_task(
      task_id=BLIND_V12_TASK_ID,
      env_cfg=pochi_lp_acrl_blind_v12_env_cfg(),
      play_env_cfg=pochi_lp_acrl_blind_v12_env_cfg(play=True),
      rl_cfg=pochi_lp_acrl_blind_v12_ppo_runner_cfg(),
      runner_cls=LPACRLYawStabilityOnPolicyRunner,
    )
  return BLIND_V12_TASK_ID
