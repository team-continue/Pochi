"""Local V11 registration; its explicit model-only initializer starts training."""

from __future__ import annotations

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from pochi_rl.mjlab.lp_acrl_blind_v11_env_cfg import (
  pochi_lp_acrl_blind_v11_env_cfg,
  pochi_lp_acrl_blind_v11_ppo_runner_cfg,
)

BLIND_V11_TASK_ID = "Pochi-LP-ACRL-Blind-v11"


def register_blind_v11() -> str:
  if BLIND_V11_TASK_ID not in list_tasks():
    from pochi_rl.mjlab.lp_acrl_stride_center_refinement import (
      LPACRLStrideCenterOnPolicyRunner,
    )

    register_mjlab_task(
      task_id=BLIND_V11_TASK_ID,
      env_cfg=pochi_lp_acrl_blind_v11_env_cfg(),
      play_env_cfg=pochi_lp_acrl_blind_v11_env_cfg(play=True),
      rl_cfg=pochi_lp_acrl_blind_v11_ppo_runner_cfg(),
      runner_cls=LPACRLStrideCenterOnPolicyRunner,
    )
  return BLIND_V11_TASK_ID
