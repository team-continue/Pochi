"""Local V16 registration with an independent model-only refinement boundary."""

from __future__ import annotations

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from pochi_rl.mjlab.lp_acrl_blind_v16_env_cfg import (
  pochi_lp_acrl_blind_v16_env_cfg,
  pochi_lp_acrl_blind_v16_ppo_runner_cfg,
)

BLIND_V16_TASK_ID = "Pochi-LP-ACRL-Blind-v16"


def register_blind_v16() -> str:
  if BLIND_V16_TASK_ID not in list_tasks():
    from pochi_rl.mjlab.lp_acrl_rough_no_touchdown_refinement import (
      LPACRLRoughNoTouchdownOnPolicyRunner,
    )

    register_mjlab_task(
      task_id=BLIND_V16_TASK_ID,
      env_cfg=pochi_lp_acrl_blind_v16_env_cfg(),
      play_env_cfg=pochi_lp_acrl_blind_v16_env_cfg(play=True),
      rl_cfg=pochi_lp_acrl_blind_v16_ppo_runner_cfg(),
      runner_cls=LPACRLRoughNoTouchdownOnPolicyRunner,
    )
  return BLIND_V16_TASK_ID
