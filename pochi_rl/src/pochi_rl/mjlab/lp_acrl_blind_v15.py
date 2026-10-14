"""Local V15 registration with an independent model-only refinement boundary."""

from __future__ import annotations

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from pochi_rl.mjlab.lp_acrl_blind_v15_env_cfg import (
  pochi_lp_acrl_blind_v15_env_cfg,
  pochi_lp_acrl_blind_v15_ppo_runner_cfg,
)

BLIND_V15_TASK_ID = "Pochi-LP-ACRL-Blind-v15"


def register_blind_v15() -> str:
  if BLIND_V15_TASK_ID not in list_tasks():
    from pochi_rl.mjlab.lp_acrl_no_touchdown_refinement import (
      LPACRLNoTouchdownOnPolicyRunner,
    )

    register_mjlab_task(
      task_id=BLIND_V15_TASK_ID,
      env_cfg=pochi_lp_acrl_blind_v15_env_cfg(),
      play_env_cfg=pochi_lp_acrl_blind_v15_env_cfg(play=True),
      rl_cfg=pochi_lp_acrl_blind_v15_ppo_runner_cfg(),
      runner_cls=LPACRLNoTouchdownOnPolicyRunner,
    )
  return BLIND_V15_TASK_ID
