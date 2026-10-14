"""Local V13 registration, with explicit model-only flat-focus initialization."""

from __future__ import annotations

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from pochi_rl.mjlab.lp_acrl_blind_v13_env_cfg import (
  pochi_lp_acrl_blind_v13_env_cfg,
  pochi_lp_acrl_blind_v13_ppo_runner_cfg,
)

BLIND_V13_TASK_ID = "Pochi-LP-ACRL-Blind-v13"


def register_blind_v13() -> str:
  if BLIND_V13_TASK_ID not in list_tasks():
    from pochi_rl.mjlab.lp_acrl_flat_focus_refinement import (
      LPACRLFlatFocusOnPolicyRunner,
    )

    register_mjlab_task(
      task_id=BLIND_V13_TASK_ID,
      env_cfg=pochi_lp_acrl_blind_v13_env_cfg(),
      play_env_cfg=pochi_lp_acrl_blind_v13_env_cfg(play=True),
      rl_cfg=pochi_lp_acrl_blind_v13_ppo_runner_cfg(),
      runner_cls=LPACRLFlatFocusOnPolicyRunner,
    )
  return BLIND_V13_TASK_ID
