"""Local V10 registration; explicit full-state migration initializes training."""

from __future__ import annotations

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from pochi_rl.mjlab.lp_acrl_blind_v10_env_cfg import (
  pochi_lp_acrl_blind_v10_env_cfg,
  pochi_lp_acrl_blind_v10_ppo_runner_cfg,
)

BLIND_V10_TASK_ID = "Pochi-LP-ACRL-Blind-v10"


def register_blind_v10() -> str:
  if BLIND_V10_TASK_ID not in list_tasks():
    from pochi_rl.mjlab.lp_acrl_wider_exploration import (
      LPACRLWiderExplorationOnPolicyRunner,
    )

    register_mjlab_task(
      task_id=BLIND_V10_TASK_ID,
      env_cfg=pochi_lp_acrl_blind_v10_env_cfg(),
      play_env_cfg=pochi_lp_acrl_blind_v10_env_cfg(play=True),
      rl_cfg=pochi_lp_acrl_blind_v10_ppo_runner_cfg(),
      runner_cls=LPACRLWiderExplorationOnPolicyRunner,
    )
  return BLIND_V10_TASK_ID
