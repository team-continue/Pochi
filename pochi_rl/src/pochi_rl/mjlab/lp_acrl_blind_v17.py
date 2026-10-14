"""Explicit registration for the independent phase gait experiment."""

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from pochi_rl.mjlab.lp_acrl_blind_v17_env_cfg import (
  pochi_lp_acrl_blind_v17_env_cfg,
  pochi_lp_acrl_blind_v17_ppo_runner_cfg,
)

TASK_ID = "Pochi-LP-ACRL-Blind-v17"


def register_blind_v17():
  if TASK_ID not in list_tasks():
    from pochi_rl.mjlab.lp_acrl_phase_runner import PhaseGaitRunner

    register_mjlab_task(
      task_id=TASK_ID,
      env_cfg=pochi_lp_acrl_blind_v17_env_cfg(),
      play_env_cfg=pochi_lp_acrl_blind_v17_env_cfg(play=True),
      rl_cfg=pochi_lp_acrl_blind_v17_ppo_runner_cfg(),
      runner_cls=PhaseGaitRunner,
    )
  return TASK_ID
