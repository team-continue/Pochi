"""Explicit independent registration of the flat omni phase gait."""

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from pochi_rl.mjlab.lp_acrl_blind_v19_env_cfg import (
  pochi_lp_acrl_blind_v19_env_cfg,
  pochi_lp_acrl_blind_v19_ppo_runner_cfg,
)

TASK_ID = "Pochi-LP-ACRL-Blind-v19"


def register_blind_v19():
  if TASK_ID not in list_tasks():
    from pochi_rl.mjlab.lp_acrl_speed_runner import SpeedPhaseGaitRunner

    register_mjlab_task(
      task_id=TASK_ID,
      env_cfg=pochi_lp_acrl_blind_v19_env_cfg(),
      play_env_cfg=pochi_lp_acrl_blind_v19_env_cfg(play=True),
      rl_cfg=pochi_lp_acrl_blind_v19_ppo_runner_cfg(),
      runner_cls=SpeedPhaseGaitRunner,
    )
  return TASK_ID
