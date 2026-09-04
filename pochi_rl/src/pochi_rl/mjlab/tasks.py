"""Register Pochi mjlab tasks."""

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from pochi_rl.mjlab.agents.rsl_rl_ppo_cfg import (
  pochi_flat_ppo_runner_cfg,
  pochi_standup_ppo_runner_cfg,
)
from pochi_rl.mjlab.runner import PochiOnPolicyRunner
from pochi_rl.mjlab.standup_env_cfg import pochi_standup_flat_env_cfg
from pochi_rl.mjlab.velocity_flat_env_cfg import pochi_velocity_flat_env_cfg

VELOCITY_TASK_ID = "Pochi-Velocity-Flat-v0"
STANDUP_TASK_ID = "Pochi-StandUp-Flat-v0"

if VELOCITY_TASK_ID not in list_tasks():
  register_mjlab_task(
    task_id=VELOCITY_TASK_ID,
    env_cfg=pochi_velocity_flat_env_cfg(),
    play_env_cfg=pochi_velocity_flat_env_cfg(play=True),
    rl_cfg=pochi_flat_ppo_runner_cfg(),
    runner_cls=PochiOnPolicyRunner,
  )

if STANDUP_TASK_ID not in list_tasks():
  register_mjlab_task(
    task_id=STANDUP_TASK_ID,
    env_cfg=pochi_standup_flat_env_cfg(),
    play_env_cfg=pochi_standup_flat_env_cfg(play=True),
    rl_cfg=pochi_standup_ppo_runner_cfg(),
    runner_cls=PochiOnPolicyRunner,
  )
