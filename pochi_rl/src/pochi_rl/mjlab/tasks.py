"""Register Pochi mjlab tasks."""

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from pochi_rl.mjlab.agents.rsl_rl_ppo_cfg import (
  pochi_flat_ppo_runner_cfg,
  pochi_standup_ppo_runner_cfg,
)
from pochi_rl.mjlab.deliberate_walk_env_cfg import (
  pochi_deliberate_walk_env_cfg,
  pochi_deliberate_walk_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_blind_env_cfg import pochi_lp_acrl_blind_env_cfg
from pochi_rl.mjlab.lp_acrl_blind_v1_env_cfg import pochi_lp_acrl_blind_v1_env_cfg
from pochi_rl.mjlab.lp_acrl_env_cfg import (
  pochi_lp_acrl_flat_env_cfg,
  pochi_lp_acrl_rough_env_cfg,
)
from pochi_rl.mjlab.lp_acrl_runner import (
  LPACRLOnPolicyRunner,
  pochi_lp_acrl_ppo_runner_cfg,
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

DELIBERATE_WALK_TASK_ID = "Pochi-Deliberate-Walk-v0"

if DELIBERATE_WALK_TASK_ID not in list_tasks():
  register_mjlab_task(
    task_id=DELIBERATE_WALK_TASK_ID,
    env_cfg=pochi_deliberate_walk_env_cfg(),
    play_env_cfg=pochi_deliberate_walk_env_cfg(play=True),
    rl_cfg=pochi_deliberate_walk_ppo_runner_cfg(),
    runner_cls=PochiOnPolicyRunner,
  )

# Separate tasks and runner: these never reuse the existing policy weights.
for _task_id, _factory, _rough in (
  ("Pochi-LP-ACRL-Flat-v0", pochi_lp_acrl_flat_env_cfg, False),
  ("Pochi-LP-ACRL-Rough-v0", pochi_lp_acrl_rough_env_cfg, True),
):
  if _task_id not in list_tasks():
    register_mjlab_task(
      task_id=_task_id,
      env_cfg=_factory(),
      play_env_cfg=_factory(play=True),
      rl_cfg=pochi_lp_acrl_ppo_runner_cfg(rough=_rough),
      runner_cls=LPACRLOnPolicyRunner,
    )

BLIND_LP_ACRL_TASK_ID = "Pochi-LP-ACRL-Blind-v0"
if BLIND_LP_ACRL_TASK_ID not in list_tasks():
  register_mjlab_task(
    task_id=BLIND_LP_ACRL_TASK_ID,
    env_cfg=pochi_lp_acrl_blind_env_cfg(),
    play_env_cfg=pochi_lp_acrl_blind_env_cfg(play=True),
    rl_cfg=pochi_lp_acrl_ppo_runner_cfg(rough=True, blind=True),
    runner_cls=LPACRLOnPolicyRunner,
  )

BLIND_LP_ACRL_V1_TASK_ID = "Pochi-LP-ACRL-Blind-v1"
if BLIND_LP_ACRL_V1_TASK_ID not in list_tasks():
  register_mjlab_task(
    task_id=BLIND_LP_ACRL_V1_TASK_ID,
    env_cfg=pochi_lp_acrl_blind_v1_env_cfg(),
    play_env_cfg=pochi_lp_acrl_blind_v1_env_cfg(play=True),
    rl_cfg=pochi_lp_acrl_ppo_runner_cfg(rough=True, blind=True),
    runner_cls=LPACRLOnPolicyRunner,
  )
