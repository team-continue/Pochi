"""Explicit local registration for the corrected-contact Blind-v4 experiment.

Importing or registering the task does not install a contact backend. The
launcher must install it before creating an environment; the dedicated runner
checks and records that backend identity at its checkpoint boundaries.
"""

from __future__ import annotations

from mjlab.tasks.registry import list_tasks, register_mjlab_task

from pochi_rl.mjlab.lp_acrl_blind_v4_env_cfg import (
  pochi_lp_acrl_blind_v4_env_cfg,
  pochi_lp_acrl_blind_v4_ppo_runner_cfg,
)

BLIND_V4_TASK_ID = "Pochi-LP-ACRL-Blind-v4"


def register_blind_v4() -> str:
  """Register only v4, with a runner that requires the explicit contact backend."""
  if BLIND_V4_TASK_ID not in list_tasks():
    from pochi_rl.mjlab.lp_acrl_contact_runner import LPACRLContactOnPolicyRunner

    register_mjlab_task(
      task_id=BLIND_V4_TASK_ID,
      env_cfg=pochi_lp_acrl_blind_v4_env_cfg(),
      play_env_cfg=pochi_lp_acrl_blind_v4_env_cfg(play=True),
      rl_cfg=pochi_lp_acrl_blind_v4_ppo_runner_cfg(),
      runner_cls=LPACRLContactOnPolicyRunner,
    )
  return BLIND_V4_TASK_ID
