"""Explicit registration for Blind-v3; importing does not register any task.

Only two reward entries differ from Blind-v2. Its quiet-stop stochastic actor,
PPO, symmetry, command sampler and physics are retained without modification.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import list_tasks, register_mjlab_task

from pochi_rl.mjlab.lp_acrl_blind_v2 import pochi_lp_acrl_blind_v2_ppo_runner_cfg
from pochi_rl.mjlab.lp_acrl_blind_v3_env_cfg import pochi_lp_acrl_blind_v3_env_cfg

BLIND_V3_TASK_ID = "Pochi-LP-ACRL-Blind-v3"


def pochi_lp_acrl_blind_v3_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Preserve the complete v2 PPO and quiet-stop exploration contract."""
  return pochi_lp_acrl_blind_v2_ppo_runner_cfg()


def register_blind_v3() -> str:
  """Register only v3, idempotently, preserving all existing task entries."""
  if BLIND_V3_TASK_ID not in list_tasks():
    from pochi_rl.mjlab.lp_acrl_quiet_stop import LPACRLQuietStopOnPolicyRunner

    register_mjlab_task(
      task_id=BLIND_V3_TASK_ID,
      env_cfg=pochi_lp_acrl_blind_v3_env_cfg(),
      play_env_cfg=pochi_lp_acrl_blind_v3_env_cfg(play=True),
      rl_cfg=pochi_lp_acrl_blind_v3_ppo_runner_cfg(),
      runner_cls=LPACRLQuietStopOnPolicyRunner,
    )
  return BLIND_V3_TASK_ID
