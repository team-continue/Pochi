"""Explicit registration for an isolated quiet-stop exploration experiment.

Blind-v2 is not registered by importing the package or this module.
New launchers call register_blind_v2(). Environment behavior is exactly Blind-v1;
only the actor's stochastic exploration at an exact-zero command differs.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg
from mjlab.tasks.registry import list_tasks, register_mjlab_task

from pochi_rl.mjlab.lp_acrl_blind_v1_env_cfg import pochi_lp_acrl_blind_v1_env_cfg
from pochi_rl.mjlab.lp_acrl_runner import pochi_lp_acrl_ppo_runner_cfg

BLIND_V2_TASK_ID = "Pochi-LP-ACRL-Blind-v2"
QUIET_STOP_ACTOR = "pochi_rl.mjlab.lp_acrl_quiet_stop:QuietStopMLPModel"


def pochi_lp_acrl_blind_v2_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Keep the complete v1 environment, including its 20/10/70 episode mixture."""
  return pochi_lp_acrl_blind_v1_env_cfg(play=play)


def pochi_lp_acrl_blind_v2_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Keep PPO, symmetry and moving exploration unchanged; select the new actor."""
  cfg = pochi_lp_acrl_ppo_runner_cfg(rough=True, blind=True)
  cfg.actor.class_name = QUIET_STOP_ACTOR
  return cfg


def register_blind_v2() -> str:
  """Idempotently register only v2; existing task registrations remain intact."""
  if BLIND_V2_TASK_ID not in list_tasks():
    from pochi_rl.mjlab.lp_acrl_quiet_stop import LPACRLQuietStopOnPolicyRunner

    register_mjlab_task(
      task_id=BLIND_V2_TASK_ID,
      env_cfg=pochi_lp_acrl_blind_v2_env_cfg(),
      play_env_cfg=pochi_lp_acrl_blind_v2_env_cfg(play=True),
      rl_cfg=pochi_lp_acrl_blind_v2_ppo_runner_cfg(),
      runner_cls=LPACRLQuietStopOnPolicyRunner,
    )
  return BLIND_V2_TASK_ID
