"""Prospective Blind-v4 reward configuration, without task registration.

This factory only replaces v3's contact-duty variance by qualified-swing
occupancy. It does not enable a contact backend correction. Binding a validated
backend and its checkpoint-aware runner is a separate, explicit pending step.
"""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import RewardTermCfg, SceneEntityCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg

from pochi_rl.mjlab.lp_acrl_blind_v3 import pochi_lp_acrl_blind_v3_ppo_runner_cfg
from pochi_rl.mjlab.lp_acrl_blind_v3_env_cfg import pochi_lp_acrl_blind_v3_env_cfg
from pochi_rl.mjlab.lp_acrl_qualified_swing_rewards import (
  moving_qualified_swing_occupancy,
)
from pochi_rl.robot import FOOT_SITES


def pochi_lp_acrl_blind_v4_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Preserve v3 except for the specified moving swing-occupancy objective."""
  cfg = pochi_lp_acrl_blind_v3_env_cfg(play=play)
  del cfg.rewards["moving_contact_duty_variance"]
  cfg.rewards["moving_qualified_swing_occupancy"] = RewardTermCfg(
    func=moving_qualified_swing_occupancy,
    weight=-8.0,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", site_names=FOOT_SITES, preserve_order=True
      ),
      "target": 0.20,
      "tau_s": 1.0,
      "min_air": 0.06,
      "min_lift": 0.025,
    },
  )
  return cfg


def pochi_lp_acrl_blind_v4_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Keep the complete v3 quiet-stop actor, optimizer and symmetry configuration."""
  return pochi_lp_acrl_blind_v3_ppo_runner_cfg()
