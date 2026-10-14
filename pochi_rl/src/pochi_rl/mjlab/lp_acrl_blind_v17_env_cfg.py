"""Independent flat 1.5 Hz /5cm lift experiment; no touchdown bonus."""

from mjlab.managers import ObservationTermCfg, RewardTermCfg, SceneEntityCfg

from pochi_rl.mjlab.lp_acrl_blind_v15_env_cfg import (
  pochi_lp_acrl_blind_v15_env_cfg,
  pochi_lp_acrl_blind_v15_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_phase_gait import (
  PhaseVelocityCommandCfg,
  gait_phase,
  phase_foot_contact_error,
  phase_foot_lift_error,
)
from pochi_rl.robot import FOOT_SITES


def pochi_lp_acrl_blind_v17_env_cfg(play=False):
  cfg = pochi_lp_acrl_blind_v15_env_cfg(play=play)
  old = cfg.commands["base_velocity"]
  cfg.commands["base_velocity"] = PhaseVelocityCommandCfg(**vars(old))
  for group in cfg.observations.values():
    group.terms["gait_phase"] = ObservationTermCfg(func=gait_phase)
  cfg.rewards["moving_qualified_swing_occupancy"].params["min_lift"] = 0.04
  cfg.rewards["phase_foot_lift_error"] = RewardTermCfg(
    func=phase_foot_lift_error,
    weight=-4.0,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES, preserve_order=True)
    },
  )
  cfg.rewards["phase_foot_contact_error"] = RewardTermCfg(
    func=phase_foot_contact_error, weight=-2.0
  )
  return cfg


def pochi_lp_acrl_blind_v17_ppo_runner_cfg():
  cfg = pochi_lp_acrl_blind_v15_ppo_runner_cfg()
  cfg.actor.class_name = "pochi_rl.mjlab.lp_acrl_phase_gait:PhaseQuietStopMLP"
  cfg.algorithm.symmetry_cfg["data_augmentation_func"] = (
    "pochi_rl.mjlab.lp_acrl_phase_gait:augment_phase_symmetry"
  )
  cfg.algorithm.learning_rate = 7.593750000000004e-05
  return cfg
