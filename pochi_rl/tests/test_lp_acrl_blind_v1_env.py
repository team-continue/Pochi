"""The reward refinement leaves observation, physics and curriculum unchanged."""

from pochi_rl.mjlab.lp_acrl_blind_env_cfg import pochi_lp_acrl_blind_env_cfg
from pochi_rl.mjlab.lp_acrl_blind_v1_env_cfg import pochi_lp_acrl_blind_v1_env_cfg


def test_refinement_only_changes_three_reward_terms():
  old = pochi_lp_acrl_blind_env_cfg()
  new = pochi_lp_acrl_blind_v1_env_cfg()
  assert old.rewards["standing_joint_velocity"].weight == -0.02
  assert new.rewards["standing_joint_velocity"].weight == -0.25
  assert "prolonged_foot_contact" in old.rewards
  assert "prolonged_foot_contact" not in new.rewards
  assert new.rewards["valid_foot_cycle_age"].weight == -0.6
  assert new.rewards["signed_translation_progress"].weight == 0.5
  old.rewards, new.rewards = {}, {}
  # EntityCfg creates a new equivalent lambda for each terrain configuration.
  assert (
    old.scene.terrain.spec_fn.__qualname__
    == new.scene.terrain.spec_fn.__qualname__
  )
  old.scene.terrain.spec_fn = new.scene.terrain.spec_fn
  assert old == new


def test_refinement_play_keeps_manual_zero_and_no_scan():
  cfg = pochi_lp_acrl_blind_v1_env_cfg(play=True)
  assert cfg.commands["base_velocity"].manual_control_on_play
  assert all(sensor.name != "terrain_scan" for sensor in cfg.scene.sensors)
  assert all("height_scan" not in group.terms for group in cfg.observations.values())
