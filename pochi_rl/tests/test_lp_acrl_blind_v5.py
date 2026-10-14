"""Only explicit noise, reward, initial-LR and augmentation changes in v5."""

from dataclasses import asdict

import pytest
from mjlab.tasks import registry

from pochi_rl.mjlab.lp_acrl_blind_v4_env_cfg import (
  pochi_lp_acrl_blind_v4_env_cfg,
  pochi_lp_acrl_blind_v4_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_blind_v5 import BLIND_V5_TASK_ID, register_blind_v5
from pochi_rl.mjlab.lp_acrl_blind_v5_env_cfg import (
  pochi_lp_acrl_blind_v5_env_cfg,
  pochi_lp_acrl_blind_v5_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_low_noise import LPACRLLowNoiseOnPolicyRunner


@pytest.mark.parametrize("play", [False, True])
def test_v5_environment_changes_exactly_actor_corruption_and_three_reward_entries(play):
  source = pochi_lp_acrl_blind_v4_env_cfg(play=play)
  target = pochi_lp_acrl_blind_v5_env_cfg(play=play)
  assert target.observations["actor"].enable_corruption is False
  assert target.observations["critic"].enable_corruption is False
  assert target.rewards["track_ang_vel_z_exp"].params["std"] == 0.15
  assert target.rewards["standing_posture"].weight == 1.5
  assert target.rewards["standing_posture"].params["std"] == 0.15
  assert target.rewards["standing_missing_contacts"].weight == -2
  assert (
    target.rewards["moving_qualified_swing_occupancy"]
    == source.rewards["moving_qualified_swing_occupancy"]
  )
  target.observations["actor"].enable_corruption = source.observations[
    "actor"
  ].enable_corruption
  for key in ("track_ang_vel_z_exp", "standing_posture", "standing_missing_contacts"):
    target.rewards[key] = source.rewards[key]
  assert (
    target.scene.terrain.spec_fn.__qualname__
    == source.scene.terrain.spec_fn.__qualname__
  )
  target.scene.terrain.spec_fn = source.scene.terrain.spec_fn
  assert target == source


def test_v5_ppo_changes_only_declared_std_initial_lr_and_data_augmentation():
  source = asdict(pochi_lp_acrl_blind_v4_ppo_runner_cfg())
  target = asdict(pochi_lp_acrl_blind_v5_ppo_runner_cfg())
  distribution = target["actor"]["distribution_cfg"]
  assert distribution["init_std"] == 0.03
  assert distribution["std_range"] == (0.03, 0.03)
  assert distribution["learn_std"] is False
  assert target["algorithm"]["learning_rate"] == 1e-4
  assert source["algorithm"]["learning_rate"] == 1e-3
  symmetry = target["algorithm"]["symmetry_cfg"]
  assert symmetry["use_data_augmentation"] is False
  assert symmetry["use_mirror_loss"] is True
  assert symmetry["mirror_loss_coeff"] == 1.0
  assert source["algorithm"]["symmetry_cfg"]["use_data_augmentation"] is True
  target["actor"]["distribution_cfg"] = source["actor"]["distribution_cfg"]
  target["algorithm"]["learning_rate"] = source["algorithm"]["learning_rate"]
  symmetry["use_data_augmentation"] = True
  assert target == source


def test_v5_registration_only_adds_v5_and_does_not_install_contact_backend(monkeypatch):
  from pochi_rl.mjlab import warp_contact_backend

  def forbidden(*args, **kwargs):
    raise AssertionError("Registration must not install a backend")

  monkeypatch.setattr(warp_contact_backend, "install_contact_backend", forbidden)
  current = dict(registry._REGISTRY)
  current.pop(BLIND_V5_TASK_ID, None)
  monkeypatch.setattr(registry, "_REGISTRY", current)
  before = dict(registry._REGISTRY)
  assert register_blind_v5() == BLIND_V5_TASK_ID
  registered = registry._REGISTRY[BLIND_V5_TASK_ID]
  assert register_blind_v5() == BLIND_V5_TASK_ID
  assert registry._REGISTRY[BLIND_V5_TASK_ID] is registered
  assert set(registry.list_tasks()) == set(before) | {BLIND_V5_TASK_ID}
  assert all(registry._REGISTRY[name] is value for name, value in before.items())
  assert registry.load_runner_cls(BLIND_V5_TASK_ID) is LPACRLLowNoiseOnPolicyRunner
