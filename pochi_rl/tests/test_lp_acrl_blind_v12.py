"""V12 adds one yaw cost and preserves V11's existing environment contracts."""

from dataclasses import asdict

import pytest
from mjlab.tasks import registry

from pochi_rl.mjlab.lp_acrl_blind_v11_env_cfg import (
  pochi_lp_acrl_blind_v11_env_cfg,
  pochi_lp_acrl_blind_v11_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_blind_v12 import BLIND_V12_TASK_ID, register_blind_v12
from pochi_rl.mjlab.lp_acrl_blind_v12_env_cfg import (
  SOURCE_LEARNING_RATE,
  pochi_lp_acrl_blind_v12_env_cfg,
  pochi_lp_acrl_blind_v12_ppo_runner_cfg,
  yaw_stability_reward_cfg,
)
from pochi_rl.mjlab.lp_acrl_yaw_stability_rewards import yaw_tracking_huber_error


@pytest.mark.parametrize("play", [False, True])
def test_one_addition_preserves_all_existing_rewards_and_nonreward_settings(play):
  source = pochi_lp_acrl_blind_v11_env_cfg(play=play)
  target = pochi_lp_acrl_blind_v12_env_cfg(play=play)
  assert len(target.rewards) == 23
  assert target.rewards.pop("yaw_tracking_huber") == yaw_stability_reward_cfg()
  assert target.rewards == source.rewards
  assert target.scene.terrain.spec_fn.__qualname__ == (
    source.scene.terrain.spec_fn.__qualname__
  )
  target.scene.terrain.spec_fn = source.scene.terrain.spec_fn
  assert target == source


def test_ppo_noise_symmetry_exact_except_initial_actual_saved_learning_rate():
  source = asdict(pochi_lp_acrl_blind_v11_ppo_runner_cfg())
  target = asdict(pochi_lp_acrl_blind_v12_ppo_runner_cfg())
  assert target["algorithm"]["learning_rate"] == SOURCE_LEARNING_RATE
  assert SOURCE_LEARNING_RATE == 2.2500000000000008e-05
  target["algorithm"]["learning_rate"] = source["algorithm"]["learning_rate"]
  assert target == source


def test_external_weight_and_delta_without_changing_positive_yaw_tracking():
  cfg = pochi_lp_acrl_blind_v12_env_cfg()
  term = cfg.rewards["yaw_tracking_huber"]
  assert term.func is yaw_tracking_huber_error
  assert term.weight == -8.0
  assert term.params == {"delta": 0.3}
  assert cfg.rewards["track_ang_vel_z_exp"].weight == 0.5
  assert cfg.rewards["track_ang_vel_z_exp"].params["std"] == 0.15


def test_local_registration_is_idempotent_without_backend_install(monkeypatch):
  from pochi_rl.mjlab import warp_contact_backend
  from pochi_rl.mjlab.lp_acrl_yaw_stability_refinement import (
    LPACRLYawStabilityOnPolicyRunner,
  )

  def forbidden(*args, **kwargs):
    raise AssertionError("Registration must not install physics")

  monkeypatch.setattr(warp_contact_backend, "install_contact_backend", forbidden)
  entries = dict(registry._REGISTRY)
  entries.pop(BLIND_V12_TASK_ID, None)
  monkeypatch.setattr(registry, "_REGISTRY", entries)
  before = dict(entries)
  assert register_blind_v12() == BLIND_V12_TASK_ID
  registered = entries[BLIND_V12_TASK_ID]
  assert register_blind_v12() == BLIND_V12_TASK_ID
  assert entries[BLIND_V12_TASK_ID] is registered
  assert set(entries) == set(before) | {BLIND_V12_TASK_ID}
  assert all(entries[key] is value for key, value in before.items())
  assert registry.load_runner_cls(BLIND_V12_TASK_ID) is LPACRLYawStabilityOnPolicyRunner
