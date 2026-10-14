"""V9 changes only v8 slip weight while reusing the complete v7 fresh PPO."""

from dataclasses import asdict

import pytest
from mjlab.tasks import registry

from pochi_rl.mjlab.lp_acrl_blind_v7_env_cfg import (
  pochi_lp_acrl_blind_v7_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_blind_v8_env_cfg import pochi_lp_acrl_blind_v8_env_cfg
from pochi_rl.mjlab.lp_acrl_blind_v9 import BLIND_V9_TASK_ID, register_blind_v9
from pochi_rl.mjlab.lp_acrl_blind_v9_env_cfg import (
  pochi_lp_acrl_blind_v9_env_cfg,
  pochi_lp_acrl_blind_v9_ppo_runner_cfg,
)


@pytest.mark.parametrize("play", [False, True])
def test_complete_environment_differs_only_by_declared_slip_weight(play):
  old = pochi_lp_acrl_blind_v8_env_cfg(play=play)
  new = pochi_lp_acrl_blind_v9_env_cfg(play=play)
  assert old.rewards["contact_slip"].weight == -8.0
  assert new.rewards["contact_slip"].weight == -0.2
  assert "moving_default_joint_deviation" not in new.rewards
  assert new.rewards["moving_common_hip_pitch_deviation"].weight == -4.0
  new.rewards["contact_slip"].weight = old.rewards["contact_slip"].weight
  assert (
    new.scene.terrain.spec_fn.__qualname__ == old.scene.terrain.spec_fn.__qualname__
  )
  new.scene.terrain.spec_fn = old.scene.terrain.spec_fn
  assert new == old


def test_entire_v7_annealed_ppo_model_noise_symmetry_and_optimizer_are_unchanged():
  assert asdict(pochi_lp_acrl_blind_v9_ppo_runner_cfg()) == asdict(
    pochi_lp_acrl_blind_v7_ppo_runner_cfg()
  )


def test_registration_is_local_idempotent_and_never_installs_backend(monkeypatch):
  from pochi_rl.mjlab import warp_contact_backend
  from pochi_rl.mjlab.lp_acrl_annealed import LPACRLAnnealedOnPolicyRunner

  def forbidden(*args, **kwargs):
    raise AssertionError("Registration must not install a backend")

  monkeypatch.setattr(warp_contact_backend, "install_contact_backend", forbidden)
  entries = dict(registry._REGISTRY)
  entries.pop(BLIND_V9_TASK_ID, None)
  monkeypatch.setattr(registry, "_REGISTRY", entries)
  before = dict(entries)
  assert register_blind_v9() == BLIND_V9_TASK_ID
  entry = registry._REGISTRY[BLIND_V9_TASK_ID]
  assert register_blind_v9() == BLIND_V9_TASK_ID
  assert registry._REGISTRY[BLIND_V9_TASK_ID] is entry
  assert set(registry._REGISTRY) == set(before) | {BLIND_V9_TASK_ID}
  assert all(registry._REGISTRY[k] is v for k, v in before.items())
  assert registry.load_runner_cls(BLIND_V9_TASK_ID) is LPACRLAnnealedOnPolicyRunner
