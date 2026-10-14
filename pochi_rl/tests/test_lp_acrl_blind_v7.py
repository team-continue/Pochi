"""Fresh v7 changes exploration/initial LR and preserves the whole v6 environment."""

from dataclasses import asdict

import pytest
from mjlab.tasks import registry

from pochi_rl.mjlab.lp_acrl_blind_v6_env_cfg import (
  pochi_lp_acrl_blind_v6_env_cfg,
  pochi_lp_acrl_blind_v6_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_blind_v7 import BLIND_V7_TASK_ID, register_blind_v7
from pochi_rl.mjlab.lp_acrl_blind_v7_env_cfg import (
  pochi_lp_acrl_blind_v7_env_cfg,
  pochi_lp_acrl_blind_v7_ppo_runner_cfg,
)


@pytest.mark.parametrize("play", [False, True])
def test_complete_environment_reward_physics_geometry_and_observations_match_v6(play):
  old = pochi_lp_acrl_blind_v6_env_cfg(play=play)
  new = pochi_lp_acrl_blind_v7_env_cfg(play=play)
  assert (
    new.scene.terrain.spec_fn.__qualname__ == old.scene.terrain.spec_fn.__qualname__
  )
  new.scene.terrain.spec_fn = old.scene.terrain.spec_fn
  assert new == old


def test_only_annealed_algorithm_initial_std_learning_rate_and_duration_change():
  old = asdict(pochi_lp_acrl_blind_v6_ppo_runner_cfg())
  new = asdict(pochi_lp_acrl_blind_v7_ppo_runner_cfg())
  assert new["actor"]["distribution_cfg"]["init_std"] == 0.35
  assert new["actor"]["distribution_cfg"]["std_range"] == (0.03, 0.35)
  assert new["actor"]["distribution_cfg"]["learn_std"] is False
  assert (
    new["algorithm"]["class_name"]
    == "pochi_rl.mjlab.lp_acrl_annealed:LPACRLAnnealedPPO"
  )
  assert new["algorithm"]["learning_rate"] == 1e-3
  assert new["max_iterations"] == 1500
  new["actor"]["distribution_cfg"] = old["actor"]["distribution_cfg"]
  for field in ("learning_rate", "class_name"):
    new["algorithm"][field] = old["algorithm"][field]
  new["max_iterations"] = old["max_iterations"]
  assert new == old


def test_registration_is_local_idempotent_and_never_installs_backend(monkeypatch):
  from pochi_rl.mjlab import warp_contact_backend
  from pochi_rl.mjlab.lp_acrl_annealed import LPACRLAnnealedOnPolicyRunner

  def forbidden(*args, **kwargs):
    raise AssertionError("Registration must not install a backend")

  monkeypatch.setattr(warp_contact_backend, "install_contact_backend", forbidden)
  entries = dict(registry._REGISTRY)
  entries.pop(BLIND_V7_TASK_ID, None)
  monkeypatch.setattr(registry, "_REGISTRY", entries)
  before = dict(entries)
  assert register_blind_v7() == BLIND_V7_TASK_ID
  entry = registry._REGISTRY[BLIND_V7_TASK_ID]
  assert register_blind_v7() == BLIND_V7_TASK_ID
  assert registry._REGISTRY[BLIND_V7_TASK_ID] is entry
  assert set(registry._REGISTRY) == set(before) | {BLIND_V7_TASK_ID}
  assert all(registry._REGISTRY[k] is v for k, v in before.items())
  assert registry.load_runner_cls(BLIND_V7_TASK_ID) is LPACRLAnnealedOnPolicyRunner
