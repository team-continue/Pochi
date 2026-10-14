"""V10 changes only fixed moving exploration and uses the guarded continuation."""

from dataclasses import asdict

import pytest
from mjlab.tasks import registry

from pochi_rl.mjlab.lp_acrl_blind_v8_env_cfg import (
  pochi_lp_acrl_blind_v8_env_cfg,
  pochi_lp_acrl_blind_v8_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_blind_v10 import BLIND_V10_TASK_ID, register_blind_v10
from pochi_rl.mjlab.lp_acrl_blind_v10_env_cfg import (
  pochi_lp_acrl_blind_v10_env_cfg,
  pochi_lp_acrl_blind_v10_ppo_runner_cfg,
)


@pytest.mark.parametrize("play", [False, True])
def test_full_environment_and_all_rewards_are_identical_to_v8(play):
  old = pochi_lp_acrl_blind_v8_env_cfg(play=play)
  new = pochi_lp_acrl_blind_v10_env_cfg(play=play)
  assert (
    new.scene.terrain.spec_fn.__qualname__ == old.scene.terrain.spec_fn.__qualname__
  )
  new.scene.terrain.spec_fn = old.scene.terrain.spec_fn
  assert new == old


def test_only_moving_distribution_initial_value_and_range_change():
  old = asdict(pochi_lp_acrl_blind_v8_ppo_runner_cfg())
  new = asdict(pochi_lp_acrl_blind_v10_ppo_runner_cfg())
  old_distribution = old["actor"]["distribution_cfg"]
  new_distribution = new["actor"]["distribution_cfg"]
  assert old_distribution["init_std"] == 0.03
  assert old_distribution["std_range"] == (0.03, 0.03)
  assert new_distribution["init_std"] == 0.15
  assert new_distribution["std_range"] == (0.15, 0.15)
  assert new_distribution["learn_std"] is False
  new_distribution["init_std"] = old_distribution["init_std"]
  new_distribution["std_range"] = old_distribution["std_range"]
  assert new == old


def test_local_registration_uses_full_state_runner_without_installing_backend(
  monkeypatch,
):
  from pochi_rl.mjlab import warp_contact_backend
  from pochi_rl.mjlab.lp_acrl_wider_exploration import (
    LPACRLWiderExplorationOnPolicyRunner,
  )

  def forbidden(*args, **kwargs):
    raise AssertionError("Registration must not install a backend")

  monkeypatch.setattr(warp_contact_backend, "install_contact_backend", forbidden)
  entries = dict(registry._REGISTRY)
  entries.pop(BLIND_V10_TASK_ID, None)
  monkeypatch.setattr(registry, "_REGISTRY", entries)
  before = dict(entries)
  assert register_blind_v10() == BLIND_V10_TASK_ID
  entry = registry._REGISTRY[BLIND_V10_TASK_ID]
  assert register_blind_v10() == BLIND_V10_TASK_ID
  assert registry._REGISTRY[BLIND_V10_TASK_ID] is entry
  assert set(registry._REGISTRY) == set(before) | {BLIND_V10_TASK_ID}
  assert all(registry._REGISTRY[k] is v for k, v in before.items())
  assert registry.load_runner_cls(BLIND_V10_TASK_ID) is (
    LPACRLWiderExplorationOnPolicyRunner
  )
