"""Blind-v3 changes exactly two reward entries; earlier tasks remain untouched."""

from dataclasses import asdict

import pytest
from mjlab.tasks import registry

from pochi_rl.mjlab.lp_acrl_blind_v2 import (
  QUIET_STOP_ACTOR,
  pochi_lp_acrl_blind_v2_env_cfg,
  pochi_lp_acrl_blind_v2_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_blind_v3 import (
  BLIND_V3_TASK_ID,
  pochi_lp_acrl_blind_v3_ppo_runner_cfg,
  register_blind_v3,
)
from pochi_rl.mjlab.lp_acrl_blind_v3_env_cfg import pochi_lp_acrl_blind_v3_env_cfg
from pochi_rl.mjlab.lp_acrl_blind_v3_rewards import moving_contact_duty_variance
from pochi_rl.mjlab.lp_acrl_quiet_stop import LPACRLQuietStopOnPolicyRunner


@pytest.mark.parametrize("play", [False, True])
def test_v3_environment_changes_only_duty_term_and_slip_weight(play):
  old = pochi_lp_acrl_blind_v2_env_cfg(play=play)
  new = pochi_lp_acrl_blind_v3_env_cfg(play=play)
  added = new.rewards.pop("moving_contact_duty_variance")
  assert added.func is moving_contact_duty_variance
  assert added.weight == -8.0 and added.params == {"tau_s": 1.0}
  assert old.rewards["contact_slip"].weight == -0.2
  assert new.rewards["contact_slip"].weight == -8.0
  new.rewards["contact_slip"].weight = -0.2
  # Each construction makes an equivalent new terrain callback lambda.
  assert (
    new.scene.terrain.spec_fn.__qualname__ == old.scene.terrain.spec_fn.__qualname__
  )
  new.scene.terrain.spec_fn = old.scene.terrain.spec_fn
  assert new == old


def test_v3_keeps_quiet_stop_ppo_unchanged():
  old = pochi_lp_acrl_blind_v2_ppo_runner_cfg()
  new = pochi_lp_acrl_blind_v3_ppo_runner_cfg()
  assert asdict(new) == asdict(old)
  assert new.actor.class_name == QUIET_STOP_ACTOR


def test_v3_registration_is_explicit_idempotent_and_preserves_all_old_entries(
  monkeypatch,
):
  existing = dict(registry._REGISTRY)
  existing.pop(BLIND_V3_TASK_ID, None)
  monkeypatch.setattr(registry, "_REGISTRY", existing)
  before = dict(registry._REGISTRY)
  assert register_blind_v3() == BLIND_V3_TASK_ID
  registered = registry._REGISTRY[BLIND_V3_TASK_ID]
  assert register_blind_v3() == BLIND_V3_TASK_ID
  assert registry._REGISTRY[BLIND_V3_TASK_ID] is registered
  assert set(registry.list_tasks()) == set(before) | {BLIND_V3_TASK_ID}
  assert all(registry._REGISTRY[name] is value for name, value in before.items())
  assert registry.load_runner_cls(BLIND_V3_TASK_ID) is LPACRLQuietStopOnPolicyRunner
  assert registry.load_rl_cfg(BLIND_V3_TASK_ID).actor.class_name == QUIET_STOP_ACTOR
  for play in (False, True):
    cfg = registry.load_env_cfg(BLIND_V3_TASK_ID, play=play)
    assert cfg.rewards["contact_slip"].weight == -8.0
    assert (
      cfg.rewards["moving_contact_duty_variance"].func is moving_contact_duty_variance
    )
