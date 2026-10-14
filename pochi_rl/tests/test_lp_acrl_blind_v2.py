"""Quiet-stop exploration does not silently change the v1 environment or PPO."""

from dataclasses import asdict

import pytest
from mjlab.tasks import registry
from rsl_rl.utils import resolve_callable

from pochi_rl.mjlab.lp_acrl_blind_v1_env_cfg import pochi_lp_acrl_blind_v1_env_cfg
from pochi_rl.mjlab.lp_acrl_blind_v2 import (
  BLIND_V2_TASK_ID,
  QUIET_STOP_ACTOR,
  pochi_lp_acrl_blind_v2_env_cfg,
  pochi_lp_acrl_blind_v2_ppo_runner_cfg,
  register_blind_v2,
)
from pochi_rl.mjlab.lp_acrl_quiet_stop import (
  LPACRLQuietStopOnPolicyRunner,
  QuietStopMLPModel,
)
from pochi_rl.mjlab.lp_acrl_runner import pochi_lp_acrl_ppo_runner_cfg


def _assert_equivalent_environments(left, right):
  # Terrain config construction creates an equivalent fresh lambda each time.
  assert (
    left.scene.terrain.spec_fn.__qualname__ == right.scene.terrain.spec_fn.__qualname__
  )
  left.scene.terrain.spec_fn = right.scene.terrain.spec_fn
  assert left == right


@pytest.mark.parametrize("play", [False, True])
def test_v2_environment_is_structurally_identical_to_v1(play):
  old = pochi_lp_acrl_blind_v1_env_cfg(play=play)
  new = pochi_lp_acrl_blind_v2_env_cfg(play=play)
  _assert_equivalent_environments(old, new)
  command = new.commands["base_velocity"]
  assert command.standing_episode_fraction == 0.20
  assert command.transition_episode_fraction == 0.10
  assert command.stage_steps == 2400
  assert set(new.scene.entities) == {"robot"}
  assert [sensor.name for sensor in new.scene.sensors] == [
    "feet_ground_contact",
    "limb_ground_contact",
  ]
  assert all("height_scan" not in group.terms for group in new.observations.values())


def test_v2_ppo_only_changes_actor_class():
  old = asdict(pochi_lp_acrl_ppo_runner_cfg(rough=True, blind=True))
  new = asdict(pochi_lp_acrl_blind_v2_ppo_runner_cfg())
  assert old["actor"]["class_name"] == "MLPModel"
  assert new["actor"]["class_name"] == QUIET_STOP_ACTOR
  new["actor"]["class_name"] = old["actor"]["class_name"]
  assert new == old
  assert resolve_callable(QUIET_STOP_ACTOR) is QuietStopMLPModel
  assert QuietStopMLPModel.QUIET_STD == 0.01
  assert QuietStopMLPModel.STOP_THRESHOLD == 1.0e-6


def test_v2_registration_is_explicit_idempotent_and_uses_quiet_stop_runner(monkeypatch):
  existing = dict(registry._REGISTRY)
  existing.pop(BLIND_V2_TASK_ID, None)
  monkeypatch.setattr(registry, "_REGISTRY", existing)
  before = dict(registry._REGISTRY)
  assert BLIND_V2_TASK_ID not in registry.list_tasks()
  assert register_blind_v2() == BLIND_V2_TASK_ID
  registered = registry._REGISTRY[BLIND_V2_TASK_ID]
  assert register_blind_v2() == BLIND_V2_TASK_ID
  assert registry._REGISTRY[BLIND_V2_TASK_ID] is registered
  assert set(registry.list_tasks()) == set(before) | {BLIND_V2_TASK_ID}
  assert all(registry._REGISTRY[name] is cfg for name, cfg in before.items())
  assert registry.load_runner_cls(BLIND_V2_TASK_ID) is LPACRLQuietStopOnPolicyRunner
  assert registry.load_rl_cfg(BLIND_V2_TASK_ID).actor.class_name == QUIET_STOP_ACTOR
  _assert_equivalent_environments(
    registry.load_env_cfg(BLIND_V2_TASK_ID), pochi_lp_acrl_blind_v1_env_cfg()
  )
  _assert_equivalent_environments(
    registry.load_env_cfg(BLIND_V2_TASK_ID, play=True),
    pochi_lp_acrl_blind_v1_env_cfg(play=True),
  )
