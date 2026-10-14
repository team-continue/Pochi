"""V15 removes positive touchdown credit without changing the rest of V13."""

from dataclasses import asdict
from types import SimpleNamespace

import pytest
from mjlab.tasks import registry

from pochi_rl.lp_acrl import FLAT_TASK_CATALOG
from pochi_rl.mjlab.lp_acrl_blind_v13_env_cfg import (
  pochi_lp_acrl_blind_v13_env_cfg,
  pochi_lp_acrl_blind_v13_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_blind_v15 import BLIND_V15_TASK_ID, register_blind_v15
from pochi_rl.mjlab.lp_acrl_blind_v15_env_cfg import (
  SOURCE_LEARNING_RATE,
  pochi_lp_acrl_blind_v15_env_cfg,
  pochi_lp_acrl_blind_v15_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_commands import LPACRLVelocityCommand


@pytest.mark.parametrize("play", [False, True])
def test_only_touchdown_reward_is_removed_from_complete_flat_environment(play):
  source = pochi_lp_acrl_blind_v13_env_cfg(play=play)
  target = pochi_lp_acrl_blind_v15_env_cfg(play=play)
  assert len(source.rewards) == 23
  assert len(target.rewards) == 22
  assert "balanced_touchdown" not in target.rewards
  assert source.rewards.pop("balanced_touchdown").weight == 2.0
  assert target.rewards == source.rewards
  # The unused inherited EntityCfg callback is freshly allocated per factory.
  assert target.scene.terrain.spec_fn.__qualname__ == (
    source.scene.terrain.spec_fn.__qualname__
  )
  target.scene.terrain.spec_fn = source.scene.terrain.spec_fn
  assert target == source


def test_no_touchdown_alias_phase_or_scan_is_added_and_command_scope_is_unchanged():
  cfg = pochi_lp_acrl_blind_v15_env_cfg()
  assert all("touchdown" not in name for name in cfg.rewards)
  assert "feet_air_time" not in cfg.rewards
  assert cfg.sim.mujoco.timestep == 0.0025
  assert cfg.decimation == 8
  assert "terrain_scan" not in {sensor.name for sensor in cfg.scene.sensors}
  source = pochi_lp_acrl_blind_v13_env_cfg()
  assert cfg.observations == source.observations
  for group in cfg.observations.values():
    assert "height_scan" not in group.terms
    assert not any("phase" in name or "clock" in name for name in group.terms)
  assert cfg.commands == source.commands
  assert cfg.commands["base_velocity"].ranges.lin_vel_y == (0.0, 0.0)
  assert cfg.commands["base_velocity"].ranges.ang_vel_z == (0.0, 0.0)
  assert cfg.scene.entities["robot"] == source.scene.entities["robot"]


def test_canonical_runtime_identity_is_the_exact_single_reward_removal():
  from pochi_rl.mjlab.lp_acrl_no_touchdown_refinement import (
    expected_no_touchdown_task_space,
  )

  cfg = pochi_lp_acrl_blind_v15_env_cfg()
  term = LPACRLVelocityCommand.__new__(LPACRLVelocityCommand)
  term._env = SimpleNamespace(cfg=cfg)
  term.cfg = cfg.commands["base_velocity"]
  term.catalog = FLAT_TASK_CATALOG
  assert term.checkpoint_identity() == expected_no_touchdown_task_space()
  assert term.catalog.num_tasks == 5


def test_only_ppo_change_is_initial_lr_from_actual_parent_checkpoint():
  source = asdict(pochi_lp_acrl_blind_v13_ppo_runner_cfg())
  target = asdict(pochi_lp_acrl_blind_v15_ppo_runner_cfg())
  assert target["algorithm"]["learning_rate"] == SOURCE_LEARNING_RATE
  assert SOURCE_LEARNING_RATE == 5.0625000000000024e-05
  target["algorithm"]["learning_rate"] = source["algorithm"]["learning_rate"]
  assert target == source


def test_factory_never_removes_touchdown_from_source_or_shares_remaining_rewards():
  first = pochi_lp_acrl_blind_v15_env_cfg()
  second = pochi_lp_acrl_blind_v15_env_cfg()
  first.rewards["contact_slip"].weight = -1.0
  assert second.rewards["contact_slip"].weight == -8.0
  source = pochi_lp_acrl_blind_v13_env_cfg()
  assert source.rewards["balanced_touchdown"].weight == 2.0
  assert source.rewards["contact_slip"].weight == -8.0


def test_local_registration_is_idempotent_without_installing_physics(monkeypatch):
  from pochi_rl.mjlab import warp_contact_backend
  from pochi_rl.mjlab.lp_acrl_no_touchdown_refinement import (
    LPACRLNoTouchdownOnPolicyRunner,
  )

  def forbidden(*args, **kwargs):
    raise AssertionError("Registration must not install physics")

  monkeypatch.setattr(warp_contact_backend, "install_contact_backend", forbidden)
  entries = dict(registry._REGISTRY)
  entries.pop(BLIND_V15_TASK_ID, None)
  monkeypatch.setattr(registry, "_REGISTRY", entries)
  before = dict(entries)
  assert register_blind_v15() == BLIND_V15_TASK_ID
  registered = entries[BLIND_V15_TASK_ID]
  assert register_blind_v15() == BLIND_V15_TASK_ID
  assert entries[BLIND_V15_TASK_ID] is registered
  assert set(entries) == set(before) | {BLIND_V15_TASK_ID}
  assert all(entries[key] is value for key, value in before.items())
  assert registry.load_runner_cls(BLIND_V15_TASK_ID) is LPACRLNoTouchdownOnPolicyRunner
