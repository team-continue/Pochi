"""The flat-focus task narrows terrain/commands without changing its controller."""

from dataclasses import asdict
from types import SimpleNamespace

import mujoco
import pytest
import torch
from mjlab.tasks import registry
from mjlab.terrains import TerrainEntity

from pochi_rl.lp_acrl import FLAT_TASK_CATALOG
from pochi_rl.mjlab.lp_acrl_blind_v12_env_cfg import (
  pochi_lp_acrl_blind_v12_env_cfg,
  pochi_lp_acrl_blind_v12_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_blind_v13 import BLIND_V13_TASK_ID, register_blind_v13
from pochi_rl.mjlab.lp_acrl_blind_v13_env_cfg import (
  SOURCE_LEARNING_RATE,
  pochi_lp_acrl_blind_v13_env_cfg,
  pochi_lp_acrl_blind_v13_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_commands import LPACRLVelocityCommand
from pochi_rl.mjlab.lp_acrl_flat_focus_identity import (
  canonical_rough_task_space,
  expected_flat_focus_task_space,
)
from pochi_rl.mjlab.lp_acrl_rewards import outside_task_terrain


@pytest.mark.parametrize("play", [False, True])
def test_only_declared_scope_changes_and_all_23_rewards_unchanged(play):
  source = pochi_lp_acrl_blind_v12_env_cfg(play=play)
  target = pochi_lp_acrl_blind_v13_env_cfg(play=play)
  assert len(target.rewards) == 23
  assert target.rewards == source.rewards
  assert target.terminations == source.terminations
  assert target.terminations["task_terrain_edge"].func is outside_task_terrain
  assert target.scene.spec_fn is None
  assert target.scene.terrain.terrain_type == "plane"
  assert target.scene.terrain.terrain_generator is None
  assert target.commands["base_velocity"].rough is False
  assert target.commands["base_velocity"].ranges.lin_vel_y == (0.0, 0.0)
  assert target.commands["base_velocity"].ranges.ang_vel_z == (0.0, 0.0)
  target.commands["base_velocity"].rough = True
  target.commands["base_velocity"].ranges = source.commands["base_velocity"].ranges
  target.scene.terrain.terrain_type = source.scene.terrain.terrain_type
  target.scene.terrain.terrain_generator = source.scene.terrain.terrain_generator
  # The inherited unused terrain EntityCfg lambda is recreated by each factory.
  target.scene.terrain.spec_fn = source.scene.terrain.spec_fn
  assert target == source


def test_ppo_and_actual_source_learning_rate_stay_exactly_v12():
  assert asdict(pochi_lp_acrl_blind_v13_ppo_runner_cfg()) == asdict(
    pochi_lp_acrl_blind_v12_ppo_runner_cfg()
  )
  assert SOURCE_LEARNING_RATE == 2.2500000000000008e-05
  cfg = pochi_lp_acrl_blind_v13_env_cfg()
  assert cfg.sim.mujoco.timestep == 0.0025
  assert cfg.decimation == 8
  assert all(not group.enable_corruption for group in cfg.observations.values())
  assert "terrain_scan" not in {sensor.name for sensor in cfg.scene.sensors}
  assert all("height_scan" not in group.terms for group in cfg.observations.values())


def test_real_native_terrain_is_one_infinite_plane_and_no_heightfield_or_box():
  cfg = pochi_lp_acrl_blind_v13_env_cfg().scene.terrain
  cfg.num_envs = 4
  terrain = TerrainEntity(cfg, device="cpu")
  model = terrain.spec.compile()
  assert model.ngeom == 1
  assert int(model.geom_type[0]) == int(mujoco.mjtGeom.mjGEOM_PLANE)
  assert model.nhfield == 0
  assert terrain.terrain_origins is None
  assert terrain.env_origins.shape == (4, 3)
  assert torch.equal(terrain.env_origins[:, 2], torch.zeros(4))
  assert model.geom_pos[0, 2] == 0.0


def test_real_checkpoint_identity_matches_explicit_scope_projection():
  cfg = pochi_lp_acrl_blind_v13_env_cfg()
  term = LPACRLVelocityCommand.__new__(LPACRLVelocityCommand)
  term._env = SimpleNamespace(cfg=cfg)
  term.cfg = cfg.commands["base_velocity"]
  term.catalog = FLAT_TASK_CATALOG
  assert term.checkpoint_identity() == expected_flat_focus_task_space(
    canonical_rough_task_space()
  )
  assert term.catalog.num_tasks == 5


def test_fixed_and_stop_start_sampling_are_signed_forward_only_and_lp_has_five_tasks():
  from test_lp_acrl_env import _command_env

  env, _ = _command_env(num_envs=100)
  env.cfg = pochi_lp_acrl_blind_v13_env_cfg()
  term = env.cfg.commands["base_velocity"].build(env)
  ids = torch.arange(env.num_envs)
  torch.manual_seed(912)
  term.prepare_episode(ids)
  term.task_ids[:] = ids % 5
  term.episode_mode[:] = ids % 3
  term._resample_command(ids)
  assert term.catalog is FLAT_TASK_CATALOG
  assert term.curriculum.num_tasks == 5
  assert term.curriculum.probabilities.shape == (5,)
  assert torch.equal(term.vel_command_b[:, 1:], torch.zeros(100, 2))
  assert torch.equal(term._movement_command[:, 1:], torch.zeros(100, 2))
  assert (term.vel_command_b[:, 0] > 0).any()
  assert (term.vel_command_b[:, 0] < 0).any()
  assert (term.vel_command_b[term.episode_mode == 1] == 0).all()
  term._transition_time_left[:] = 0.0
  env.episode_length_buf[:] = 1
  term.compute(0.02)
  assert torch.equal(term.vel_command_b[:, 1:], torch.zeros(100, 2))
  assert (term.vel_command_b[term.episode_mode == 1] == 0).all()
  env.termination_manager.dones[:] = True
  env.episode_length_buf[:] = 1000
  fixed_count = int((term.episode_mode == 0).sum())
  term.prepare_episode(ids)
  assert int(term.curriculum.episode_counts.sum()) == fixed_count


def test_local_registration_is_idempotent_and_does_not_install_backend(monkeypatch):
  from pochi_rl.mjlab import warp_contact_backend
  from pochi_rl.mjlab.lp_acrl_flat_focus_refinement import (
    LPACRLFlatFocusOnPolicyRunner,
  )

  def forbidden(*args, **kwargs):
    raise AssertionError("Registration must not install physics")

  monkeypatch.setattr(warp_contact_backend, "install_contact_backend", forbidden)
  entries = dict(registry._REGISTRY)
  entries.pop(BLIND_V13_TASK_ID, None)
  monkeypatch.setattr(registry, "_REGISTRY", entries)
  before = dict(entries)
  assert register_blind_v13() == BLIND_V13_TASK_ID
  registered = entries[BLIND_V13_TASK_ID]
  assert register_blind_v13() == BLIND_V13_TASK_ID
  assert entries[BLIND_V13_TASK_ID] is registered
  assert set(entries) == set(before) | {BLIND_V13_TASK_ID}
  assert all(entries[key] is value for key, value in before.items())
  assert registry.load_runner_cls(BLIND_V13_TASK_ID) is LPACRLFlatFocusOnPolicyRunner
