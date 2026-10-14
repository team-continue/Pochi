"""Exact scope projection and rejection without constructing a simulator."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from pochi_rl.lp_acrl import FLAT_TASK_CATALOG, ROUGH_TASK_CATALOG
from pochi_rl.mjlab.lp_acrl_blind_v11_env_cfg import pochi_lp_acrl_blind_v11_env_cfg
from pochi_rl.mjlab.lp_acrl_blind_v12_env_cfg import pochi_lp_acrl_blind_v12_env_cfg
from pochi_rl.mjlab.lp_acrl_commands import LPACRLVelocityCommand
from pochi_rl.mjlab.lp_acrl_flat_focus_identity import (
  FLAT_FOCUS_TASK_SPACE_PATHS,
  FLAT_FOCUS_TRAINING_PATHS,
  canonical_rough_task_space,
  expected_flat_focus_task_space,
  validate_flat_focus_refinement_identity,
)


def _identity(cfg):
  generator = cfg.scene.terrain.terrain_generator
  if generator is not None:
    for family in generator.sub_terrains.values():
      family.size = generator.size
  term = LPACRLVelocityCommand.__new__(LPACRLVelocityCommand)
  term._env = SimpleNamespace(cfg=cfg)
  term.cfg = cfg.commands["base_velocity"]
  term.catalog = ROUGH_TASK_CATALOG if term.cfg.rough else FLAT_TASK_CATALOG
  return term.checkpoint_identity()


def _flat_identity():
  cfg = pochi_lp_acrl_blind_v12_env_cfg()
  command = cfg.commands["base_velocity"]
  command.rough = False
  command.ranges.lin_vel_y = (0.0, 0.0)
  command.ranges.ang_vel_z = (0.0, 0.0)
  cfg.scene.terrain.terrain_type = "plane"
  cfg.scene.terrain.terrain_generator = None
  return _identity(cfg)


def _validate(rough, flat, source_seed=42, target_seed=None):
  return validate_flat_focus_refinement_identity(
    rough,
    flat,
    source_training_terrain_seed=source_seed,
    target_training_terrain_seed=target_seed,
  )


def test_real_command_identity_matches_five_task_projection_and_exact_rewards():
  rough = canonical_rough_task_space()
  flat = _flat_identity()
  assert expected_flat_focus_task_space(rough) == flat
  assert _validate(rough, flat) == FLAT_FOCUS_TRAINING_PATHS
  assert flat["forward_edges"] == rough["forward_edges"]
  assert flat["yaw_edges"] == (0.0, 0.0)
  assert flat["terrains"] == ("flat",) and flat["num_levels"] == 1
  assert flat["signed_forward"] is True and flat["signed_yaw"] is False
  assert flat["lateral_range"] == (0.0, 0.0)
  assert len(flat["environment"]["rewards"]) == 23
  assert flat["environment"]["rewards"] == rough["environment"]["rewards"]
  assert flat["environment"]["terminations"] == rough["environment"]["terminations"]
  assert "task_terrain_edge" in flat["environment"]["terminations"]
  assert flat["command_protocol"] == rough["command_protocol"]
  assert set(FLAT_FOCUS_TASK_SPACE_PATHS) == {
    "yaw_edges",
    "terrains",
    "num_levels",
    "signed_yaw",
    "lateral_range",
    "environment.terrain_type",
    "environment.terrain_generator",
  }


def test_builder_does_not_install_backend_or_construct_command_or_simulator(
  monkeypatch,
):
  from mjlab.envs import ManagerBasedRlEnv

  from pochi_rl.mjlab import lp_acrl_flat_focus_identity as module
  from pochi_rl.mjlab import warp_contact_backend

  def forbidden(*args, **kwargs):
    raise AssertionError("Identity projection must not create runtime state")

  monkeypatch.setattr(ManagerBasedRlEnv, "__init__", forbidden)
  monkeypatch.setattr(LPACRLVelocityCommand, "__init__", forbidden)
  monkeypatch.setattr(warp_contact_backend, "install_contact_backend", forbidden)
  module._canonical_rough_task_space.cache_clear()
  assert expected_flat_focus_task_space(canonical_rough_task_space())["num_levels"] == 1


def test_projection_never_mutates_inputs_or_shares_nested_cached_state():
  rough = canonical_rough_task_space()
  before = deepcopy(rough)
  flat = expected_flat_focus_task_space(rough)
  _validate(rough, flat)
  assert rough == before
  flat["environment"]["rewards"]["yaw_tracking_huber"]["weight"] = -80
  rough["command_protocol"]["standing_episode_fraction"] = 0
  assert canonical_rough_task_space() == before
  assert expected_flat_focus_task_space(before) != flat


def test_raw_v11_is_rejected_until_caller_validates_huber_addition():
  source = _identity(pochi_lp_acrl_blind_v11_env_cfg())
  with pytest.raises(ValueError, match="exact V12"):
    expected_flat_focus_task_space(source)


@pytest.mark.parametrize("side", ["rough", "flat"])
@pytest.mark.parametrize(
  "field",
  [
    "physics",
    "decimation",
    "actions",
    "robot_articulation",
    "robot_default_pose",
    "rewards",
    "terminations",
    "events",
    "sensors",
    "observations",
    "scale_rewards_by_dt",
    "episode_length_s",
  ],
)
def test_any_nonterrain_environment_difference_rejected(side, field):
  rough, flat = canonical_rough_task_space(), _flat_identity()
  selected = rough if side == "rough" else flat
  selected["environment"][field] = {"unexpected": True}
  with pytest.raises(ValueError):
    _validate(rough, flat)


@pytest.mark.parametrize(
  "field,value",
  [
    ("forward_edges", (0.0, 0.25, 0.5)),
    ("yaw_edges", (0.0, 0.2)),
    ("terrains", ("ascending_stairs",)),
    ("num_levels", 4),
    ("signed_forward", False),
    ("signed_yaw", True),
    ("lateral_range", (-0.1, 0.1)),
    ("command_protocol", {}),
  ],
)
def test_wrong_catalog_command_scope_or_stop_protocol_rejected(field, value):
  rough, flat = canonical_rough_task_space(), _flat_identity()
  flat[field] = value
  with pytest.raises(ValueError, match="declared scope"):
    _validate(rough, flat)


@pytest.mark.parametrize(
  "field,value",
  [
    ("terrain_type", "generator"),
    ("terrain_generator", {}),
  ],
)
def test_target_must_have_plane_and_no_generator(field, value):
  rough, flat = canonical_rough_task_space(), _flat_identity()
  flat["environment"][field] = value
  with pytest.raises(ValueError, match="declared scope"):
    _validate(rough, flat)


@pytest.mark.parametrize(
  "source_seed,target_seed",
  [
    (None, None),
    (123, None),
    (42.0, None),
    (True, None),
    (42, 42),
    (42, 0),
  ],
)
def test_training_seed_transition_explicit_and_strict(source_seed, target_seed):
  with pytest.raises(ValueError, match="source terrain seed42"):
    _validate(canonical_rough_task_space(), _flat_identity(), source_seed, target_seed)


@pytest.mark.parametrize("fault", ["missing", "extra", "bool_as_int", "list_as_tuple"])
def test_missing_extra_or_type_coerced_metadata_rejected(fault):
  rough, flat = canonical_rough_task_space(), _flat_identity()
  if fault == "missing":
    flat.pop("signed_forward")
  elif fault == "extra":
    flat["unapproved"] = True
  elif fault == "bool_as_int":
    flat["num_levels"] = True
  else:
    flat["yaw_edges"] = [0.0, 0.0]
  with pytest.raises(ValueError, match="declared scope"):
    _validate(rough, flat)


def test_native_actor_bridge_projection_preserves_real_rough_catalog():
  native = _identity(pochi_lp_acrl_blind_v12_env_cfg(play=True))
  before = deepcopy(native)
  flat_saved = _flat_identity()
  assert expected_flat_focus_task_space(native) == flat_saved
  assert native == before and native["terrains"] == ROUGH_TASK_CATALOG.terrains
  assert ROUGH_TASK_CATALOG.num_tasks == 600
  assert FLAT_TASK_CATALOG.num_tasks == 5
  assert 360 >= FLAT_TASK_CATALOG.num_tasks


def test_canonical_rough_sizes_match_native_generator_config_finalization():
  from mjlab.terrains.terrain_generator import TerrainGenerator

  cfg = pochi_lp_acrl_blind_v12_env_cfg()
  generator = cfg.scene.terrain.terrain_generator
  assert generator is not None
  assert all(family.size == (10.0, 10.0) for family in generator.sub_terrains.values())
  TerrainGenerator(generator, device="cpu")
  assert all(family.size == (24.0, 24.0) for family in generator.sub_terrains.values())
  assert _identity(cfg) == canonical_rough_task_space()


def test_arbitrary_or_unfinalized_rough_family_size_is_not_silently_normalized():
  rough = canonical_rough_task_space()
  rough["environment"]["terrain_generator"]["sub_terrains"]["rough"]["size"] = (
    10.0,
    10.0,
  )
  with pytest.raises(ValueError, match="exact V12"):
    expected_flat_focus_task_space(rough)
