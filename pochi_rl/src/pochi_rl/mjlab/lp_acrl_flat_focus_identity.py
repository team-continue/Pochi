"""Strict scope projection from the V12 objective to five blind plane tasks.

The caller first validates the explicit V11-to-V12 Huber addition. This module
then changes only the task catalog, commanded-motion scope and terrain identity.
It does not load models, rewrite checkpoint metadata, install a backend or build
an environment. In particular, every reward and termination stays unchanged.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from functools import lru_cache
from types import SimpleNamespace
from typing import Any

from pochi_rl.lp_acrl import FLAT_TASK_CATALOG, ROUGH_TASK_CATALOG
from pochi_rl.mjlab.lp_acrl_blind_v12_env_cfg import pochi_lp_acrl_blind_v12_env_cfg
from pochi_rl.mjlab.lp_acrl_commands import LPACRLVelocityCommand

FLAT_FOCUS_TASK_SPACE_PATHS = (
  "yaw_edges",
  "terrains",
  "num_levels",
  "signed_yaw",
  "lateral_range",
  "environment.terrain_type",
  "environment.terrain_generator",
)
FLAT_FOCUS_TRAINING_PATHS = (
  *FLAT_FOCUS_TASK_SPACE_PATHS,
  "training_terrain_seed",
)


@lru_cache(maxsize=1)
def _canonical_rough_task_space() -> dict[str, Any]:
  cfg = pochi_lp_acrl_blind_v12_env_cfg()
  # TerrainGenerator.__init__ finalizes each family to the shared tile size.
  # Mirror only that deterministic config mutation, without generating geometry,
  # so this canonical identity matches saved and live native rough environments.
  generator = cfg.scene.terrain.terrain_generator
  assert generator is not None
  for family in generator.sub_terrains.values():
    family.size = generator.size
  # checkpoint_identity reads configuration only. Avoid the command constructor,
  # which needs a simulated robot and allocates episode/curriculum state.
  term = LPACRLVelocityCommand.__new__(LPACRLVelocityCommand)
  term._env = SimpleNamespace(cfg=cfg)
  term.cfg = cfg.commands["base_velocity"]
  term.catalog = ROUGH_TASK_CATALOG
  return term.checkpoint_identity()


def canonical_rough_task_space() -> dict[str, Any]:
  """Return an independent CPU-only copy of the complete frozen V12 identity."""
  return deepcopy(_canonical_rough_task_space())


def _first_difference(first: Any, second: Any, path: str = "task_space") -> str | None:
  if type(first) is not type(second):
    return path
  if isinstance(first, dict):
    if first.keys() != second.keys():
      return path
    for key, value in first.items():
      mismatch = _first_difference(value, second[key], f"{path}.{key}")
      if mismatch is not None:
        return mismatch
    return None
  if isinstance(first, (tuple, list)):
    if len(first) != len(second):
      return path
    for index, (left, right) in enumerate(zip(first, second, strict=True)):
      mismatch = _first_difference(left, right, f"{path}[{index}]")
      if mismatch is not None:
        return mismatch
    return None
  return None if first == second else path


def expected_flat_focus_task_space(rough_task_space: dict) -> dict:
  """Project an exact V12 rough identity; never normalize an arbitrary source.

  A raw V11 identity is rejected because it lacks the separately authorized
  Huber reward. For a native actor-only bridge, project the actual original V12
  rough environment and compare with the saved flat identity. This does not
  turn a flat catalog ID into rough task360 or authorize training-state resume.
  """
  mismatch = _first_difference(_canonical_rough_task_space(), rough_task_space)
  if mismatch is not None:
    raise ValueError(f"Require the exact V12 rough task identity: {mismatch}")
  result = deepcopy(rough_task_space)
  result.update(asdict(FLAT_TASK_CATALOG))
  result["signed_yaw"] = False
  result["lateral_range"] = (0.0, 0.0)
  result["environment"]["terrain_type"] = "plane"
  result["environment"]["terrain_generator"] = None
  return result


def validate_flat_focus_refinement_identity(
  rough_task_space: dict,
  flat_task_space: dict,
  *,
  source_training_terrain_seed: int | None,
  target_training_terrain_seed: int | None,
) -> tuple[str, ...]:
  """Validate the full scope-only projection and explicit generator42-to-none.

  Models, source checkpoint SHA, optimizer/LP freshness, runtime observation
  corruption, exploration, backend and compiled ground are the caller's separate
  contracts. Task-space identity alone cannot prove which geometry was compiled.
  """
  expected = expected_flat_focus_task_space(rough_task_space)
  mismatch = _first_difference(expected, flat_task_space)
  if mismatch is not None:
    raise ValueError(f"Flat focus differs outside the declared scope: {mismatch}")
  source_cfg = pochi_lp_acrl_blind_v12_env_cfg()
  generator = source_cfg.scene.terrain.terrain_generator
  assert generator is not None
  if (
    type(source_training_terrain_seed) is not int
    or source_training_terrain_seed != generator.seed
    or target_training_terrain_seed is not None
  ):
    raise ValueError(
      "Flat focus requires source terrain seed42 and no target generator"
    )
  return FLAT_FOCUS_TRAINING_PATHS
