"""Physical checkpoint compatibility without starting a simulator."""

import json
from dataclasses import replace
from types import SimpleNamespace

import pytest

from pochi_rl.lp_acrl import (
  FLAT_TASK_CATALOG,
  ROUGH_TASK_CATALOG,
  LPACRLCurriculum,
)
from pochi_rl.mjlab.lp_acrl_commands import LPACRLVelocityCommand
from pochi_rl.mjlab.lp_acrl_env_cfg import (
  pochi_lp_acrl_flat_env_cfg,
  pochi_lp_acrl_rough_env_cfg,
)


def _identity_command(*, rough=True, play=False):
  factory = pochi_lp_acrl_rough_env_cfg if rough else pochi_lp_acrl_flat_env_cfg
  cfg = factory(play=play)
  term = LPACRLVelocityCommand.__new__(LPACRLVelocityCommand)
  term._env = SimpleNamespace(cfg=cfg)
  term.cfg = cfg.commands["base_velocity"]
  term.catalog = ROUGH_TASK_CATALOG if rough else FLAT_TASK_CATALOG
  term.curriculum = LPACRLCurriculum(term.catalog.num_tasks)
  return term


@pytest.mark.parametrize("rough", [False, True])
def test_play_identity_allows_noise_disabled_and_different_batch_size(rough):
  train = _identity_command(rough=rough)
  play = _identity_command(rough=rough, play=True)
  assert train.checkpoint_identity() == play.checkpoint_identity()
  json.dumps(train.curriculum_state_dict()["task_space"], allow_nan=False)


@pytest.mark.parametrize("setting", ["reward", "pd", "terrain", "observations"])
def test_physical_changes_reject_curriculum_resume(setting):
  original = _identity_command()
  changed = _identity_command()
  cfg = changed._env.cfg
  if setting == "reward":
    cfg.rewards["track_lin_vel_xy_exp"].weight += .1
  elif setting == "pd":
    articulation = cfg.scene.entities["robot"].articulation
    articulation.actuators = tuple(
      replace(actuator, stiffness=150.0) for actuator in articulation.actuators
    )
  elif setting == "terrain":
    generator = cfg.scene.terrain.terrain_generator
    stairs = generator.sub_terrains["ascending_stairs"].terrain
    stairs.step_height_range = (0.0, 0.09)
  else:
    group = cfg.observations["actor"]
    group.terms = dict(reversed(list(group.terms.items())))
  with pytest.raises(ValueError, match="task space"):
    changed.load_curriculum_state_dict(original.curriculum_state_dict())


def test_heldout_seed_is_compatible_for_actor_but_rejected_for_training_resume():
  original = _identity_command()
  heldout = _identity_command(play=True)
  heldout._env.cfg.scene.terrain.terrain_generator.seed = 4242
  assert original.checkpoint_identity() == heldout.checkpoint_identity()
  with pytest.raises(ValueError, match="terrain seed"):
    heldout.load_curriculum_state_dict(original.curriculum_state_dict())


def test_unchanged_checkpoint_resumes_curriculum():
  original = _identity_command()
  resumed = _identity_command()
  original.curriculum.advance(7)
  resumed.load_curriculum_state_dict(original.curriculum_state_dict())
  assert resumed.curriculum.steps_in_stage == 7
