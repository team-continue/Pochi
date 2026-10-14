"""Strict v4-to-v5 transfer; only the declared exploration and reward changes."""

from __future__ import annotations

import hashlib
from copy import deepcopy
from pathlib import Path

import torch
import yaml

from pochi_rl.mjlab.lp_acrl_low_noise import expected_exploration_identity
from pochi_rl.mjlab.lp_acrl_refinement import (
  MODEL_ONLY_LOAD,
  _assert_fresh_training_state,
  without_rewards,
)

SOURCE_TASK = "Pochi-LP-ACRL-Blind-v4"
SOURCE_OBSERVATION_CORRUPTION = {
  "version": 1,
  "groups": {"actor": True, "critic": False},
}
TARGET_OBSERVATION_CORRUPTION = {
  "version": 1,
  "groups": {"actor": False, "critic": False},
}


def read_parent_observation_corruption(path: Path) -> dict:
  """Read only boolean YAML nodes, never instantiate its serialized Python tags."""
  content = path.read_bytes()
  root = yaml.compose(content, Loader=yaml.SafeLoader)

  def lookup(node, key):
    if not isinstance(node, yaml.MappingNode):
      raise ValueError(f"Parent environment YAML lacks mapping for {key}")
    found = [value for name, value in node.value if name.value == key]
    if len(found) != 1:
      raise ValueError(f"Parent environment YAML requires exactly one {key}")
    return found[0]

  observations = lookup(root, "observations")
  groups = {}
  for group in ("actor", "critic"):
    value = lookup(lookup(observations, group), "enable_corruption")
    if (
      not isinstance(value, yaml.ScalarNode)
      or value.tag != "tag:yaml.org,2002:bool"
      or value.value not in ("true", "false")
    ):
      raise ValueError("Parent corruption flags must be explicit YAML booleans")
    groups[group] = value.value == "true"
  identity = {"version": 1, "groups": groups}
  if identity != SOURCE_OBSERVATION_CORRUPTION:
    raise ValueError("Parent must use actor corruption and an uncorrupted critic")
  return {
    "path": str(path.resolve()),
    "sha256": hashlib.sha256(content).hexdigest(),
    "identity": identity,
  }


def validate_low_noise_identity(runner, saved: dict, parent: dict, source_noise: dict):
  if parent.get("task") != SOURCE_TASK:
    raise ValueError("Low-noise refinement requires a Blind-v4 parent")
  if (
    parent.get("learning_rate") != 1e-3
    or runner.alg.learning_rate != 1e-4
    or any(group["lr"] != 1e-4 for group in runner.alg.optimizer.param_groups)
  ):
    raise ValueError("Require the explicit initial learning-rate change .001 to .0001")
  infos = saved.get("infos") or {}
  if infos.get("lp_acrl_checkpoint_version") != 1:
    raise ValueError("Expected an independent LP-ACRL checkpoint")
  term = runner.command_term
  current = term.checkpoint_identity()
  source = (infos.get("lp_acrl") or {}).get("task_space")
  if without_rewards(source) != without_rewards(current):
    raise ValueError("Low-noise environment differs outside rewards")
  if not source.get("command_protocol"):
    raise ValueError("Source must retain the blind stop/start command protocol")
  old_rewards = source["environment"]["rewards"]
  expected_rewards = deepcopy(old_rewards)
  try:
    if (
      old_rewards["track_ang_vel_z_exp"]["params"]["std"] != 0.35
      or old_rewards["standing_posture"]["weight"] != 0.5
      or old_rewards["standing_posture"]["params"].get("std", 0.20) != 0.20
      or old_rewards["standing_missing_contacts"]["weight"] != -1
      or old_rewards["moving_qualified_swing_occupancy"]["weight"] != -8
    ):
      raise ValueError("Unexpected v4 source reward configuration")
    expected_rewards["track_ang_vel_z_exp"]["params"]["std"] = 0.15
    expected_rewards["standing_posture"]["weight"] = 1.5
    expected_rewards["standing_posture"]["params"]["std"] = 0.15
    expected_rewards["standing_missing_contacts"]["weight"] = -2.0
  except KeyError as exc:
    raise ValueError("Source lacks required v4 rewards") from exc
  if current["environment"]["rewards"] != expected_rewards:
    raise ValueError("Only the three approved v5 reward entries may change")
  terrain = term._env.cfg.scene.terrain
  generator = terrain.terrain_generator if terrain is not None else None
  terrain_seed = None if generator is None else generator.seed
  if infos["lp_acrl"].get("training_terrain_seed") != terrain_seed:
    raise ValueError("Terrain seed must remain unchanged")
  symmetry = runner.symmetry_identity()
  expected_source_symmetry = {**symmetry, "use_data_augmentation": True}
  if (
    symmetry.get("use_data_augmentation") is not False
    or symmetry.get("use_mirror_loss") is not True
    or symmetry.get("mirror_loss_coeff") != 1.0
    or infos.get("lp_acrl_training_symmetry") != expected_source_symmetry
  ):
    raise ValueError("Only disable symmetry data augmentation; keep mirror loss exact")
  backend = runner.contact_backend_identity()
  if (
    not backend
    or infos.get("lp_acrl_contact_backend") != backend
    or parent.get("contact_backend") != backend
  ):
    raise ValueError(
      "Parent checkpoint, manifest and target backend must match exactly"
    )
  if source_noise != SOURCE_OBSERVATION_CORRUPTION:
    raise ValueError("Verified parent observation corruption is required")
  recorded_noise = infos.get("lp_acrl_observation_corruption")
  if recorded_noise is not None and recorded_noise != source_noise:
    raise ValueError("Source checkpoint corruption metadata contradicts parent YAML")
  if runner.observation_corruption_identity() != TARGET_OBSERVATION_CORRUPTION:
    raise ValueError("Target actor and critic must disable observation corruption")
  exploration = runner.exploration_identity()
  if exploration != expected_exploration_identity():
    raise ValueError(
      "Target requires unchanged quiet-stop actor with fixed moving std .03"
    )
  expected_source = {
    **exploration,
    "moving_std_range": (0.1, 1e6),
    "moving_std_learned": True,
  }
  if infos.get("lp_acrl_training_exploration") != expected_source:
    raise ValueError("Source exploration differs beyond the approved moving std change")
  return exploration


def initialize_low_noise_refinement(
  runner, saved: dict, parent: dict, source_noise: dict
):
  """Transfer both complete networks, resetting only the nonlearned action std."""
  exploration = validate_low_noise_identity(runner, saved, parent, source_noise)
  _assert_fresh_training_state(runner)
  for key, model in (
    ("actor_state_dict", runner.alg._raw_actor),
    ("critic_state_dict", runner.alg._raw_critic),
  ):
    source, target = saved.get(key), model.state_dict()
    if not isinstance(source, dict) or source.keys() != target.keys():
      raise ValueError(f"Model state keys differ: {key}")
    for name, value in source.items():
      if (
        not isinstance(value, torch.Tensor)
        or value.shape != target[name].shape
        or value.dtype != target[name].dtype
        or (value.is_floating_point() and not torch.isfinite(value).all())
      ):
        raise ValueError(f"Invalid model tensor: {key}/{name}")
  if runner.alg._raw_actor.distribution.std_param.requires_grad:
    raise ValueError("Moving std must be a non-trainable parameter before loading")
  restore_iteration = runner.alg.load(saved, dict(MODEL_ONLY_LOAD), strict=True)
  if restore_iteration:
    raise ValueError("Model-only refinement unexpectedly restored iteration")
  with torch.no_grad():
    runner.alg._raw_actor.distribution.std_param.fill_(0.03)
  for key, model in (
    ("actor_state_dict", runner.alg._raw_actor),
    ("critic_state_dict", runner.alg._raw_critic),
  ):
    for name, value in model.state_dict().items():
      expected = (
        torch.full_like(value, 0.03)
        if key == "actor_state_dict" and name == "distribution.std_param"
        else saved[key][name].to(value.device)
      )
      if not torch.equal(value, expected):
        raise ValueError(f"Model transfer changed an unexpected state: {key}/{name}")
  _assert_fresh_training_state(runner)
  return {
    "loaded": ["actor_mean", "critic", "actor_normalizer", "critic_normalizer"],
    "not_loaded": ["optimizer", "LP_statistics", "iteration", "RND"],
    "reset_actor_state": {"distribution.std_param": 0.03},
    "all_other_actor_and_critic_states_bitwise_preserved": True,
    "source_iteration": int(saved["iter"]),
    "initial_iteration": 0,
    "initial_lp_stage": 0,
    "initial_lp_steps": 0,
    "initial_lp_episode_count": 0,
    "initial_optimizer_state_entries": 0,
    "optimizer_initial_learning_rate_change": {"source": 1e-3, "target": 1e-4},
    "symmetry_transition": {
      "source": deepcopy(saved["infos"]["lp_acrl_training_symmetry"]),
      "target": runner.symmetry_identity(),
      "only_changed_field": "use_data_augmentation",
    },
    "initial_exploration_identity": exploration,
    "observation_corruption_transition": {
      "source": deepcopy(source_noise),
      "target": runner.observation_corruption_identity(),
    },
    "contact_backend_unchanged": runner.contact_backend_identity(),
    "allowed_environment_identity_difference": [
      "environment.rewards.track_ang_vel_z_exp.params.std",
      "environment.rewards.standing_posture.weight",
      "environment.rewards.standing_posture.params.std",
      "environment.rewards.standing_missing_contacts.weight",
    ],
  }
