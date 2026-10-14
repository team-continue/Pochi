"""Transfer the no-touchdown flat controller to original600 blind rough tasks.

All22 rewards,48D observations, physical assets and exploration are retained.
Actor/critic/normalizers transfer exactly; Adam/LP begin fresh for the new tasks.
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import torch

from pochi_rl.mjlab.lp_acrl_annealed import _full_load
from pochi_rl.mjlab.lp_acrl_contact_runner import LPACRLContactOnPolicyRunner
from pochi_rl.mjlab.lp_acrl_flat_focus_identity import (
  FLAT_FOCUS_TRAINING_PATHS,
)
from pochi_rl.mjlab.lp_acrl_no_touchdown_refinement import (
  STATE_KEY as SOURCE_STATE_KEY,
)
from pochi_rl.mjlab.lp_acrl_no_touchdown_refinement import (
  expected_no_touchdown_rough_task_space as expected_rough_task_space,
)
from pochi_rl.mjlab.lp_acrl_no_touchdown_refinement import (
  expected_no_touchdown_task_space as expected_source_task_space,
)
from pochi_rl.mjlab.lp_acrl_no_touchdown_refinement import (
  validate_no_touchdown_checkpoint as validate_source_checkpoint,
)
from pochi_rl.mjlab.lp_acrl_refinement import (
  MODEL_ONLY_LOAD,
  _assert_fresh_training_state,
)
from pochi_rl.mjlab.lp_acrl_resume import _equal_tree
from pochi_rl.mjlab.lp_acrl_stride_center_refinement import (
  STATE_KEY as STRIDE_STATE_KEY,
)
from pochi_rl.mjlab.lp_acrl_wider_exploration import (
  CORRUPTION_IDENTITY,
  _environment_counter,
  _fresh_reset,
  _learning_rate,
  _preflight_models,
  _validate_std,
  expected_exploration_identity,
  expected_symmetry_identity,
)
from pochi_rl.mjlab.lp_acrl_wider_exploration import (
  STATE_KEY as WIDER_STATE_KEY,
)

LEGACY_STATE_KEYS = (
  WIDER_STATE_KEY, STRIDE_STATE_KEY,
  "lp_acrl_yaw_stability_state", "lp_acrl_flat_focus_state",
  "lp_acrl_duty_balance_state", SOURCE_STATE_KEY,
)

SOURCE_TASK = "Pochi-LP-ACRL-Blind-v15"
SOURCE_ITERATION = 600
STATE_KEY = "lp_acrl_rough_no_touchdown_state"
INITIALIZATION = "model_only_fresh_optimizer_rough_lp"
REMOVED_REWARD = "balanced_touchdown"


def validate_rough_no_touchdown_checkpoint(saved: dict) -> int:
  """Validate the portable fresh-refinement contract before creating an env."""
  info = saved.get("infos") or {}
  if any(key in info or key in saved for key in LEGACY_STATE_KEYS):
    raise ValueError("No-touchdown refinement cannot contain prior refinement progress")
  if STATE_KEY in saved:
    raise ValueError("No-touchdown progress must occur only in checkpoint infos")
  if info.get("lp_acrl_observation_corruption") != CORRUPTION_IDENTITY:
    raise ValueError("Checkpoint observation corruption differs")
  if info.get("lp_acrl_training_exploration") != expected_exploration_identity():
    raise ValueError("Checkpoint fixed exploration differs")
  if info.get("lp_acrl_training_symmetry") != expected_symmetry_identity():
    raise ValueError("Checkpoint symmetry differs")
  _validate_std((saved.get("actor_state_dict") or {}).get("distribution.std_param"))
  state = info.get(STATE_KEY)
  if (
    not isinstance(state, dict)
    or set(state) != {"version", "completed_updates", "learning_rate", "initialization"}
    or type(state["version"]) is not int
    or state["version"] != 1
  ):
    raise ValueError("Missing or malformed no-touchdown progress state")
  if state["initialization"] != INITIALIZATION:
    raise ValueError("Checkpoint initialization contract differs")
  count, iteration = state["completed_updates"], saved.get("iter")
  if (
    type(count) is not int
    or count < 0
    or type(iteration) is not int
    or iteration != count - 1
  ):
    raise ValueError("Local completed count differs from post-update iteration")
  rate = state["learning_rate"]
  if type(rate) not in (int, float) or rate != _learning_rate(saved):
    raise ValueError("Saved adaptive LR differs from optimizer groups")
  _environment_counter(saved)
  lp = info.get("lp_acrl") or {}
  if (
    lp.get("task_space") != expected_rough_task_space()
    or "training_terrain_seed" not in lp
    or type(lp["training_terrain_seed"]) is not int
    or lp["training_terrain_seed"] != 42
  ):
    raise ValueError(
      "Checkpoint requires exact V16 rough22 identity and terrain seed42"
    )
  return count


class LPACRLRoughNoTouchdownOnPolicyRunner(LPACRLContactOnPolicyRunner):
  """Independent local updates, including unambiguous initial count0/iter−1."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._training_ready = False
    self._in_learning = False
    self._failed_learning = False
    self.validate_rough_no_touchdown_configuration()

  def observation_corruption_identity(self) -> dict:
    groups = self.env.unwrapped.cfg.observations
    value = {
      "version": 1,
      "groups": {name: bool(group.enable_corruption) for name, group in groups.items()},
    }
    if value != CORRUPTION_IDENTITY:
      raise ValueError(
        "No-touchdown refinement requires clean actor/critic observations"
      )
    return value

  def validate_rough_no_touchdown_configuration(self) -> None:
    self.observation_corruption_identity()
    if self.exploration_identity() != expected_exploration_identity():
      raise ValueError("No-touchdown refinement requires fixed moving .15/stopped .01")
    if self.symmetry_identity() != expected_symmetry_identity():
      raise ValueError(
        "No-touchdown refinement requires mirror loss without augmentation"
      )
    _validate_std(self.alg._raw_actor.distribution.std_param)

  def completed_updates(self) -> int:
    return self.current_learning_iteration + int(self._in_learning)

  def _require_training(self) -> None:
    if not self._training_ready or self._failed_learning:
      raise ValueError(
        "Uninitialized, inference-only or failed runner cannot save/train"
      )
    self.validate_rough_no_touchdown_configuration()

  def save(self, path: str, infos=None) -> None:
    self._require_training()
    if any(key in (infos or {}) for key in LEGACY_STATE_KEYS):
      raise ValueError(
        "Prior progress must not enter a fresh no-touchdown refinement checkpoint"
      )
    count = self.completed_updates()
    if type(count) is not int or count < 0:
      raise ValueError("Local completed count must be nonnegative")
    rate = self.alg.learning_rate
    groups = self.alg.optimizer.param_groups
    if (
      type(rate) not in (int, float)
      or not math.isfinite(rate)
      or rate <= 0
      or not groups
      or any(group["lr"] != rate for group in groups)
    ):
      raise ValueError("Adaptive LR differs from optimizer groups")
    cursor = self.current_learning_iteration
    self.current_learning_iteration = count - 1
    try:
      super().save(
        path,
        {
          **(infos or {}),
          "lp_acrl_observation_corruption": self.observation_corruption_identity(),
          STATE_KEY: {
            "version": 1,
            "completed_updates": count,
            "learning_rate": rate,
            "initialization": INITIALIZATION,
          },
        },
      )
    finally:
      self.current_learning_iteration = cursor

  def load(self, path, load_cfg=None, strict=True, map_location=None) -> dict:
    self.validate_rough_no_touchdown_configuration()
    saved = torch.load(Path(path), map_location="cpu", weights_only=False)
    count = validate_rough_no_touchdown_checkpoint(saved)
    full = _full_load(load_cfg)
    if full and type(self.cfg.get("seed")) is not int:
      raise ValueError("Resume needs an explicit fresh-simulator seed")
    _preflight_models(self, saved, full=full)
    self._training_ready = False
    try:
      result = super().load(path, load_cfg, strict, map_location)
      if full:
        self.alg.learning_rate = _learning_rate(saved)
        self.current_learning_iteration = count
        _fresh_reset(self, saved["infos"]["lp_acrl"])
      self.validate_rough_no_touchdown_configuration()
    except BaseException:
      self._failed_learning = True
      raise
    self._training_ready = full
    self._failed_learning = False
    return result

  def learn(self, num_learning_iterations: int, init_at_random_ep_len=False) -> None:
    self._require_training()
    if type(num_learning_iterations) is not int or num_learning_iterations < 1:
      raise ValueError("Require a positive number of additional PPO updates")
    start = self.current_learning_iteration
    self._in_learning = True
    try:
      super().learn(num_learning_iterations, init_at_random_ep_len=False)
    except BaseException:
      self._failed_learning = True
      self._training_ready = False
      raise
    else:
      self.current_learning_iteration = start + num_learning_iterations
    finally:
      self._in_learning = False


def validate_rough_refinement_identity(runner, saved: dict, parent: dict) -> None:
  if parent.get("task") != SOURCE_TASK or saved.get("iter") != SOURCE_ITERATION:
    raise ValueError("Rough refinement requires pinned V15/model600")
  if validate_source_checkpoint(saved) != SOURCE_ITERATION + 1:
    raise ValueError("Source update count differs")
  info = saved["infos"]
  if info.get("lp_acrl_checkpoint_version") != 1 or STATE_KEY in info:
    raise ValueError("Source must be an independent V15 checkpoint")
  if runner.cfg.get("seed") != parent.get("seed") or type(
    runner.cfg.get("seed")
  ) is not int:
    raise ValueError("Fresh rough reset seed must match parent")
  rate = _learning_rate(saved)
  if runner.alg.learning_rate != rate or any(
    group["lr"] != rate for group in runner.alg.optimizer.param_groups
  ):
    raise ValueError("Fresh Adam must start at actual source adaptive LR")
  if info["lp_acrl"]["task_space"] != expected_source_task_space():
    raise ValueError("Require complete canonical V15 plane22 identity")
  term = runner.command_term
  if term.checkpoint_identity() != expected_rough_task_space():
    raise ValueError("Require complete canonical original rough22 identity")
  if term.curriculum.num_tasks != 600:
    raise ValueError("Require original600 rough tasks")
  generator = term._env.cfg.scene.terrain.terrain_generator
  if generator is None or generator.seed != 42:
    raise ValueError("Require original rough terrain generator seed42")
  if info["lp_acrl"]["training_terrain_seed"] is not None:
    raise ValueError("Source must be true plane without generator")
  runner.validate_rough_no_touchdown_configuration()
  backend = runner.contact_backend_identity()
  if not backend or any(value != backend for value in (
    info.get("lp_acrl_contact_backend"), parent.get("contact_backend"),
  )):
    raise ValueError("Source/parent/target contact backend must match")


def initialize_rough_no_touchdown_refinement(runner, saved: dict, parent: dict) -> dict:
  """Transfer models exactly, with independent fresh Adam, LP and simulator state."""
  validate_rough_refinement_identity(runner, saved, parent)
  _assert_fresh_training_state(runner)
  if runner.env.unwrapped.common_step_counter != 0:
    raise ValueError("New reward refinement requires a fresh environment counter")
  _preflight_models(runner, saved, full=True)
  fresh_lp = deepcopy(runner.command_term.curriculum_state_dict())
  runner._training_ready = False
  try:
    if runner.alg.load(saved, dict(MODEL_ONLY_LOAD), strict=True):
      raise ValueError("Model-only refinement unexpectedly restored iteration")
    for key, model in (
      ("actor_state_dict", runner.alg._raw_actor),
      ("critic_state_dict", runner.alg._raw_critic),
    ):
      if not _equal_tree(saved[key], model.state_dict()):
        raise ValueError(f"Model-only transfer changed source state: {key}")
    _assert_fresh_training_state(runner)
    _fresh_reset(runner, fresh_lp)
    runner.validate_rough_no_touchdown_configuration()
  except BaseException:
    runner._failed_learning = True
    raise
  runner._training_ready = True
  runner._failed_learning = False
  return {
    "source_task": SOURCE_TASK,
    "source_iteration": SOURCE_ITERATION,
    "source_completed_updates": SOURCE_ITERATION + 1,
    "loaded": ["actor", "critic", "actor_normalizer", "critic_normalizer"],
    "not_loaded": ["optimizer", "LP_statistics", "iteration", "RND", "env_state"],
    "all_actor_and_critic_states_bitwise_preserved": True,
    "reset_actor_state": {},
    "initial_iteration": 0,
    "initial_completed_updates": 0,
    "initial_optimizer_state_entries": 0,
    "initial_learning_rate": runner.alg.learning_rate,
    "initial_lp_stage": 0,
    "initial_lp_steps": 0,
    "initial_lp_episode_count": 0,
    "initial_lp_num_tasks": 600,
    "initial_environment_step_counter": 0,
    "fresh_simulator_reset": True,
    "fresh_reset_seed": runner.cfg["seed"],
    "initial_exploration_identity": runner.exploration_identity(),
    "observation_corruption_unchanged": runner.observation_corruption_identity(),
    "symmetry_unchanged": runner.symmetry_identity(),
    "contact_backend_unchanged": runner.contact_backend_identity(),
    "allowed_environment_identity_difference": list(FLAT_FOCUS_TRAINING_PATHS),
    "all_22_rewards_unchanged": True,
    "target_training_terrain_seed": 42,
  }
