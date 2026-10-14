"""Explicit full-state continuation with wider moving exploration, unchanged mean.

V10 preserves V8/model300's learned networks, Adam and LP state. Only the fixed
moving Gaussian standard deviation changes from .03 to .15. Unsaved simulator
and RNG histories are restarted, so this is not a paired-trajectory continuation.
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import torch

from pochi_rl.mjlab.lp_acrl_annealed import _full_load, _validate_optimizer
from pochi_rl.mjlab.lp_acrl_contact_runner import LPACRLContactOnPolicyRunner
from pochi_rl.mjlab.lp_acrl_low_noise import (
  CORRUPTION_IDENTITY,
  expected_symmetry_identity,
  validate_low_noise_checkpoint,
)
from pochi_rl.mjlab.lp_acrl_low_noise import (
  expected_exploration_identity as low_noise_identity,
)
from pochi_rl.mjlab.lp_acrl_refinement import _assert_fresh_training_state
from pochi_rl.mjlab.lp_acrl_resume import _assert_finite_tree, _equal_tree

MOVING_STD = 0.15
SOURCE_TASK = "Pochi-LP-ACRL-Blind-v8"
SOURCE_ITERATION = 300
STATE_KEY = "lp_acrl_wider_exploration_state"
STD_PARAMETER = "distribution.std_param"


def expected_exploration_identity() -> dict:
  identity = low_noise_identity()
  identity["moving_std_range"] = (MOVING_STD, MOVING_STD)
  return identity


def _validate_std(value) -> None:
  if (
    not isinstance(value, torch.Tensor)
    or value.shape != (12,)
    or not value.is_floating_point()
    or not torch.isfinite(value).all()
    or not torch.equal(value, torch.full_like(value, MOVING_STD))
  ):
    raise ValueError("Wider exploration requires fixed 12D moving std 0.15")


def _learning_rate(saved: dict) -> float:
  groups = (saved.get("optimizer_state_dict") or {}).get("param_groups")
  if not isinstance(groups, list) or not groups:
    raise ValueError("Continuation requires saved optimizer parameter groups")
  rates = [group.get("lr") for group in groups if isinstance(group, dict)]
  if (
    len(rates) != len(groups)
    or any(
      type(rate) not in (int, float) or not math.isfinite(rate) or rate <= 0
      for rate in rates
    )
    or len(set(rates)) != 1
  ):
    raise ValueError("Continuation requires one finite positive optimizer LR")
  return float(rates[0])


def _environment_counter(saved: dict) -> int:
  state = (saved.get("infos") or {}).get("env_state")
  counter = state.get("common_step_counter") if isinstance(state, dict) else None
  if type(counter) is not int or counter < 0:
    raise ValueError("Continuation requires the saved environment step counter")
  return counter


def validate_wider_exploration_checkpoint(saved: dict) -> int:
  """Validate portable exploration/count/LR contracts before constructing an env."""
  info = saved.get("infos") or {}
  if info.get("lp_acrl_observation_corruption") != CORRUPTION_IDENTITY:
    raise ValueError("Checkpoint observation corruption differs")
  if info.get("lp_acrl_training_exploration") != expected_exploration_identity():
    raise ValueError("Checkpoint wider exploration identity differs")
  if info.get("lp_acrl_training_symmetry") != expected_symmetry_identity():
    raise ValueError("Checkpoint symmetry differs")
  _validate_std((saved.get("actor_state_dict") or {}).get(STD_PARAMETER))
  state = info.get(STATE_KEY)
  if (
    not isinstance(state, dict)
    or set(state) != {"version", "completed_updates", "learning_rate"}
    or type(state["version"]) is not int
    or state["version"] != 1
  ):
    raise ValueError("Missing or malformed wider exploration continuation state")
  count = state["completed_updates"]
  iteration = saved.get("iter")
  if (
    type(count) is not int
    or count < SOURCE_ITERATION + 1
    or type(iteration) is not int
    or iteration != count - 1
  ):
    raise ValueError("Continuation count differs from post-update iteration")
  rate = state["learning_rate"]
  if type(rate) not in (int, float) or rate != _learning_rate(saved):
    raise ValueError("Saved adaptive LR differs from optimizer groups")
  _environment_counter(saved)
  return count


def _preflight_models(runner, saved: dict, *, full: bool) -> None:
  models = [("actor_state_dict", runner.alg._raw_actor)]
  if full:
    models.append(("critic_state_dict", runner.alg._raw_critic))
  for key, model in models:
    source, target = saved.get(key), model.state_dict()
    if not isinstance(source, dict) or source.keys() != target.keys():
      raise ValueError(f"Checkpoint model keys differ: {key}")
    for name, value in source.items():
      if (
        not isinstance(value, torch.Tensor)
        or value.shape != target[name].shape
        or value.dtype != target[name].dtype
        or (value.is_floating_point() and not torch.isfinite(value).all())
      ):
        raise ValueError(f"Invalid checkpoint model tensor: {key}/{name}")
  if full:
    _validate_optimizer(runner.alg.optimizer, saved.get("optimizer_state_dict"))


def _fresh_reset(runner, expected_lp: dict) -> None:
  seed = runner.cfg.get("seed")
  if type(seed) is not int:
    raise ValueError("Continuation needs an explicit fresh-simulator seed")
  counter = runner.env.unwrapped.common_step_counter
  runner.env.seed(seed)
  runner.env.reset()
  if runner.env.unwrapped.common_step_counter != counter:
    raise ValueError("Fresh reset changed restored environment step counter")
  if not _equal_tree(expected_lp, runner.command_term.curriculum_state_dict()):
    raise ValueError("Fresh reset changed restored LP statistics")
  if runner.env.episode_length_buf.any():
    raise ValueError("Fresh simulator episodes must start at length zero")


class LPACRLWiderExplorationOnPolicyRunner(LPACRLContactOnPolicyRunner):
  """Strict fixed-noise checkpoints with correct next-update resume semantics."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._training_ready = False
    self._in_learning = False
    self._failed_learning = False
    self.validate_wider_configuration()

  def observation_corruption_identity(self) -> dict:
    groups = self.env.unwrapped.cfg.observations
    value = {
      "version": 1,
      "groups": {name: bool(group.enable_corruption) for name, group in groups.items()},
    }
    if value != CORRUPTION_IDENTITY:
      raise ValueError("Wider exploration requires clean actor and critic observations")
    return value

  def validate_wider_configuration(self) -> None:
    self.observation_corruption_identity()
    if self.exploration_identity() != expected_exploration_identity():
      raise ValueError("Wider exploration requires fixed moving .15/stopped .01")
    if self.symmetry_identity() != expected_symmetry_identity():
      raise ValueError("Wider exploration requires mirror loss without augmentation")
    _validate_std(self.alg._raw_actor.distribution.std_param)

  def completed_updates(self) -> int:
    return self.current_learning_iteration + int(self._in_learning)

  def _require_training(self) -> None:
    if not self._training_ready or self._failed_learning:
      raise ValueError(
        "Uninitialized, inference-only or failed runner cannot save/train"
      )
    self.validate_wider_configuration()

  def save(self, path: str, infos=None) -> None:
    self._require_training()
    count = self.completed_updates()
    if count < SOURCE_ITERATION + 1:
      raise ValueError("Continuation cannot precede its source checkpoint")
    rates = [group["lr"] for group in self.alg.optimizer.param_groups]
    rate = self.alg.learning_rate
    if (
      type(rate) not in (int, float)
      or not math.isfinite(rate)
      or rate <= 0
      or not rates
      or any(group_rate != rate for group_rate in rates)
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
            "learning_rate": self.alg.learning_rate,
          },
        },
      )
    finally:
      self.current_learning_iteration = cursor

  def load(self, path, load_cfg=None, strict=True, map_location=None) -> dict:
    self.validate_wider_configuration()
    saved = torch.load(Path(path), map_location="cpu", weights_only=False)
    count = validate_wider_exploration_checkpoint(saved)
    full = _full_load(load_cfg)
    if full and type(self.cfg.get("seed")) is not int:
      raise ValueError("Continuation needs an explicit fresh-simulator seed")
    _preflight_models(self, saved, full=full)
    self._training_ready = False
    try:
      result = super().load(path, load_cfg, strict, map_location)
      if full:
        self.alg.learning_rate = _learning_rate(saved)
        self.current_learning_iteration = count
        _fresh_reset(self, saved["infos"]["lp_acrl"])
      self.validate_wider_configuration()
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


def initialize_wider_exploration(runner, saved: dict, parent: dict) -> dict:
  """One explicit V8/model300 migration; caller additionally pins source bytes."""
  runner.validate_wider_configuration()
  _assert_fresh_training_state(runner)
  if (
    parent.get("task") != SOURCE_TASK
    or type(saved.get("iter")) is not int
    or saved["iter"] != SOURCE_ITERATION
  ):
    raise ValueError("Wider exploration requires Blind-v8 model300")
  if type(runner.cfg.get("seed")) is not int or runner.cfg["seed"] != parent.get(
    "seed"
  ):
    raise ValueError("Continuation seed must match its source run")
  validate_low_noise_checkpoint(saved)
  info = saved.get("infos") or {}
  term = runner.command_term
  if info.get("lp_acrl_checkpoint_version") != 1:
    raise ValueError("Source must be an independent LP-ACRL checkpoint")
  lp = info.get("lp_acrl") or {}
  if lp.get("task_space") != term.checkpoint_identity():
    raise ValueError("Continuation environment, rewards and task space must match")
  backend = runner.contact_backend_identity()
  if not backend or any(
    value != backend
    for value in (info.get("lp_acrl_contact_backend"), parent.get("contact_backend"))
  ):
    raise ValueError("Source, parent and target contact backend must match")
  _preflight_models(runner, saved, full=True)
  _assert_finite_tree(lp)
  rate = _learning_rate(saved)
  counter = _environment_counter(saved)
  old_lp = deepcopy(term.curriculum_state_dict())
  runner._training_ready = False
  try:
    # The LP loader validates the full partial stage before network mutation.
    term.load_curriculum_state_dict(lp)
    if not runner.alg.load(saved, None, strict=True):
      raise ValueError("Full PPO state loader did not request iteration restoration")
    for key, model in (
      ("actor_state_dict", runner.alg._raw_actor),
      ("critic_state_dict", runner.alg._raw_critic),
    ):
      if not _equal_tree(saved[key], model.state_dict()):
        raise ValueError(f"Full transfer changed source network state: {key}")
    if not _equal_tree(
      saved["optimizer_state_dict"], runner.alg.optimizer.state_dict()
    ):
      raise ValueError("Full transfer changed source optimizer state")
    if not _equal_tree(lp, term.curriculum_state_dict()):
      raise ValueError("Full transfer changed source LP state")
    runner.alg.learning_rate = rate
    runner.current_learning_iteration = SOURCE_ITERATION + 1
    runner.env.unwrapped.common_step_counter = counter
    runner.alg._raw_actor.distribution.std_param.fill_(MOVING_STD)
    _fresh_reset(runner, lp)
    runner.validate_wider_configuration()
  except BaseException:
    term.load_curriculum_state_dict(old_lp)
    runner._failed_learning = True
    raise
  runner._training_ready = True
  runner._failed_learning = False
  return {
    "source_task": SOURCE_TASK,
    "source_post_update_iteration": SOURCE_ITERATION,
    "completed_updates": SOURCE_ITERATION + 1,
    "next_iteration": SOURCE_ITERATION + 1,
    "loaded": ["actor", "critic", "normalizers", "optimizer", "LP", "iteration"],
    "network_and_optimizer_exact_before_std_change": True,
    "changed_actor_tensor": {STD_PARAMETER: {"from": 0.03, "to": MOVING_STD}},
    "adaptive_learning_rate_restored": rate,
    "environment_step_counter_restored": counter,
    "lp_statistics_preserved": True,
    "fresh_simulator_reset": True,
    "fresh_reset_seed": runner.cfg["seed"],
    "simulator_state_and_rng_restored": False,
    "unsaved_partial_episodes": "discarded; reset adds no completed LP samples",
    "mean_inference": "unchanged; no added noise or action replacement",
  }
