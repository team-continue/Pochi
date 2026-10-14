"""Fresh blind training with exploration annealed only between PPO batches.

The native conditional Gaussian is unchanged throughout rollout and likelihood
updates. A successful PPO update advances the next rollout's standard deviation.
Deterministic inference remains the inherited mean, without action replacement.
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import torch
from rsl_rl.algorithms import PPO

from pochi_rl.mjlab.lp_acrl_contact_runner import LPACRLContactOnPolicyRunner
from pochi_rl.mjlab.lp_acrl_low_noise import (
  CORRUPTION_IDENTITY,
  expected_symmetry_identity,
)
from pochi_rl.mjlab.lp_acrl_low_noise import (
  expected_exploration_identity as fixed_exploration_identity,
)

INITIAL_STD = 0.35
FINAL_STD = 0.03
ANNEAL_UPDATES = 600
SCHEDULE_KEY = "lp_acrl_exploration_schedule"
PPO_STATE_KEY = "lp_acrl_annealed_ppo_state"


def schedule_identity() -> dict:
  return {
    "version": 1,
    "initial_moving_std": INITIAL_STD,
    "final_moving_std": FINAL_STD,
    "anneal_completed_updates": ANNEAL_UPDATES,
    "curve": "exponential_then_constant",
    "advance": "once after each successful complete PPO update",
    "held_fixed": "entire rollout and its PPO likelihood updates",
    "standing_std": 0.01,
  }


def moving_std_for_updates(completed_updates: int) -> float:
  if type(completed_updates) is not int or completed_updates < 0:
    raise ValueError("Completed update count must be a nonnegative integer")
  if completed_updates >= ANNEAL_UPDATES:
    return FINAL_STD
  return INITIAL_STD * (FINAL_STD / INITIAL_STD) ** (completed_updates / ANNEAL_UPDATES)


def expected_exploration_identity() -> dict:
  identity = fixed_exploration_identity()
  identity["moving_std_range"] = (FINAL_STD, INITIAL_STD)
  return identity


def _validate_std(value, expected: float) -> None:
  if (
    not isinstance(value, torch.Tensor)
    or value.shape != (12,)
    or not value.is_floating_point()
    or not torch.isfinite(value).all()
    or not torch.allclose(value, torch.full_like(value, expected), rtol=0, atol=1e-8)
  ):
    raise ValueError("Actor std parameter differs from the saved exploration schedule")


def _validate_schedule(saved: dict) -> int:
  state = saved.get(SCHEDULE_KEY)
  if not isinstance(state, dict) or set(state) != {
    "identity",
    "completed_updates",
    "next_moving_std",
    "training_resume_allowed",
  }:
    raise ValueError("Missing or malformed exploration schedule state")
  if state["identity"] != schedule_identity():
    raise ValueError("Exploration schedule identity differs")
  count = state["completed_updates"]
  expected = moving_std_for_updates(count)
  value = state["next_moving_std"]
  if (
    type(value) not in (int, float)
    or not math.isfinite(value)
    or abs(value - expected) > 1e-12
    or type(state["training_resume_allowed"]) is not bool
  ):
    raise ValueError("Exploration schedule count/value is inconsistent")
  _validate_std(
    (saved.get("actor_state_dict") or {}).get("distribution.std_param"), expected
  )
  algorithm_state = saved.get(PPO_STATE_KEY)
  if not isinstance(algorithm_state, dict) or set(algorithm_state) != {"learning_rate"}:
    raise ValueError("Missing annealed PPO learning-rate state")
  learning_rate = algorithm_state["learning_rate"]
  groups = (saved.get("optimizer_state_dict") or {}).get("param_groups")
  if (
    type(learning_rate) not in (int, float)
    or not math.isfinite(learning_rate)
    or learning_rate <= 0
    or not isinstance(groups, list)
    or not groups
    or any(
      not isinstance(group, dict) or group.get("lr") != learning_rate
      for group in groups
    )
  ):
    raise ValueError("Saved adaptive learning rate and optimizer differ")
  return count


def validate_annealed_checkpoint(saved: dict) -> int:
  """Validate the complete portable contract before environment creation/load."""
  count = _validate_schedule(saved)
  iteration = saved.get("iter")
  if type(iteration) is not int or iteration != max(0, count - 1):
    raise ValueError("Checkpoint iteration differs from completed PPO updates")
  metadata = saved.get("infos") or {}
  if metadata.get("lp_acrl_observation_corruption") != CORRUPTION_IDENTITY:
    raise ValueError("Checkpoint observation corruption differs or is missing")
  if metadata.get("lp_acrl_training_exploration") != expected_exploration_identity():
    raise ValueError("Checkpoint annealed exploration differs or is missing")
  if metadata.get("lp_acrl_training_symmetry") != expected_symmetry_identity():
    raise ValueError("Checkpoint symmetry differs or is missing")
  return count


def _full_load(load_cfg: dict | None) -> bool:
  if load_cfg is None:
    return True
  if all(
    load_cfg.get(key, False) for key in ("actor", "critic", "optimizer", "iteration")
  ):
    return True
  if load_cfg == {"actor": True}:
    return False
  raise ValueError("Annealed policy permits only full resume or actor-only inference")


def _validate_optimizer(optimizer, saved: dict) -> None:
  """Exercise Adam's loader on an isolated copy before changing live networks."""
  candidate = deepcopy(optimizer)
  try:
    candidate.load_state_dict(deepcopy(saved))
  except (KeyError, TypeError, ValueError, RuntimeError) as exc:
    raise ValueError("Checkpoint optimizer structure differs") from exc
  for parameter, state in candidate.state.items():
    if not isinstance(parameter, torch.Tensor) or not isinstance(state, dict):
      raise ValueError("Checkpoint optimizer contains unknown parameter state")
    if not state:
      continue
    if not {"step", "exp_avg", "exp_avg_sq"} <= state.keys():
      raise ValueError("Checkpoint Adam state is incomplete")
    for name in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
      if name not in state:
        continue
      value = state[name]
      if (
        not isinstance(value, torch.Tensor)
        or value.shape != parameter.shape
        or value.dtype != parameter.dtype
        or not torch.isfinite(value).all()
        or (name != "exp_avg" and (value < 0).any())
      ):
        raise ValueError(f"Checkpoint Adam moment is invalid: {name}")
    step = state["step"]
    if isinstance(step, torch.Tensor):
      if step.numel() != 1 or not torch.isfinite(step).all():
        raise ValueError("Checkpoint Adam step is invalid")
      step = step.item()
    if type(step) not in (int, float) or not math.isfinite(step) or step < 0:
      raise ValueError("Checkpoint Adam step is invalid")


class LPACRLAnnealedPPO(PPO):
  """Own the successful-update counter and change variance between batches only."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.completed_updates = 0
    self.training_resume_allowed = True
    self._failed_update = False
    self.validate_schedule_configuration()

  def validate_schedule_configuration(self) -> None:
    actor = self._raw_actor
    if actor.exploration_identity() != expected_exploration_identity():
      raise ValueError("Annealed PPO requires the declared fixed-parameter Gaussian")
    _validate_std(
      actor.distribution.std_param, moving_std_for_updates(self.completed_updates)
    )

  def schedule_state(self) -> dict:
    self.validate_schedule_configuration()
    return {
      "identity": schedule_identity(),
      "completed_updates": self.completed_updates,
      "next_moving_std": moving_std_for_updates(self.completed_updates),
      "training_resume_allowed": self.training_resume_allowed,
    }

  def update(self) -> dict[str, float]:
    if not self.training_resume_allowed:
      raise ValueError("Actor-only checkpoints are for inference, not training resume")
    self.validate_schedule_configuration()
    # No schedule mutation before/during super().update(); a failed update does
    # not advance this counter. Its optimizer may already be partially changed,
    # so exceptions propagate and the caller must stop that failed training run.
    try:
      result = super().update()
    except BaseException:
      self._failed_update = True
      self.training_resume_allowed = False
      raise
    self.completed_updates += 1
    with torch.no_grad():
      self._raw_actor.distribution.std_param.fill_(
        moving_std_for_updates(self.completed_updates)
      )
    result["exploration/next_moving_std"] = moving_std_for_updates(
      self.completed_updates
    )
    return result

  def save(self) -> dict:
    if self._failed_update:
      raise ValueError("A failed partial PPO update cannot become a checkpoint")
    return {
      **super().save(),
      SCHEDULE_KEY: self.schedule_state(),
      PPO_STATE_KEY: {"learning_rate": self.learning_rate},
    }

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    count = _validate_schedule(loaded_dict)
    full = _full_load(load_cfg)
    if full and not loaded_dict[SCHEDULE_KEY]["training_resume_allowed"]:
      raise ValueError("Cannot resume an actor-only inference checkpoint")
    if full:
      _validate_optimizer(self.optimizer, loaded_dict.get("optimizer_state_dict"))
    for key, model in (
      ("actor_state_dict", self._raw_actor),
      ("critic_state_dict", self._raw_critic),
    ):
      if not full and key == "critic_state_dict":
        continue
      state, target = loaded_dict.get(key), model.state_dict()
      if not isinstance(state, dict) or state.keys() != target.keys():
        raise ValueError(f"Checkpoint model keys differ: {key}")
      for name, value in state.items():
        if (
          not isinstance(value, torch.Tensor)
          or value.shape != target[name].shape
          or value.dtype != target[name].dtype
          or (value.is_floating_point() and not torch.isfinite(value).all())
        ):
          raise ValueError(f"Invalid checkpoint model tensor: {key}/{name}")
    restored_iteration = super().load(loaded_dict, load_cfg, strict)
    self.completed_updates = count
    self.training_resume_allowed = full
    self._failed_update = False
    if full:
      self.learning_rate = loaded_dict[PPO_STATE_KEY]["learning_rate"]
    self.validate_schedule_configuration()
    return restored_iteration


class LPACRLAnnealedOnPolicyRunner(LPACRLContactOnPolicyRunner):
  """Keep the observation/backend contracts and resume at the next update."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if type(self.alg) is not LPACRLAnnealedPPO:
      raise ValueError("Annealed runner requires LPACRLAnnealedPPO")
    self.validate_annealed_configuration()

  def observation_corruption_identity(self) -> dict:
    groups = self.env.unwrapped.cfg.observations
    if set(groups) != {"actor", "critic"}:
      raise ValueError("Require exactly actor and critic observation groups")
    value = {
      "version": 1,
      "groups": {name: bool(group.enable_corruption) for name, group in groups.items()},
    }
    if value != CORRUPTION_IDENTITY:
      raise ValueError("Observation corruption must remain disabled")
    return value

  def validate_annealed_configuration(self) -> None:
    self.observation_corruption_identity()
    self.alg.validate_schedule_configuration()
    if self.symmetry_identity() != expected_symmetry_identity():
      raise ValueError("Require mirror loss without data augmentation")

  def save(self, path: str, infos=None) -> None:
    self.validate_annealed_configuration()
    original_iteration = self.current_learning_iteration
    self.current_learning_iteration = max(0, self.alg.completed_updates - 1)
    try:
      super().save(
        path,
        {
          **(infos or {}),
          "lp_acrl_observation_corruption": self.observation_corruption_identity(),
        },
      )
    finally:
      self.current_learning_iteration = original_iteration

  def load(self, path: str, load_cfg=None, strict=True, map_location=None) -> dict:
    self.validate_annealed_configuration()
    saved = torch.load(Path(path), map_location="cpu", weights_only=False)
    count = validate_annealed_checkpoint(saved)
    full = _full_load(load_cfg)
    if full and type(self.cfg.get("seed")) is not int:
      raise ValueError("Full resume requires an explicit fresh-simulator seed")
    result = super().load(path, load_cfg, strict, map_location)
    if full:
      # RSL otherwise repeats the saved iteration label. Our count names the
      # next rollout and continues the saved schedule without repeating a step.
      self.current_learning_iteration = count
      # Constructor episodes preceded the restored LP distribution. Start fresh
      # episodes from that distribution; this does not restore simulator/RNG state.
      from pochi_rl.mjlab.lp_acrl_resume import _equal_tree

      curriculum = deepcopy(self.command_term.curriculum_state_dict())
      self.env.seed(self.cfg["seed"])
      self.env.reset()
      if (
        not _equal_tree(curriculum, self.command_term.curriculum_state_dict())
        or self.env.episode_length_buf.any()
      ):
        self.alg.training_resume_allowed = False
        raise ValueError("Fresh resume reset changed LP statistics or episode age")
    self.validate_annealed_configuration()
    return result

  def learn(self, num_learning_iterations: int, init_at_random_ep_len=False) -> None:
    if not self.alg.training_resume_allowed:
      raise ValueError("Actor-only inference loads cannot start training")
    if self.current_learning_iteration != self.alg.completed_updates:
      raise ValueError("Learning must start at the next completed-update count")
    super().learn(num_learning_iterations, init_at_random_ep_len=False)
    self.current_learning_iteration = self.alg.completed_updates
