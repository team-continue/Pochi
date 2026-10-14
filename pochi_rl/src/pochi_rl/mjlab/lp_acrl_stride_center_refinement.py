"""Fresh reward refinement from V10 models, with independent local progress.

All learned model tensors are retained. Adam and LP restart because the reward
objective changes; the initial LR is the source checkpoint's actual saved LR.
The new counter starts at zero, independently of the parent's 401 updates.
"""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import torch

from pochi_rl.mjlab.lp_acrl_annealed import _full_load
from pochi_rl.mjlab.lp_acrl_commands import _config_identity
from pochi_rl.mjlab.lp_acrl_contact_runner import LPACRLContactOnPolicyRunner
from pochi_rl.mjlab.lp_acrl_refinement import (
  MODEL_ONLY_LOAD,
  _assert_fresh_training_state,
  without_rewards,
)
from pochi_rl.mjlab.lp_acrl_resume import _equal_tree
from pochi_rl.mjlab.lp_acrl_wider_exploration import (
  CORRUPTION_IDENTITY,
  _environment_counter,
  _fresh_reset,
  _learning_rate,
  _preflight_models,
  _validate_std,
  expected_exploration_identity,
  expected_symmetry_identity,
  validate_wider_exploration_checkpoint,
)
from pochi_rl.mjlab.lp_acrl_wider_exploration import (
  STATE_KEY as WIDER_STATE_KEY,
)

SOURCE_TASK = "Pochi-LP-ACRL-Blind-v10"
SOURCE_ITERATION = 400
STATE_KEY = "lp_acrl_stride_center_state"
INITIALIZATION = "model_only_fresh_optimizer_lp"
OLD_REWARD = "moving_common_hip_pitch_deviation"
NEW_REWARD = "moving_fore_aft_stride_center"
STOP_REWARD = "standing_joint_velocity"


def validate_stride_center_checkpoint(saved: dict) -> int:
  """Validate the portable fresh-refinement contract before creating an env."""
  info = saved.get("infos") or {}
  if WIDER_STATE_KEY in info or WIDER_STATE_KEY in saved:
    raise ValueError("Fresh refinement cannot contain V10 continuation state")
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
    raise ValueError("Missing or malformed stride-center progress state")
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
  return count


class LPACRLStrideCenterOnPolicyRunner(LPACRLContactOnPolicyRunner):
  """Independent local updates, including unambiguous initial count0/iter−1."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self._training_ready = False
    self._in_learning = False
    self._failed_learning = False
    self.validate_stride_configuration()

  def observation_corruption_identity(self) -> dict:
    groups = self.env.unwrapped.cfg.observations
    value = {
      "version": 1,
      "groups": {name: bool(group.enable_corruption) for name, group in groups.items()},
    }
    if value != CORRUPTION_IDENTITY:
      raise ValueError("Stride refinement requires clean actor/critic observations")
    return value

  def validate_stride_configuration(self) -> None:
    self.observation_corruption_identity()
    if self.exploration_identity() != expected_exploration_identity():
      raise ValueError("Stride refinement requires fixed moving .15/stopped .01")
    if self.symmetry_identity() != expected_symmetry_identity():
      raise ValueError("Stride refinement requires mirror loss without augmentation")
    _validate_std(self.alg._raw_actor.distribution.std_param)

  def completed_updates(self) -> int:
    return self.current_learning_iteration + int(self._in_learning)

  def _require_training(self) -> None:
    if not self._training_ready or self._failed_learning:
      raise ValueError(
        "Uninitialized, inference-only or failed runner cannot save/train"
      )
    self.validate_stride_configuration()

  def save(self, path: str, infos=None) -> None:
    self._require_training()
    if WIDER_STATE_KEY in (infos or {}):
      raise ValueError("V10 progress must not enter a fresh refinement checkpoint")
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
    self.validate_stride_configuration()
    saved = torch.load(Path(path), map_location="cpu", weights_only=False)
    count = validate_stride_center_checkpoint(saved)
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
      self.validate_stride_configuration()
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


def validate_stride_refinement_identity(runner, saved: dict, parent: dict) -> None:
  """Whitelist the two declared reward changes; preserve every other identity."""
  from pochi_rl.mjlab.lp_acrl_blind_v10_env_cfg import (
    pochi_lp_acrl_blind_v10_env_cfg,
  )
  from pochi_rl.mjlab.lp_acrl_blind_v11_env_cfg import (
    stride_center_reward_cfg,
  )
  from pochi_rl.mjlab.lp_acrl_stride_center_rewards import (
    standing_joint_velocity_bounded_all_contact,
  )

  if (
    parent.get("task") != SOURCE_TASK
    or type(saved.get("iter")) is not int
    or saved["iter"] != SOURCE_ITERATION
  ):
    raise ValueError("Stride refinement requires pinned Blind-v10 model400")
  validate_wider_exploration_checkpoint(saved)
  info = saved["infos"]
  if info.get("lp_acrl_checkpoint_version") != 1 or STATE_KEY in info:
    raise ValueError("Source must be an independent V10 checkpoint")
  rate = _learning_rate(saved)
  if runner.alg.learning_rate != rate or any(
    group["lr"] != rate for group in runner.alg.optimizer.param_groups
  ):
    raise ValueError("Fresh Adam must start at the source's actual saved LR")
  if type(runner.cfg.get("seed")) is not int or runner.cfg["seed"] != parent.get(
    "seed"
  ):
    raise ValueError("Fresh refinement seed must match its declared parent seed")
  term = runner.command_term
  current = term.checkpoint_identity()
  source = (info.get("lp_acrl") or {}).get("task_space")
  if without_rewards(source) != without_rewards(current):
    raise ValueError("Stride refinement differs outside rewards")
  if not source.get("command_protocol"):
    raise ValueError("Source lacks the blind stop/start command protocol")
  rewards = deepcopy(source["environment"]["rewards"])
  source_rewards = pochi_lp_acrl_blind_v10_env_cfg().rewards
  if rewards != _config_identity(source_rewards):
    raise ValueError("Source rewards differ from the complete V10 factory")
  rewards.pop(OLD_REWARD)
  rewards[NEW_REWARD] = _config_identity(stride_center_reward_cfg())
  expected_stop = deepcopy(source_rewards[STOP_REWARD])
  expected_stop.func = standing_joint_velocity_bounded_all_contact
  rewards[STOP_REWARD] = _config_identity(expected_stop)
  if current["environment"]["rewards"] != rewards:
    raise ValueError(
      "Only the exact stride-center and stopped-qdot changes are allowed"
    )
  terrain = term._env.cfg.scene.terrain
  generator = terrain.terrain_generator if terrain is not None else None
  if info["lp_acrl"].get("training_terrain_seed") != (
    None if generator is None else generator.seed
  ):
    raise ValueError("Training terrain seed must remain unchanged")
  runner.validate_stride_configuration()
  backend = runner.contact_backend_identity()
  if not backend or any(
    value != backend
    for value in (info.get("lp_acrl_contact_backend"), parent.get("contact_backend"))
  ):
    raise ValueError("Checkpoint, parent and target contact backend must match")


def initialize_stride_center_refinement(runner, saved: dict, parent: dict) -> dict:
  """Transfer models exactly, with independent fresh Adam, LP and simulator state."""
  validate_stride_refinement_identity(runner, saved, parent)
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
    runner.validate_stride_configuration()
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
    "initial_environment_step_counter": 0,
    "fresh_simulator_reset": True,
    "fresh_reset_seed": runner.cfg["seed"],
    "initial_exploration_identity": runner.exploration_identity(),
    "observation_corruption_unchanged": runner.observation_corruption_identity(),
    "symmetry_unchanged": runner.symmetry_identity(),
    "contact_backend_unchanged": runner.contact_backend_identity(),
    "allowed_environment_identity_difference": [
      f"environment.rewards.{name}" for name in (OLD_REWARD, NEW_REWARD, STOP_REWARD)
    ],
  }
