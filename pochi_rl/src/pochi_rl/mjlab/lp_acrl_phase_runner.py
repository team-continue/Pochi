"""Audited48-to50 input expansion; independent optimizer and phase checkpoints."""

from __future__ import annotations

import math
import os
from copy import deepcopy
from pathlib import Path
from types import MethodType

import torch

from pochi_rl.mjlab.lp_acrl_commands import _config_identity
from pochi_rl.mjlab.lp_acrl_no_touchdown_refinement import (
  expected_no_touchdown_task_space,
  validate_no_touchdown_checkpoint,
)
from pochi_rl.mjlab.lp_acrl_phase_gait import PHASE_CONTRACT, PhaseQuietStopMLP
from pochi_rl.mjlab.lp_acrl_runner import (
  LPACRLOnPolicyRunner,
  _capture_with_replay_inputs,
  _log_with_all_episode_keys,
)
from pochi_rl.mjlab.lp_acrl_symmetry import BLIND_OBSERVATION_TERMS
from pochi_rl.mjlab.lp_acrl_wider_exploration import _fresh_reset, _learning_rate
from pochi_rl.mjlab.runner import PochiOnPolicyRunner
from pochi_rl.mjlab.warp_contact_backend import active_contact_backend_identity
from pochi_rl.robot import JOINT_NAMES

STATE_KEY = "lp_acrl_phase_gait_state"
PARENT_SHA = "977cb7793281ec7ab043617e95bd472ccfb250b84840b4ccfe34e36ca995149b"


def expected_task_space():
  from pochi_rl.mjlab.lp_acrl_blind_v17_env_cfg import pochi_lp_acrl_blind_v17_env_cfg

  cfg = pochi_lp_acrl_blind_v17_env_cfg()
  identity = deepcopy(expected_no_touchdown_task_space())
  identity["phase_guidance"] = PHASE_CONTRACT
  identity["environment"]["rewards"] = _config_identity(cfg.rewards)
  identity["environment"]["observations"] = {
    name: tuple((key, _config_identity(term)) for key, term in group.terms.items())
    for name, group in cfg.observations.items()
  }
  return identity


def expand_body_model(source):
  result = deepcopy(source)
  for name in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
    value = source[name]
    if value.shape != (1, 48):
      raise ValueError("Source normalization must be48D")
    pad = value.new_full((1, 2), 0.0 if name.endswith("_mean") else 1.0)
    result[name] = torch.cat((value, pad), dim=-1)
  weight = source["mlp.0.weight"]
  if weight.shape != (256, 48):
    raise ValueError("Source first layer must be256x48")
  result["mlp.0.weight"] = torch.cat((weight, weight.new_zeros((256, 2))), dim=-1)
  return result


def validate_phase_checkpoint(saved):
  info = saved.get("infos") or {}
  state = info.get(STATE_KEY)
  if (
    not isinstance(state, dict)
    or set(state)
    != {
      "version",
      "completed_updates",
      "learning_rate",
      "phase_contract",
      "parent_sha256",
    }
    or state["version"] != 1
    or state["phase_contract"] != PHASE_CONTRACT
    or state["parent_sha256"] != PARENT_SHA
  ):
    raise ValueError("Missing or incompatible phase gait contract")
  if any(
    k in info
    for k in ("lp_acrl_no_touchdown_state", "lp_acrl_rough_no_touchdown_state")
  ):
    raise ValueError("Prior optimizer progress cannot enter phase checkpoints")
  count = state["completed_updates"]
  if type(count) is not int or count < 0 or saved.get("iter") != count - 1:
    raise ValueError("Phase checkpoint iteration/count mismatch")
  if state["learning_rate"] != _learning_rate(saved):
    raise ValueError("Phase adaptive LR mismatch")
  if info.get("lp_acrl", {}).get("task_space") != expected_task_space():
    raise ValueError("Phase checkpoint task/reward/observation identity differs")
  lp = info["lp_acrl"]
  if (
    lp.get("training_terrain_seed", "missing") is not None
    or lp["curriculum"]["beta"] != 5.0
  ):
    raise ValueError("Phase flat terrain/sampler identity differs")
  if info.get("env_state", {}).get("common_step_counter") != count * 24:
    raise ValueError("Phase environment control-step count differs")
  for key in ("actor_state_dict", "critic_state_dict"):
    model = saved[key]
    if model["mlp.0.weight"].shape != (256, 50):
      raise ValueError("Phase model must have50 input columns")
    if any(not torch.isfinite(t).all() for t in model.values()):
      raise ValueError("Nonfinite phase model")
  std = saved["actor_state_dict"]["distribution.std_param"]
  if std.shape != (12,) or not torch.equal(std, torch.full_like(std, 0.15)):
    raise ValueError("Phase moving exploration differs")
  return count


class PhaseGaitRunner(LPACRLOnPolicyRunner):
  def __init__(self, *args, **kwargs):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
      raise ValueError("Phase gait currently supports one training GPU")
    # LP's old constructor validates a48D mirror. Use the same base setup with
    # a separate, explicit50D validator instead of weakening that old check.
    PochiOnPolicyRunner.__init__(self, *args, **kwargs)
    self._backend = deepcopy(active_contact_backend_identity())
    self._ready = False
    self._learning = False
    self._failed = False
    cfg = self.env.unwrapped.cfg
    for name in ("actor", "critic"):
      if tuple(cfg.observations[name].terms) != (
        *BLIND_OBSERVATION_TERMS,
        "gait_phase",
      ):
        raise ValueError("Phase gait observation order differs")
      if cfg.observations[name].enable_corruption:
        raise ValueError("Phase experiment requires clean observations")
    if tuple(cfg.actions["joint_pos"].actuator_names) != tuple(JOINT_NAMES):
      raise ValueError("Phase joint order differs")
    if not isinstance(self.alg._raw_actor, PhaseQuietStopMLP):
      raise ValueError("Phase actor class differs")
    if self.command_term.checkpoint_identity() != expected_task_space():
      raise ValueError("Phase runtime task identity differs")
    if self.alg.symmetry.data_augmentation_func.__name__ != "augment_phase_symmetry":
      raise ValueError("Phase mirror callback differs")
    self.logger.git_status_repos = []
    self.logger._lp_acrl_original_log = self.logger.log
    self.logger.log = MethodType(_log_with_all_episode_keys, self.logger)
    guard = self.env.unwrapped.sim.nan_guard
    if guard.enabled and not hasattr(guard, "_lp_acrl_original_capture"):
      guard._lp_acrl_original_capture = guard.capture
      guard.capture = MethodType(_capture_with_replay_inputs, guard)

  def initialize_from_body(self, saved):
    if validate_no_touchdown_checkpoint(saved) != 601 or saved["iter"] != 600:
      raise ValueError("Phase experiment requires V15/model600")
    if saved["infos"]["lp_acrl_contact_backend"] != self._backend:
      raise ValueError("Source contact backend differs")
    if self.alg.optimizer.state:
      raise ValueError("Phase optimizer must be fresh")
    actor = expand_body_model(saved["actor_state_dict"])
    critic = expand_body_model(saved["critic_state_dict"])
    self.alg._raw_actor.load_state_dict(actor, strict=True)
    self.alg._raw_critic.load_state_dict(critic, strict=True)
    for model, expected in (
      (self.alg._raw_actor, actor),
      (self.alg._raw_critic, critic),
    ):
      if not all(
        torch.equal(v.cpu(), expected[k]) for k, v in model.state_dict().items()
      ):
        raise ValueError("Body-state expansion changed copied tensors")
    self.current_learning_iteration = 0
    self.env.unwrapped.common_step_counter = 0
    self._ready = True
    return {
      "48_body_columns_preserved_bitwise": True,
      "two_phase_weight_columns_initially_zero": True,
      "normalization_count_and_body_statistics_preserved": True,
      "fresh_Adam_and_five_task_LP": True,
      "from_scratch": False,
    }

  def save(self, path, infos=None):
    if not self._ready or self._failed:
      raise ValueError("Uninitialized/inference/failed phase runner cannot save")
    if active_contact_backend_identity() != self._backend:
      raise ValueError("Phase backend changed")
    count = self.current_learning_iteration + int(self._learning)
    rate = self.alg.learning_rate
    if not math.isfinite(rate) or any(
      g["lr"] != rate for g in self.alg.optimizer.param_groups
    ):
      raise ValueError("Phase optimizer LR differs")
    cursor = self.current_learning_iteration
    self.current_learning_iteration = count - 1
    try:
      super().save(
        path,
        {
          **(infos or {}),
          "lp_acrl_contact_backend": self._backend,
          "lp_acrl_training_exploration": self.alg._raw_actor.exploration_identity(),
          STATE_KEY: {
            "version": 1,
            "completed_updates": count,
            "learning_rate": rate,
            "phase_contract": PHASE_CONTRACT,
            "parent_sha256": PARENT_SHA,
          },
        },
      )
    finally:
      self.current_learning_iteration = cursor

  def load(self, path, load_cfg=None, strict=True, map_location=None):
    saved = torch.load(Path(path), map_location="cpu", weights_only=False)
    count = validate_phase_checkpoint(saved)
    info = saved["infos"]
    if info["lp_acrl_contact_backend"] != self._backend:
      raise ValueError("Phase checkpoint backend differs")
    if (
      info["lp_acrl_training_exploration"] != self.alg._raw_actor.exploration_identity()
    ):
      raise ValueError("Phase exploration differs")
    if info["lp_acrl_training_symmetry"] != self.symmetry_identity():
      raise ValueError("Phase symmetry differs")
    self._ready = False
    result = super().load(path, load_cfg, strict, map_location)
    full = load_cfg is None or load_cfg.get("optimizer", False)
    if full:
      self.alg.learning_rate = _learning_rate(saved)
      self.current_learning_iteration = count
      _fresh_reset(self, info["lp_acrl"])
      self._ready = True
    return result

  def learn(self, num_learning_iterations, init_at_random_ep_len=False):
    if not self._ready or self._failed:
      raise ValueError("Phase runner is not ready for training")
    self._learning = True
    try:
      super().learn(num_learning_iterations, init_at_random_ep_len=False)
    except BaseException:
      self._failed = True
      raise
    finally:
      self._learning = False
