"""Strict Blind-v3 training-state resume with a fresh simulator reset."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import torch

TASK_ID = "Pochi-LP-ACRL-Blind-v3"


def sha256(path: Path) -> str:
  value = hashlib.sha256()
  with path.open("rb") as stream:
    for block in iter(lambda: stream.read(1024 * 1024), b""):
      value.update(block)
  return value.hexdigest()


def validate_resume_sources(project: Path, parent: dict) -> dict[str, str]:
  """Require every recorded source/asset to match; new resume files are additive."""
  if parent.get("task") != TASK_ID:
    raise ValueError("Full resume requires the same Blind-v3 task")
  expected = parent.get("source_sha256")
  required = {
    "assets/pochi/pochi.xml", "src/pochi_rl/mjlab/lp_acrl_quiet_stop.py",
    "src/pochi_rl/mjlab/lp_acrl_blind_v3_rewards.py",
    "src/pochi_rl/mjlab/lp_acrl_runner.py",
  }
  if not isinstance(expected, dict) or not required.issubset(expected) or not any(
    name.startswith("assets/pochi/meshes/") for name in expected
  ):
    raise ValueError("Parent must record complete source and robot asset hashes")
  project = project.resolve()
  for name, digest in expected.items():
    source = (project / name).resolve()
    if not source.is_relative_to(project) or sha256(source) != digest:
      raise ValueError(f"Resume source or physical asset changed: {name}")
  return dict(expected)


def _equal_tree(first, second) -> bool:
  if isinstance(first, torch.Tensor):
    return isinstance(second, torch.Tensor) and torch.equal(
      first.detach().cpu(), second.detach().cpu()
    )
  if isinstance(first, dict):
    return isinstance(second, dict) and first.keys() == second.keys() and all(
      _equal_tree(value, second[key]) for key, value in first.items()
    )
  if isinstance(first, (list, tuple)):
    return type(first) is type(second) and len(first) == len(second) and all(
      _equal_tree(a, b) for a, b in zip(first, second, strict=True)
    )
  return first == second


def _assert_finite_tree(value) -> None:
  if isinstance(value, torch.Tensor):
    if not torch.isfinite(value).all():
      raise ValueError("Resume checkpoint contains nonfinite training state")
  elif isinstance(value, dict):
    for item in value.values():
      _assert_finite_tree(item)
  elif isinstance(value, (list, tuple)):
    for item in value:
      _assert_finite_tree(item)
  elif isinstance(value, float) and not math.isfinite(value):
    raise ValueError("Resume checkpoint contains a nonfinite scalar")


def resume_training_state(runner, checkpoint: Path, parent: dict, seed: int) -> dict:
  """Restore model/optimizer/LP exactly, then reset unsaved simulation state.

  RSL checkpoints are post-update but store that update's zero-based index.
  RSL's optimizer loader does not restore the separate adaptive-learning-rate
  attribute. Both details are handled explicitly here.
  """
  if parent.get("task") != TASK_ID or seed != parent.get("seed"):
    raise ValueError("Resume requires the same Blind-v3 task and reset seed")
  saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
  iteration = saved.get("iter")
  if not isinstance(iteration, int) or isinstance(iteration, bool) or iteration < 0:
    raise ValueError("Invalid post-update checkpoint iteration")
  metadata = saved.get("infos") or {}
  if metadata.get("lp_acrl_checkpoint_version") != 1:
    raise ValueError("Expected an independent LP-ACRL checkpoint")
  term = runner.command_term
  lp = metadata.get("lp_acrl") or {}
  if lp.get("task_space") != term.checkpoint_identity():
    raise ValueError("Resume environment identity differs, including rewards")
  terrain = term._env.cfg.scene.terrain
  generator = terrain.terrain_generator if terrain is not None else None
  terrain_seed = None if generator is None else generator.seed
  if "training_terrain_seed" not in lp or lp["training_terrain_seed"] != terrain_seed:
    raise ValueError("Resume training terrain seed differs")
  if metadata.get("lp_acrl_training_symmetry") != runner.symmetry_identity():
    raise ValueError("Resume training symmetry differs")
  exploration = runner.exploration_identity()
  if not exploration or metadata.get("lp_acrl_training_exploration") != exploration:
    raise ValueError("Resume exploration identity differs")
  for key, model in (
    ("actor_state_dict", runner.alg._raw_actor),
    ("critic_state_dict", runner.alg._raw_critic),
  ):
    source = saved.get(key)
    target = model.state_dict()
    if not isinstance(source, dict) or source.keys() != target.keys():
      raise ValueError(f"Resume {key} architecture/state keys differ")
    for name, value in source.items():
      if (
        not isinstance(value, torch.Tensor) or value.shape != target[name].shape
        or value.dtype != target[name].dtype
      ):
        raise ValueError(f"Resume {key}/{name} shape or dtype differs")
    _assert_finite_tree(source)
  optimizer = saved.get("optimizer_state_dict")
  if not isinstance(optimizer, dict):
    raise ValueError("Resume requires optimizer state")
  _assert_finite_tree(optimizer)
  _assert_finite_tree(lp.get("curriculum"))
  groups = optimizer.get("param_groups", [])
  target_groups = runner.alg.optimizer.state_dict()["param_groups"]
  if len(groups) != len(target_groups) or any(
    len(group["params"]) != len(target["params"])
    for group, target in zip(groups, target_groups, strict=True)
  ):
    raise ValueError("Resume optimizer parameter groups differ")
  rates = [group.get("lr") for group in groups]
  if not rates or any(
    not isinstance(rate, (float, int)) or not math.isfinite(rate) or rate <= 0
    for rate in rates
  ) or len(set(rates)) != 1:
    raise ValueError("RSL adaptive PPO requires a finite shared optimizer LR")

  # The frozen runner additionally validates and restores the complete LP state.
  runner.load(str(checkpoint), load_cfg=None, strict=True, map_location=runner.device)
  if runner.current_learning_iteration != iteration:
    raise ValueError("Runner failed to restore the saved iteration")
  for key, model in (
    ("actor_state_dict", runner.alg._raw_actor),
    ("critic_state_dict", runner.alg._raw_critic),
  ):
    if not _equal_tree(saved[key], model.state_dict()):
      raise ValueError(f"Resume {key} differs after strict load")
  if not _equal_tree(optimizer, runner.alg.optimizer.state_dict()):
    raise ValueError("Optimizer differs after full-state load")
  if not _equal_tree(lp, term.curriculum_state_dict()):
    raise ValueError("LP curriculum differs after full-state load")
  runner.alg.learning_rate = float(rates[0])
  runner.current_learning_iteration = iteration + 1
  # Initial wrapper construction has already reset once; reseed and reset again
  # so these episodes sample the restored curriculum, not the default sampler.
  runner.env.seed(seed)
  runner.env.reset()
  if not _equal_tree(lp, term.curriculum_state_dict()):
    raise ValueError("Fresh simulator reset altered restored LP statistics")
  if runner.env.episode_length_buf.any():
    raise ValueError("Fresh simulator episodes must start at length zero")
  return {
    "loaded": ["actor", "critic", "normalizers", "optimizer", "LP", "iteration"],
    "preload_equality_verified": True,
    "saved_post_update_iteration": iteration,
    "next_iteration": runner.current_learning_iteration,
    "adaptive_learning_rate_restored": runner.alg.learning_rate,
    "optimizer_learning_rates": rates,
    "lp_stage": term.curriculum.stage_index,
    "lp_steps_in_stage": term.curriculum.steps_in_stage,
    "fresh_simulator_reset": True, "fresh_reset_seed": seed,
    "simulator_state_and_rng_restored": False,
    "unsaved_partial_episodes": "discarded; reset adds no completed LP samples",
    "rollout_storage": "fresh; checkpoint was saved after a completed PPO update",
  }
