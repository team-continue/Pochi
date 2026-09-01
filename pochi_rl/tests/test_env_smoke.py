from __future__ import annotations

import pytest
import torch


def _assert_finite_tree(value) -> None:
  if isinstance(value, torch.Tensor):
    assert torch.isfinite(value).all()
  elif isinstance(value, dict):
    for child in value.values():
      _assert_finite_tree(child)
  elif isinstance(value, (tuple, list)):
    for child in value:
      _assert_finite_tree(child)


def test_env_random_rollout_nan_free() -> None:
  pytest.importorskip("mjlab")
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  import pochi_rl  # noqa: F401

  cfg = load_env_cfg("Pochi-Velocity-Flat-v0")
  cfg.scene.num_envs = 4
  cfg.events.pop("base_mass", None)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  try:
    obs, _ = env.reset()
    _assert_finite_tree(obs)
    for _ in range(50):
      actions = 2.0 * torch.rand(env.action_space.shape) - 1.0
      obs, reward, terminated, truncated, info = env.step(actions)
      _assert_finite_tree(obs)
      _assert_finite_tree(reward)
      _assert_finite_tree(terminated)
      _assert_finite_tree(truncated)
      _assert_finite_tree(info)
  finally:
    env.close()
