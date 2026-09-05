"""Loading a trained actor's forward pass out of an RSL-RL checkpoint.

Shared by ``scripts/export_policy.py`` and
``pochi_hardware/src/run_walk_hardware.py`` -- both need only the actor's
forward pass and nothing else from the checkpoint.

RSL-RL's own checkpoints (``{"actor_state_dict": ..., "critic_state_dict":
..., ...}``) store the actor as a plain ``state_dict`` of tensors, not a
pickled module -- there is no ``nn.Module`` to just unpickle and call, so
``load_rsl_rl_actor`` rebuilds one (empirical observation normalization +
MLP) from the tensor shapes and loads the weights into it. The action
distribution's own std (``distribution.std_param``) is intentionally not
reproduced: a hardware policy should run the deterministic mean action, not
a sampled one.

A handful of older/hypothetical checkpoint shapes -- a bare ``nn.Module``, or
a dict that already holds one under a conventional key -- are still handled
by ``find_actor`` as a fallback.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import torch
from torch import nn

_ACTIVATIONS: dict[str, type[nn.Module]] = {
  "elu": nn.ELU,
  "relu": nn.ReLU,
  "tanh": nn.Tanh,
  "selu": nn.SELU,
}


class NormalizedMlpActor(nn.Module):
  """Empirical observation normalization, then an MLP. See module docstring."""

  def __init__(self, mlp: nn.Sequential, obs_dim: int) -> None:
    super().__init__()
    self.mlp = mlp
    self.register_buffer("obs_mean", torch.zeros(1, obs_dim))
    self.register_buffer("obs_std", torch.ones(1, obs_dim))

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    # Matches rsl_rl.modules.normalization.EmpiricalNormalization.forward
    # exactly: (x - mean) / (std + eps), eps=1e-2 by default, no clipping
    # (checked against the actual rsl-rl-lib==5.2.0 source, the version this
    # project pins). An earlier version of this used `/ std` with no eps and
    # a +/-10 clamp that doesn't exist in rsl_rl at all -- confirmed wrong by
    # reconstructing observations from a hardware log the buggy version had
    # produced: replaying them through *this* (eps=1e-2, no-clip) formula
    # reproduces the log's recorded actions much worse (RMSE ~0.10 on a
    # near-stationary segment) than replaying through the old formula does
    # (RMSE ~0.002, as expected -- that's what actually generated the log).
    normalized = (obs - self.obs_mean) / (self.obs_std + 0.01)
    return self.mlp(normalized)


def _build_mlp(
  state_dict: dict[str, torch.Tensor], *, prefix: str, activation: str
) -> nn.Sequential:
  indices = sorted(
    {
      int(match.group(1))
      for key in state_dict
      if (match := re.fullmatch(rf"{re.escape(prefix)}(\d+)\.weight", key))
    }
  )
  if not indices:
    raise RuntimeError(f"no '{prefix}<n>.weight' tensors in the state_dict")
  layers: list[nn.Module] = []
  for position, index in enumerate(indices):
    out_features, in_features = state_dict[f"{prefix}{index}.weight"].shape
    layers.append(nn.Linear(in_features, out_features))
    if position < len(indices) - 1:
      layers.append(_ACTIVATIONS[activation]())
  return nn.Sequential(*layers)


def load_rsl_rl_actor(
  state_dict: dict[str, torch.Tensor], *, activation: str = "elu"
) -> NormalizedMlpActor:
  """Build and load an actor from an RSL-RL ``actor_state_dict``."""
  mlp = _build_mlp(state_dict, prefix="mlp.", activation=activation)
  obs_dim = mlp[0].in_features
  actor = NormalizedMlpActor(mlp, obs_dim)
  mlp_state = {
    key[len("mlp.") :]: value
    for key, value in state_dict.items()
    if key.startswith("mlp.")
  }
  actor.mlp.load_state_dict(mlp_state)
  if "obs_normalizer._mean" in state_dict:
    actor.obs_mean.copy_(state_dict["obs_normalizer._mean"])
    actor.obs_std.copy_(state_dict["obs_normalizer._std"])
  return actor.eval()


def find_actor(checkpoint: Any) -> torch.nn.Module:
  """Fallback for a checkpoint that already holds a pickled ``nn.Module``."""
  if isinstance(checkpoint, torch.nn.Module):
    return checkpoint
  if isinstance(checkpoint, dict):
    for key in ("actor", "policy", "actor_critic", "model", "student"):
      value = checkpoint.get(key)
      if isinstance(value, torch.nn.Module):
        if hasattr(value, "actor") and isinstance(value.actor, torch.nn.Module):
          return value.actor
        return value
    for value in checkpoint.values():
      if isinstance(value, torch.nn.Module):
        return value
  raise RuntimeError(
    "Could not find or rebuild an actor from the checkpoint (got a "
    f"{type(checkpoint).__name__}, with keys "
    f"{list(checkpoint.keys()) if isinstance(checkpoint, dict) else 'n/a'})."
  )


def load_actor(path: str | Path, *, map_location: str = "cpu") -> torch.nn.Module:
  checkpoint = torch.load(path, map_location=map_location, weights_only=False)
  if isinstance(checkpoint, dict) and isinstance(
    checkpoint.get("actor_state_dict"), dict
  ):
    return load_rsl_rl_actor(checkpoint["actor_state_dict"])
  return find_actor(checkpoint).eval()
