"""The RSL-RL checkpoint loader: normalization formula and shape inference.

The normalization formula in particular is a pin, not a guess -- see
``NormalizedMlpActor.forward``'s docstring for how a mismatched version of
it (missing rsl_rl's epsilon, plus a clamp rsl_rl doesn't have) was found on
a real hardware log.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from pochi_rl.policy_io import NormalizedMlpActor, load_rsl_rl_actor  # noqa: E402


def _identity_mlp_state_dict(dim: int) -> dict:
  """A single Linear(dim, dim) layer set to the identity, so the actor's
  output is exactly its normalized input -- isolates the normalization
  formula from anything the MLP itself might do."""
  return {
    "obs_normalizer._mean": torch.zeros(1, dim),
    "obs_normalizer._var": torch.ones(1, dim),
    "obs_normalizer._std": torch.full((1, dim), 2.0),
    "obs_normalizer.count": torch.tensor(0),
    "mlp.0.weight": torch.eye(dim),
    "mlp.0.bias": torch.zeros(dim),
  }


def test_normalization_matches_rsl_rl_formula() -> None:
  """(x - mean) / (std + 0.01), no clipping -- rsl_rl.modules.normalization.
  EmpiricalNormalization.forward, exactly."""
  dim = 4
  actor = load_rsl_rl_actor(_identity_mlp_state_dict(dim))
  obs = torch.tensor([[1.0, -1.0, 100.0, -100.0]])

  expected = obs / (2.0 + 0.01)
  with torch.inference_mode():
    got = actor(obs)
  assert got.numpy() == pytest.approx(expected.numpy(), abs=1e-6)


def test_normalization_does_not_clip() -> None:
  """A prior version clamped to +/-10 post-normalization; rsl_rl never does."""
  dim = 2
  mlp = torch.nn.Sequential(torch.nn.Identity())
  actor = NormalizedMlpActor(mlp, dim)
  actor.obs_std.fill_(0.001)  # tiny std -> a huge normalized value
  obs = torch.tensor([[1.0, -1.0]])
  with torch.inference_mode():
    got = actor(obs)
  assert np.abs(got.numpy()).max() > 10.0
