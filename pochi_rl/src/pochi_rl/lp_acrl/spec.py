"""Task discretization and explicitly chosen Pochi adaptation parameters.

The rough experiment follows the paper's 5 x 6 x 5 x 4 Cartesian product.
Speed bounds are scaled to Pochi here. The 50 Hz control rate, 20 s episode,
beta, stage duration, and minimum sample count are engineering choices: the
paper does not give their numerical values. No previous weights are involved.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TaskCatalog:
  """A complete Cartesian product, with stable task IDs across checkpoints.

  Task-index columns are ``forward_bin, yaw_bin, terrain_index, level``.
  Bounds are nonnegative magnitudes; an environment may independently sample
  yaw direction. The zero-width (0, 0) yaw interval denotes no yaw command.
  """

  forward_edges: tuple[float, ...]
  yaw_edges: tuple[float, ...]
  terrains: tuple[str, ...]
  num_levels: int

  def __post_init__(self) -> None:
    for name, edges in (
      ("forward_edges", self.forward_edges),
      ("yaw_edges", self.yaw_edges),
    ):
      if len(edges) < 2 or not all(math.isfinite(x) and x >= 0 for x in edges):
        raise ValueError(f"{name} requires at least two finite nonnegative edges")
      if name == "yaw_edges" and edges == (0.0, 0.0):
        continue
      if any(right <= left for left, right in zip(edges, edges[1:], strict=False)):
        raise ValueError(f"{name} must be strictly increasing")
    if (
      not self.terrains
      or any(not isinstance(name, str) or not name for name in self.terrains)
      or len(set(self.terrains)) != len(self.terrains)
    ):
      raise ValueError("terrains must contain distinct nonempty names")
    if isinstance(self.num_levels, bool) or not isinstance(self.num_levels, int):
      raise ValueError("num_levels must be a positive integer")
    if self.num_levels < 1:
      raise ValueError("num_levels must be positive")

  @property
  def num_tasks(self) -> int:
    return (
      (len(self.forward_edges) - 1)
      * (len(self.yaw_edges) - 1)
      * len(self.terrains)
      * self.num_levels
    )

  def task_indices(self, device: str | torch.device = "cpu") -> torch.Tensor:
    """Return every task's four public discrete indices as int64 [N, 4]."""
    return torch.cartesian_prod(
      torch.arange(len(self.forward_edges) - 1, device=device),
      torch.arange(len(self.yaw_edges) - 1, device=device),
      torch.arange(len(self.terrains), device=device),
      torch.arange(self.num_levels, device=device),
    )

  def task_bounds(self, device: str | torch.device = "cpu") -> torch.Tensor:
    """Return [N, 2, 2]: task, (forward m/s, yaw rad/s), (low, high)."""
    indices = self.task_indices(device)
    forward = torch.tensor(self.forward_edges, device=device)
    yaw = torch.tensor(self.yaw_edges, device=device)
    return torch.stack(
      (
        torch.stack((forward[indices[:, 0]], forward[indices[:, 0] + 1]), -1),
        torch.stack((yaw[indices[:, 1]], yaw[indices[:, 1] + 1]), -1),
      ),
      dim=1,
    )


@dataclass(frozen=True)
class LPACRLSpec:
  beta: float = 5.0
  stage_steps: int = 2400  # 100 PPO iterations, each 24 control steps.
  min_episodes: int = 2
  control_hz: float = 50.0
  episode_length_s: float = 20.0
  ppo_steps_per_iteration: int = 24

  def __post_init__(self) -> None:
    for name in ("beta", "control_hz", "episode_length_s"):
      value = getattr(self, name)
      if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    for name in ("stage_steps", "min_episodes", "ppo_steps_per_iteration"):
      value = getattr(self, name)
      if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")

  @property
  def episode_steps(self) -> int:
    return round(self.control_hz * self.episode_length_s)


FLAT_TASK_CATALOG = TaskCatalog(
  forward_edges=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
  yaw_edges=(0.0, 0.0),
  terrains=("flat",),
  num_levels=1,
)

ROUGH_TASK_CATALOG = TaskCatalog(
  forward_edges=(0.0, 0.1, 0.2, 0.3, 0.4, 0.5),
  yaw_edges=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2),
  terrains=("ascending_stairs", "descending_stairs", "rough", "uphill", "downhill"),
  num_levels=4,
)

LP_ACRL_SPEC = LPACRLSpec()
