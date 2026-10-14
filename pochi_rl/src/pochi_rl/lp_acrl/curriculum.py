"""Signed learning-progress task sampling from arXiv:2601.17428.

Only completed, unnormalized episodic returns enter the stage means. Returns
of unfinished episodes must remain in the environment until those episodes
end. This module neither combines PPO losses with returns nor computes
absolute learning progress: deterioration is deliberately negative.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class StageUpdate:
  stage_index: int
  mean_returns: torch.Tensor
  episode_counts: torch.Tensor
  learning_progress: torch.Tensor
  probabilities: torch.Tensor
  compared_tasks: torch.Tensor


class LPACRLCurriculum:
  """LP_i = mean_return_i(now) - mean_return_i(previous); p=softmax(LP/beta).

  Each task needs ``min_episodes`` in both consecutive stages for comparison.
  New, missing, or insufficiently sampled tasks receive LP=0. Missing stages
  invalidate the baseline, avoiding comparisons across unequal time spans.
  This cold-start/sample-count policy is an explicit implementation choice;
  the paper does not specify it. Initial sampling is uniform. Sampling uses
  the pure paper softmax, without absolute values or an exploration mixture.
  """

  _VERSION = 1

  def __init__(
    self,
    num_tasks: int,
    *,
    beta: float = 5.0,
    stage_steps: int = 2400,
    min_episodes: int = 2,
    device: str | torch.device = "cpu",
  ) -> None:
    for name, value in (
      ("num_tasks", num_tasks),
      ("stage_steps", stage_steps),
      ("min_episodes", min_episodes),
    ):
      if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    if not math.isfinite(beta) or beta <= 0:
      raise ValueError("beta must be finite and positive")
    self.num_tasks = num_tasks
    self.beta = float(beta)
    self.stage_steps = stage_steps
    self.min_episodes = min_episodes
    self.device = torch.device(device)
    self.stage_index = 0
    self.steps_in_stage = 0
    self.return_sums = torch.zeros(num_tasks, dtype=torch.float64, device=device)
    self.episode_counts = torch.zeros(num_tasks, dtype=torch.long, device=device)
    self.previous_means = torch.zeros_like(self.return_sums)
    self.previous_valid = torch.zeros(num_tasks, dtype=torch.bool, device=device)
    self.learning_progress = torch.zeros_like(self.return_sums)
    self.probabilities = torch.full_like(self.return_sums, 1 / num_tasks)

  @torch.no_grad()
  def record_episodes(self, task_ids: torch.Tensor, returns: torch.Tensor) -> None:
    """Add one true completed return per task ID; repeated task IDs are summed."""
    if task_ids.ndim != 1 or returns.ndim != 1 or task_ids.shape != returns.shape:
      raise ValueError("task_ids and returns must be equally sized vectors")
    if task_ids.dtype not in (torch.int32, torch.int64):
      raise ValueError("task_ids must be integer tensors")
    if not returns.is_floating_point() or not torch.isfinite(returns).all():
      raise ValueError("returns must be finite floating-point values")
    ids = task_ids.to(device=self.device, dtype=torch.long)
    if (ids < 0).any() or (ids >= self.num_tasks).any():
      raise ValueError("task_ids are outside the configured catalog")
    values = returns.detach().to(device=self.device, dtype=torch.float64)
    new_sums = self.return_sums.scatter_add(0, ids, values)
    if not torch.isfinite(new_sums).all():
      raise ValueError("episode return accumulation overflowed")
    new_counts = self.episode_counts.scatter_add(0, ids, torch.ones_like(ids))
    if (new_counts < self.episode_counts).any():
      raise ValueError("episode count accumulation overflowed")
    self.return_sums.copy_(new_sums)
    self.episode_counts.copy_(new_counts)

  @torch.no_grad()
  def advance(self, num_steps: int = 1) -> StageUpdate | None:
    """Advance control-step cadence, updating at a stage boundary.

    At most one boundary may be crossed per call, since aggregated episode
    returns cannot be assigned retrospectively to multiple unseen stages.
    Record completions for the final step before calling this method.
    """
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps < 1:
      raise ValueError("num_steps must be a positive integer")
    total = self.steps_in_stage + num_steps
    if total >= 2 * self.stage_steps:
      raise ValueError("advance must not cross more than one stage boundary")
    if total < self.stage_steps:
      self.steps_in_stage = total
      return None
    update = self.end_stage()
    self.steps_in_stage = total - self.stage_steps
    return update

  @torch.no_grad()
  def end_stage(self) -> StageUpdate:
    """Close the collected stage explicitly and return an immutable snapshot."""
    valid = self.episode_counts >= self.min_episodes
    means = self.return_sums / self.episode_counts.clamp_min(1)
    compared = valid & self.previous_valid
    progress = torch.where(compared, means - self.previous_means, 0.0)
    # Two finite extreme means can overflow on subtraction. Saturation keeps
    # the sign and ordering usable, without silently allowing NaN probabilities.
    progress = torch.nan_to_num(progress)
    # Center before dividing: positive beta cannot turn the max logit into inf.
    # Negative infinity is safe here: exp(-inf)=0, and at least one logit is 0.
    logits = (progress - progress.max()) / self.beta
    probabilities = torch.softmax(logits, dim=0)
    self.previous_means.copy_(means)
    self.previous_valid.copy_(valid)
    self.learning_progress.copy_(progress)
    self.probabilities.copy_(probabilities)
    self.stage_index += 1
    result = StageUpdate(
      stage_index=self.stage_index,
      mean_returns=means.clone(),
      episode_counts=self.episode_counts.clone(),
      learning_progress=progress.clone(),
      probabilities=probabilities.clone(),
      compared_tasks=compared.clone(),
    )
    self.return_sums.zero_()
    self.episode_counts.zero_()
    self.steps_in_stage = 0
    return result

  @torch.no_grad()
  def sample(
    self, num_samples: int, *, generator: torch.Generator | None = None
  ) -> torch.Tensor:
    """Sample task IDs with replacement; generator must match this device."""
    if (
      isinstance(num_samples, bool)
      or not isinstance(num_samples, int)
      or num_samples < 0
    ):
      raise ValueError("num_samples must be a nonnegative integer")
    if num_samples == 0:
      return torch.empty(0, dtype=torch.long, device=self.device)
    return torch.multinomial(
      self.probabilities, num_samples, replacement=True, generator=generator
    )

  def state_dict(self) -> dict[str, Any]:
    """Independent CPU copies, including partial-stage episode aggregates."""
    result: dict[str, Any] = {
      "version": self._VERSION,
      "num_tasks": self.num_tasks,
      "beta": self.beta,
      "stage_steps": self.stage_steps,
      "min_episodes": self.min_episodes,
      "stage_index": self.stage_index,
      "steps_in_stage": self.steps_in_stage,
    }
    for name in self._tensor_names():
      result[name] = getattr(self, name).detach().cpu().clone()
    return result

  def load_state_dict(self, state: dict[str, Any]) -> None:
    """Validate fully before restoring; incompatible experiment state is refused."""
    expected = self.state_dict()
    if set(state) != set(expected):
      raise ValueError("curriculum state has missing or unexpected keys")
    for name in ("version", "num_tasks", "beta", "stage_steps", "min_episodes"):
      if state[name] != expected[name] or type(state[name]) is not type(expected[name]):
        raise ValueError(f"incompatible curriculum {name}")
    for name in ("stage_index", "steps_in_stage"):
      if type(state[name]) is not int or state[name] < 0:
        raise ValueError(f"invalid {name}")
    if state["steps_in_stage"] >= self.stage_steps:
      raise ValueError("steps_in_stage must be smaller than stage_steps")
    tensors = {}
    for name in self._tensor_names():
      value = state[name]
      template = getattr(self, name)
      if (
        not isinstance(value, torch.Tensor)
        or value.shape != template.shape
        or value.dtype != template.dtype
      ):
        raise ValueError(f"invalid shape or dtype for {name}")
      if value.is_floating_point() and not torch.isfinite(value).all():
        raise ValueError(f"nonfinite curriculum state {name}")
      tensors[name] = value.detach().to(self.device).clone()
    if (tensors["episode_counts"] < 0).any():
      raise ValueError("episode_counts must be nonnegative")
    empty = tensors["episode_counts"] == 0
    if (tensors["return_sums"][empty] != 0).any():
      raise ValueError("empty episode counts must have zero return sums")
    probabilities = tensors["probabilities"]
    if (probabilities < 0).any() or not torch.isclose(
      probabilities.sum(), probabilities.new_tensor(1.0), rtol=1e-10, atol=1e-12
    ):
      raise ValueError("probabilities must be nonnegative and sum to one")
    progress = tensors["learning_progress"]
    expected_probabilities = torch.softmax((progress - progress.max()) / self.beta, 0)
    if not torch.allclose(
      probabilities, expected_probabilities, rtol=1e-10, atol=1e-12
    ):
      raise ValueError("probabilities do not match signed learning progress")
    for name, value in tensors.items():
      getattr(self, name).copy_(value)
    self.stage_index = state["stage_index"]
    self.steps_in_stage = state["steps_in_stage"]

  @staticmethod
  def _tensor_names() -> tuple[str, ...]:
    return (
      "return_sums",
      "episode_counts",
      "previous_means",
      "previous_valid",
      "learning_progress",
      "probabilities",
    )
