"""Paper evaluation metrics, with explicit near-zero command conventions."""

from __future__ import annotations

import math

import torch


def percentage_tracking_error(
  actual: torch.Tensor,
  command: torch.Tensor,
  *,
  denominator_floor: float = 0.05,
) -> torch.Tensor:
  """Absolute per-axis relative error; 0.30 means 30 percent, without clipping.

  The paper does not define percentage-error normalization at zero command.
  This implementation divides by max(abs(command), denominator_floor), in the
  axis's own units. Thus command=0 and actual=0 produce zero error; an actual
  speed of 0.05 against a zero command with floor=0.05 produces error 1.0.
  The floor is an engineering choice and should be saved with evaluation data.
  Average the returned errors over the steps preceding the first fall.
  """
  if not math.isfinite(denominator_floor) or denominator_floor <= 0:
    raise ValueError("denominator_floor must be finite and positive")
  if not actual.is_floating_point() or not command.is_floating_point():
    raise ValueError("actual and command must be floating-point tensors")
  if not torch.isfinite(actual).all() or not torch.isfinite(command).all():
    raise ValueError("actual and command must be finite")
  return (actual - command).abs() / command.abs().clamp_min(denominator_floor)


def epte_sp(
  tracking_error: torch.Tensor,
  alive_steps: torch.Tensor | int,
  horizon_steps: int,
) -> torch.Tensor:
  """Equation (8): (epsilon * k_f + (K - k_f)) / K.

  ``tracking_error`` is the mean percentage error before the first fall, and
  ``alive_steps`` is k_f, or K when no fall occurs. For an immediate fall pass
  zero error and zero alive steps, giving the maximum survival penalty 1.0.
  The metric can exceed 1.0 because tracking error is not clipped. To evaluate
  separate axes use errors of shape [episodes, axes] and counts [episodes, 1].
  """
  if (
    isinstance(horizon_steps, bool)
    or not isinstance(horizon_steps, int)
    or horizon_steps < 1
  ):
    raise ValueError("horizon_steps must be a positive integer")
  if not tracking_error.is_floating_point():
    raise ValueError("tracking_error must be a floating-point tensor")
  if not torch.isfinite(tracking_error).all() or (tracking_error < 0).any():
    raise ValueError("tracking_error must be finite and nonnegative")
  steps = torch.as_tensor(alive_steps, device=tracking_error.device)
  if (
    not torch.isfinite(steps).all()
    or (steps < 0).any()
    or (steps > horizon_steps).any()
    or (steps != steps.floor()).any()
  ):
    raise ValueError("alive_steps must contain integers between zero and horizon")
  fraction = steps.to(tracking_error.dtype) / horizon_steps
  return tracking_error * fraction + (1 - fraction)
