"""Opt-in contact-duty balance for Blind-v3, without a gait phase or schedule.

Equal contact duty alone cannot establish walking: all feet planted, all feet in
flight, and synchronous hopping can each have zero variance. Tracking, valid
foot-cycle age, support/attitude and slip terms remain necessary. This kernel
returns an unweighted cost; RewardManager applies its weight and control dt once.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from pochi_rl.mjlab.lp_acrl_balanced_rewards import _contacts, _standing

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg


class moving_contact_duty_variance:
  """Moving-only population variance of four exponentially averaged contacts.

  Duty follows each named foot with a time constant in seconds. Exact-zero
  commands and episode resets restore a neutral 0.5 for every affected foot, so
  standing contacts and previous episodes cannot bias the next walking onset.
  No contact sequence, diagonal pairing or target duty ratio is prescribed.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    tau_s = cfg.params.get("tau_s", 1.0)
    if not math.isfinite(tau_s) or tau_s <= 0:
      raise ValueError("tau_s must be finite and positive")
    if not math.isfinite(env.step_dt) or env.step_dt <= 0:
      raise ValueError("control dt must be finite and positive")
    self.duty = torch.full((env.num_envs, 4), 0.5, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.duty[ids] = 0.5

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    sensor_name: str = "feet_ground_contact",
    command_name: str = "base_velocity",
    tau_s: float = 1.0,
  ) -> torch.Tensor:
    contact = _contacts(env, sensor_name)
    moving = ~_standing(env, command_name)
    # Exact zero-order-hold exponential update avoids a dt-dependent time constant.
    alpha = -math.expm1(-env.step_dt / tau_s)
    updated = self.duty + alpha * (contact.to(self.duty.dtype) - self.duty)
    self.duty.copy_(torch.where(moving[:, None], updated, 0.5))
    centered = self.duty - self.duty.mean(dim=1, keepdim=True)
    return centered.square().mean(dim=1)
