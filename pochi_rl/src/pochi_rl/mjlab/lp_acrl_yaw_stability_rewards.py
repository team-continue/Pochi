"""Stateless yaw-error cost for the explicitly registered blind refinement.

This is classical Huber loss, not normalized smooth-L1. It uses the same
commanded and measured body-frame yaw velocity as the existing Gaussian term.
The RewardManager applies its external negative weight and control dt once.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

ROBOT = SceneEntityCfg("robot")


def yaw_tracking_huber_error(
  env: ManagerBasedRlEnv,
  command_name: str = "base_velocity",
  delta: float = 0.3,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  """Return H_delta(command_yaw - measured_body_yaw), unweighted per world.

  H_delta(e) = .5*e**2 for |e| <= delta, otherwise
  delta*(|e| - .5*delta). Correct nonzero turns therefore have zero cost.
  There is no standing gate, phase, memory, observation or action modification.
  The linear tail retains sensitivity where a Gaussian error reward saturates.
  """
  if (
    isinstance(delta, bool)
    or not isinstance(delta, (int, float))
    or not math.isfinite(delta)
    or delta <= 0
  ):
    raise ValueError("yaw Huber delta must be finite and positive")
  command = env.command_manager.get_command(command_name)
  velocity = env.scene[asset_cfg.name].data.root_link_ang_vel_b[:, 2]
  absolute_error = (command[:, 2] - velocity).abs()
  quadratic_part = absolute_error.clamp_max(delta)
  # Do not square unbounded tail errors, even in an unselected torch.where
  # branch. This stays finite for large finite errors representable in dtype.
  return 0.5 * quadratic_part.square() + delta * (absolute_error - quadratic_part)
