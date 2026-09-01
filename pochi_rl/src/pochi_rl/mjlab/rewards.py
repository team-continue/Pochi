"""Small Pochi-specific reward helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

ROBOT = SceneEntityCfg("robot")


def base_lin_vel_z_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def base_ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def _standing(env: ManagerBasedRlEnv, command_name: str, threshold: float):
  """1.0 for environments whose velocity command is (near) zero."""
  command = env.command_manager.get_command(command_name)
  magnitude = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return (magnitude < threshold).float()


def stand_still_joint_deviation(
  env: ManagerBasedRlEnv,
  command_name: str = "base_velocity",
  command_threshold: float = 0.5,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  """Penalize drifting off the default stance while commanded to stand.

  ``feet_air_time`` already stops paying out below ``command_threshold``, so a
  standing robot gains nothing by stepping -- but it loses nothing either, and
  it marches in place.  This gives that idling a cost, measured as L1 deviation
  from the default joint angles so a single badly placed leg is penalized as
  much as a whole gait cycle of small ones.
  """
  asset = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  deviation = torch.sum(
    torch.abs(
      asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    ),
    dim=1,
  )
  return deviation * _standing(env, command_name, command_threshold)
