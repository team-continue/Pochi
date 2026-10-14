"""Reward kernels for the independent LP-ACRL task (arXiv:2601.17428)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pochi_rl.lp_acrl.physics import invalid_physics_mask

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

ROBOT = SceneEntityCfg("robot")


def assert_valid_physics(env: ManagerBasedRlEnv) -> torch.Tensor:
  """Abort invalid training steps before any reward or episode credit is added.

  This zero-valued term must be first and have nonzero weight, because mjlab
  skips zero-weight terms. Evaluation records failures using the same mask.
  """
  term = env.command_manager.get_term("base_velocity")
  if not term._evaluation:
    invalid = invalid_physics_mask(env.scene["robot"].data)
    if invalid.any():
      ids = invalid.nonzero(as_tuple=False).flatten()
      env._lp_acrl_physics_fault_ids = ids.detach().clone()
      raise ValueError(
        "LP-ACRL invalid physics before reward/PPO/LP credit; "
        f"environment IDs: {ids[:32].tolist()}"
      )
  return torch.zeros(env.num_envs, device=env.device)


def track_linear_velocity(
  env: ManagerBasedRlEnv,
  command_name: str = "base_velocity",
  std: float = 0.5,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  velocity = env.scene[asset_cfg.name].data.root_link_lin_vel_b[:, :2]
  return torch.exp(-((command[:, :2] - velocity) ** 2).sum(dim=1) / std**2)


def track_yaw_velocity(
  env: ManagerBasedRlEnv,
  command_name: str = "base_velocity",
  std: float = 0.5,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  velocity = env.scene[asset_cfg.name].data.root_link_ang_vel_b[:, 2]
  return torch.exp(-((command[:, 2] - velocity) ** 2) / std**2)


def target_position_rate(env: ManagerBasedRlEnv, action_scale: float = 0.25):
  """Squared change of joint targets in radians, including the action scale."""
  delta = env.action_manager.action - env.action_manager.prev_action
  return (delta * action_scale).square().sum(dim=1)


def touchdown_air_time(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  command_name: str = "base_velocity",
  threshold: float = 0.5,
) -> torch.Tensor:
  """Pay (last air time - 0.5 s) only on touchdown; no scripted contact phase."""
  sensor = env.scene[sensor_name]
  last_air_time = sensor.data.last_air_time
  assert last_air_time is not None
  touchdown = sensor.compute_first_contact(dt=env.step_dt)
  reward = ((last_air_time - threshold) * touchdown).sum(dim=1)
  command = env.command_manager.get_command(command_name)
  moving = torch.norm(command[:, :2], dim=1) + command[:, 2].abs() > 0.05
  return reward * moving


def undesired_contact_count(
  env: ManagerBasedRlEnv,
  sensor_name: str = "limb_ground_contact",
  force_threshold: float = 1.0,
) -> torch.Tensor:
  data = env.scene[sensor_name].data
  if data.force_history is not None:
    return (data.force_history.norm(dim=-1) > force_threshold).any(dim=-1).sum(dim=-1)
  assert data.found is not None
  return (data.found > 0).sum(dim=-1)


def outside_task_terrain(
  env: ManagerBasedRlEnv,
  half_extent: float = 11.0,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  """End at tile edges so an episode cannot enter another task's terrain."""
  position = env.scene[asset_cfg.name].data.root_link_pos_w[:, :2]
  relative = position - env.scene.env_origins[:, :2]
  return (relative.abs() > half_extent).any(dim=1)
