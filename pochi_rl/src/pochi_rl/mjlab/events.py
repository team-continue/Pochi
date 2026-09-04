"""Custom mjlab event terms for Pochi.

The velocity task uses stock mjlab terms throughout; only the stand-up task
needs one of its own, because there is no stock way to say "put the robot in
*this* pose" rather than "near its default one".
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import quat_from_euler_xyz, sample_uniform

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

ROBOT = SceneEntityCfg("robot")


def reset_collapsed(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | None,
  joint_pos: tuple[float, ...],
  base_height: float,
  joint_pos_range: tuple[float, float],
  joint_vel_range: tuple[float, float],
  height_range: tuple[float, float],
  roll_pitch_range: tuple[float, float],
  asset_cfg: SceneEntityCfg = ROBOT,
) -> None:
  """Lay the robot on the floor in the pose it collapses into with motors off.

  ``joint_pos`` is given in the order of ``asset_cfg.joint_names``, so that cfg
  must be built with ``preserve_order=True``.  Yaw is randomised over the full
  circle and the rest jittered by the ranges, so the policy cannot memorise one
  exact starting configuration.
  """
  if env_ids is None:
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.int)
  n = len(env_ids)

  asset: Entity = env.scene[asset_cfg.name]
  joint_ids: torch.Tensor | slice = (
    torch.tensor(asset_cfg.joint_ids, device=env.device)
    if isinstance(asset_cfg.joint_ids, list)
    else asset_cfg.joint_ids
  )

  pose = torch.tensor(joint_pos, device=env.device, dtype=torch.float32)
  q = pose.unsqueeze(0).repeat(n, 1)
  q += sample_uniform(*joint_pos_range, q.shape, env.device)
  limits = asset.data.soft_joint_pos_limits
  assert limits is not None
  bounds = limits[env_ids][:, joint_ids]
  q = q.clamp(bounds[..., 0], bounds[..., 1])
  qd = sample_uniform(*joint_vel_range, q.shape, env.device)
  asset.write_joint_state_to_sim(q, qd, env_ids=env_ids, joint_ids=joint_ids)

  root = asset.data.default_root_state[env_ids].clone()
  root[:, 0:3] = env.scene.env_origins[env_ids]
  root[:, 2] += base_height + sample_uniform(*height_range, (n,), env.device)
  roll = sample_uniform(*roll_pitch_range, (n,), env.device)
  pitch = sample_uniform(*roll_pitch_range, (n,), env.device)
  yaw = sample_uniform(-torch.pi, torch.pi, (n,), env.device)
  root[:, 3:7] = quat_from_euler_xyz(roll, pitch, yaw)
  root[:, 7:] = 0.0
  asset.write_root_state_to_sim(root, env_ids=env_ids)
