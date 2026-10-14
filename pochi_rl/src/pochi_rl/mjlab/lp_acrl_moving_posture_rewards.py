"""Optional moving-pose regularization; no phases, contacts or action overrides."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pochi_rl.mjlab.lp_acrl_balanced_rewards import _standing
from pochi_rl.robot import JOINT_NAMES

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def moving_default_joint_deviation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str = "base_velocity",
) -> torch.Tensor:
  """Moving-only mean12 squared deviation from each environment's default pose.

  The output is unweighted rad²: RewardManager applies the external weight and
  control dt once. The term is stateless and treats every named joint equally.
  It discourages persistent pose shifts but does not enforce a support split or
  guarantee a gait; legitimate swing and terrain adaptation also incur cost.
  """
  if (
    not isinstance(asset_cfg, SceneEntityCfg)
    or not asset_cfg.preserve_order
    or tuple(asset_cfg.joint_names or ()) != JOINT_NAMES
  ):
    raise ValueError("Require ordered JOINT_NAMES with preserve_order=True")
  entity = env.scene[asset_cfg.name]
  indices = asset_cfg.joint_ids
  names = entity.joint_names
  selected = (
    names[indices] if isinstance(indices, slice) else [names[i] for i in indices]
  )
  if tuple(selected) != JOINT_NAMES:
    raise ValueError("Resolved joint_ids must select all12 named joints in order")
  data = entity.data
  position = data.joint_pos[:, indices]
  default = data.default_joint_pos[:, indices]
  if position.shape != (env.num_envs, 12) or default.shape not in (
    (env.num_envs, 12),
    (1, 12),
  ):
    raise ValueError("Require all12 joint positions and broadcastable default pose")
  cost = (position - default).square().mean(dim=1)
  return cost * ~_standing(env, command_name)
