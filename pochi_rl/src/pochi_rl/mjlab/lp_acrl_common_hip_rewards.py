"""Moving common sagittal hip cost, independent of any prescribed gait phase."""

from __future__ import annotations

from typing import TYPE_CHECKING

from mjlab.managers.scene_entity_config import SceneEntityCfg

from pochi_rl.mjlab.lp_acrl_balanced_rewards import _standing

if TYPE_CHECKING:
  import torch
  from mjlab.envs import ManagerBasedRlEnv

HIP_PITCH_NAMES = ("FL_hip_pitch", "FR_hip_pitch", "RL_hip_pitch", "RR_hip_pitch")


def moving_common_hip_pitch_deviation(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  command_name: str = "base_velocity",
) -> torch.Tensor:
  """Square the mean four-hip residual from each world's default joint pose.

  Output is unweighted rad²; RewardManager applies weight and dt once. Opposite
  excursions cancel and knee/roll positions do not enter this term. Consequently
  it discourages a collective fore/aft hip offset but does not guarantee normal
  limb posture or balanced support. No contact, time, phase or reset state enters
  the calculation. Only exact-zero commands disable it, using the shared gate.
  """
  if (
    not isinstance(asset_cfg, SceneEntityCfg)
    or not asset_cfg.preserve_order
    or tuple(asset_cfg.joint_names or ()) != HIP_PITCH_NAMES
  ):
    raise ValueError("Require ordered HIP_PITCH_NAMES with preserve_order=True")
  entity = env.scene[asset_cfg.name]
  indices = asset_cfg.joint_ids
  names = entity.joint_names
  try:
    selected = (
      names[indices] if isinstance(indices, slice) else [names[i] for i in indices]
    )
  except (IndexError, TypeError) as exc:
    raise ValueError("Cannot resolve the four named hip-pitch joint indices") from exc
  if tuple(selected) != HIP_PITCH_NAMES:
    raise ValueError("Resolved joint_ids must select HIP_PITCH_NAMES in named order")
  data = entity.data
  width = len(names)
  if data.joint_pos.shape != (
    env.num_envs,
    width,
  ) or data.default_joint_pos.shape not in ((env.num_envs, width), (1, width)):
    raise ValueError("Require joint positions and per-world or broadcast defaults")
  residual = data.joint_pos[:, indices] - data.default_joint_pos[:, indices]
  if residual.shape != (env.num_envs, 4):
    raise ValueError("Require exactly four resolved hip-pitch residuals")
  cost = residual.mean(dim=1).square()
  return cost * ~_standing(env, command_name)
