"""Conservative physics validity checks, independent of simulator internals.

These limits detect numerical failure, not successful locomotion. They are
intentionally much wider than the task commands and Pochi motor limits. An
invalid training state must abort before its reward enters PPO or LP statistics;
valid rewards and states are never clipped or repaired by these functions.
"""

from __future__ import annotations

from typing import Any

import torch

PHYSICS_LIMITS = {
  "joint_speed_rad_s": 200.0,
  "joint_acceleration_rad_s2": 1.0e6,
  "base_linear_speed_m_s": 20.0,
  "base_angular_speed_rad_s": 50.0,
}

PHYSICS_FIELDS = (
  "root_link_pos_w",
  "root_link_quat_w",
  "root_link_lin_vel_b",
  "root_link_ang_vel_b",
  "joint_pos",
  "joint_vel",
  "joint_acc",
)

# Real articulated entities expose these combined pose/velocity tensors for
# every body. Minimal callers may provide only the required root/joint fields.
PHYSICS_BODY_FIELDS = ("body_link_pose_w", "body_link_vel_w")


def physics_fault_masks(data: Any) -> dict[str, torch.Tensor]:
  """Return named boolean masks [num_envs] without changing any state.

  ``data`` can be simulator entity data or a namespace exposing the fields in
  ``PHYSICS_FIELDS``. Joint limits apply per joint; base limits use vector norms.
  Values exactly on a threshold remain valid.
  """
  masks = {
    f"nonfinite/{name}": ~torch.isfinite(getattr(data, name)).flatten(1).all(-1)
    for name in PHYSICS_FIELDS
  }
  for name in PHYSICS_BODY_FIELDS:
    value = getattr(data, name, None)
    if value is not None:
      masks[f"nonfinite/{name}"] = ~torch.isfinite(value).flatten(1).all(-1)
  masks.update(
    {
      "joint_speed": (data.joint_vel.abs() > PHYSICS_LIMITS["joint_speed_rad_s"]).any(
        -1
      ),
      "joint_acceleration": (
        data.joint_acc.abs() > PHYSICS_LIMITS["joint_acceleration_rad_s2"]
      ).any(-1),
      "base_linear_speed": data.root_link_lin_vel_b.norm(dim=-1)
      > PHYSICS_LIMITS["base_linear_speed_m_s"],
      "base_angular_speed": data.root_link_ang_vel_b.norm(dim=-1)
      > PHYSICS_LIMITS["base_angular_speed_rad_s"],
    }
  )
  return masks


def invalid_physics_mask(data: Any) -> torch.Tensor:
  """Return True for nonfinite or physically impossible states, shape [N]."""
  return torch.stack(tuple(physics_fault_masks(data).values()), dim=0).any(dim=0)
