"""Left/right reflection for the blind policy's48D proprioceptive observation."""

from __future__ import annotations

import torch

BLIND_OBSERVATION_TERMS = (
  "base_lin_vel", "base_ang_vel", "projected_gravity", "velocity_commands",
  "joint_pos_rel", "joint_vel_rel", "last_action",
)


def mirror_joint_vectors(values: torch.Tensor) -> torch.Tensor:
  """Exchange FL/FR and RL/RR; negate X-axis hip rolls, preserve Y joints."""
  if values.shape[-1] != 12:
    raise ValueError("Pochi reflection requires12 ordered joint/action values")
  permutation = torch.tensor(
    [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],
    device=values.device, dtype=torch.long,
  )
  signs = values.new_tensor([-1, 1, 1] * 4)
  return values.index_select(-1, permutation) * signs


def mirror_blind_observations(values: torch.Tensor) -> torch.Tensor:
  """Reflect raw observations before the actor/critic's own normalizers.

  Linear velocity/gravity are polar vectors; angular velocity is an axial
  vector. Commands contain vx,vy,wz. Joint-relative default poses are symmetric.
  """
  if values.shape[-1] != 48:
    raise ValueError("Blind reflection requires exactly48 observation values")
  polar = values.new_tensor([1, -1, 1])
  axial = values.new_tensor([-1, 1, -1])
  commands = values.new_tensor([1, -1, -1])
  return torch.cat(
    (
      values[..., :3] * polar,
      values[..., 3:6] * axial,
      values[..., 6:9] * polar,
      values[..., 9:12] * commands,
      mirror_joint_vectors(values[..., 12:24]),
      mirror_joint_vectors(values[..., 24:36]),
      mirror_joint_vectors(values[..., 36:48]),
    ), dim=-1,
  )


def augment_blind_symmetry(env, obs=None, actions=None):
  """RSL-RL callback: original batch followed by reflected batch.

  Both observation-only and action-only calls are required by mirror loss.
  ``env`` is provided by RSL-RL; validation of layout happens in the runner.
  """
  del env
  if obs is not None:
    mirrored = obs.clone()
    for name in obs.keys():
      if name not in ("actor", "critic"):
        raise ValueError(f"Unsupported blind observation group: {name}")
      mirrored[name] = mirror_blind_observations(obs[name])
    obs = torch.cat((obs, mirrored), dim=0)
  if actions is not None:
    actions = torch.cat((actions, mirror_joint_vectors(actions)), dim=0)
  return obs, actions
