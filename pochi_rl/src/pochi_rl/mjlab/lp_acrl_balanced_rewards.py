"""Blind Pochi standing and phase-free four-leg locomotion rewards.

These kernels are opt-in adaptations for the blind balanced task. They require
joint/base state, foot sites and the existing contact sensor; they never read a
height scan. Positive touchdown credit replaces the paper's negative reward for
short swings. All scalar outputs are unweighted; RewardManager applies weight
and control dt once.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pochi_rl.robot import FOOT_GEOMS

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg


def _standing(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Only exact-zero commands request standing, including stop/start segments."""
  command = env.command_manager.get_command(command_name)
  return command.abs().amax(dim=1) <= 1.0e-6


def _contact_indices(env: ManagerBasedRlEnv, sensor_name: str) -> list[int]:
  sensor = env.scene[sensor_name]
  names = [name.rsplit("/", 1)[-1] for name in sensor.primary_names]
  if len(names) != 4 or set(names) != set(FOOT_GEOMS):
    raise ValueError("Balanced rewards require one contact primary per Pochi foot")
  return [names.index(name) for name in FOOT_GEOMS]


def _contacts(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor = env.scene[sensor_name]
  found = sensor.data.found
  if found is None or found.shape != (env.num_envs, 4):
    raise ValueError("Balanced rewards require four foot contact slots")
  return found[:, _contact_indices(env, sensor_name)] > 0


def standing_posture(
  env: ManagerBasedRlEnv,
  command_name: str = "base_velocity",
  std: float = 0.20,
) -> torch.Tensor:
  """Soft default-pose reward; uneven ground may require small joint deviations."""
  data = env.scene["robot"].data
  error = (data.joint_pos - data.default_joint_pos).square().mean(dim=1)
  return torch.exp(-error / std**2) * _standing(env, command_name)


def standing_joint_velocity(
  env: ManagerBasedRlEnv,
  command_name: str = "base_velocity",
) -> torch.Tensor:
  """Mean squared joint speed while stopped; configure with negative weight."""
  speed = env.scene["robot"].data.joint_vel.square().mean(dim=1)
  return speed * _standing(env, command_name)


def standing_missing_contacts(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  command_name: str = "base_velocity",
) -> torch.Tensor:
  """Fraction of feet off the ground at zero command (negative weight)."""
  missing = (~_contacts(env, sensor_name)).float().mean(dim=1)
  return missing * _standing(env, command_name)


def contact_slip(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  """Mean planar foot speed squared in contact, at both zero and moving commands.

  ``asset_cfg`` must resolve FOOT_SITES with preserve_order=True. A swinging foot
  is not penalized, and the denominator stays four instead of rewarding fewer
  support feet by changing normalization.
  """
  velocity = env.scene[asset_cfg.name].data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
  return (velocity.square().sum(dim=-1) * _contacts(env, sensor_name)).mean(dim=1)


def upright_error(env: ManagerBasedRlEnv) -> torch.Tensor:
  """World-up roll/pitch error, deliberately independent of terrain sensing."""
  gravity = env.scene["robot"].data.projected_gravity_b
  # Include Z so an upside-down robot cannot earn the same value as upright.
  return gravity[:, :2].square().sum(dim=1) + (gravity[:, 2] + 1).square()


def support_height_error(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg,
  sensor_name: str = "feet_ground_contact",
  target_height: float = 0.3114,
  std: float = 0.08,
) -> torch.Tensor:
  """Bounded root height cost above the mean of feet currently in contact.

  This is a support-height proxy, not a terrain measurement. It follows stairs
  and slopes instead of comparing with the fixed spawn origin. The nominal
  root-to-foot-site distance is 0.3114 m; the pad-ground base height is 0.3173 m.
  No support reference is defined in flight, so this term is zero then.
  """
  data = env.scene[asset_cfg.name].data
  contact = _contacts(env, sensor_name)
  count = contact.sum(dim=1)
  foot_z = data.site_pos_w[:, asset_cfg.site_ids, 2]
  support_z = (foot_z * contact).sum(dim=1) / count.clamp_min(1)
  height = data.root_link_pos_w[:, 2] - support_z
  error = ((height - target_height) / std).square().clamp_max(4.0)
  return error * (count > 0)


def prolonged_foot_contact(
  env: ManagerBasedRlEnv,
  sensor_name: str = "feet_ground_contact",
  command_name: str = "base_velocity",
  max_stance: float = 0.6,
  max_air: float = 0.4,
) -> torch.Tensor:
  """Penalize each persistently planted or hovering foot while moving.

  The sensor owns per-foot dwell clocks; there is no desired phase, contact
  sequence, or diagonal pairing. Cost ramps from zero at the limit to one at
  twice the limit for each foot. Zero commands disable it immediately.
  """
  data = env.scene[sensor_name].data
  if data.current_contact_time is None or data.current_air_time is None:
    raise ValueError("Balanced rewards require contact track_air_time=True")
  stance = ((data.current_contact_time - max_stance) / max_stance).clamp(0, 1)
  air = ((data.current_air_time - max_air) / max_air).clamp(0, 1)
  return (stance + air).mean(dim=1) * (~_standing(env, command_name))


class balanced_touchdown:
  """Nonnegative credit for a real, reasonably short lift followed by contact.

  Lift is measured against that foot's last contact-site height. It is only a
  displacement proxy: it does not claim obstacle clearance ahead of the foot.
  Higher terrain steps are allowed without an upper lift penalty. A touchdown
  without an observed prior contact, including initial spawn contact, earns no
  credit. Buffers are allocated outside inference mode and reset per episode.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.contact_height = torch.zeros((env.num_envs, 4), device=env.device)
    self.peak_lift = torch.zeros_like(self.contact_height)
    self.has_reference = torch.zeros_like(self.contact_height, dtype=torch.bool)
    params = cfg.params
    min_air = params.get("min_air", 0.06)
    full_air = params.get("full_air", 0.10)
    long_air = params.get("long_air", 0.4)
    max_air = params.get("max_air", 0.8)
    if not 0 <= min_air < full_air <= long_air < max_air:
      raise ValueError("Require 0 <= min_air < full_air <= long_air < max_air")
    if params.get("target_lift", 0.035) <= 0:
      raise ValueError("target_lift must be positive")

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.contact_height[ids] = 0
    self.peak_lift[ids] = 0
    self.has_reference[ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    sensor_name: str = "feet_ground_contact",
    command_name: str = "base_velocity",
    target_lift: float = 0.035,
    min_air: float = 0.06,
    full_air: float = 0.10,
    long_air: float = 0.4,
    max_air: float = 0.8,
  ) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    indices = _contact_indices(env, sensor_name)
    contact = _contacts(env, sensor_name)
    last_air_time = sensor.data.last_air_time
    if last_air_time is None:
      raise ValueError("Balanced rewards require contact track_air_time=True")
    duration = last_air_time[:, indices]
    touchdown = sensor.compute_first_contact(dt=env.step_dt)[:, indices]
    moving = ~_standing(env, command_name)
    foot_z = env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids, 2]
    lift = (foot_z - self.contact_height).clamp_min(0)
    peak = torch.maximum(self.peak_lift, lift)
    # Short chatter and very long hovering both earn zero, never negative credit.
    duration_score = ((duration - min_air) / (full_air - min_air)).clamp(0, 1)
    duration_score *= ((max_air - duration) / (max_air - long_air)).clamp(0, 1)
    lift_score = (peak / target_lift).clamp(0, 1)
    valid = touchdown & self.has_reference & moving[:, None]
    reward = (duration_score * lift_score * valid).mean(dim=1)

    self.contact_height.copy_(torch.where(contact, foot_z, self.contact_height))
    self.peak_lift.copy_(torch.where(contact | ~moving[:, None], 0.0, peak))
    # A foot that hovered throughout a stop must establish a new contact before
    # earning another swing reward; old lift cannot leak across stop/start.
    self.has_reference.copy_(
      torch.where(moving[:, None], self.has_reference | contact, contact)
    )
    return reward
