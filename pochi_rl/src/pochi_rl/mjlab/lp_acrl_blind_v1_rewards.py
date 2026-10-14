"""Opt-in v1 rewards for real foot cycles and quiet stops, without height scans.

The v0 kernels remain unchanged. Here, contact chatter cannot clear the moving
leg's age: only an observed lift, sufficient airborne time, and touchdown do.
All outputs are unweighted; RewardManager applies weight and control dt once.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pochi_rl.robot import FOOT_GEOMS

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg


def _standing(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_command(command_name)
  return command.abs().amax(dim=1) <= 1.0e-6


def standing_joint_velocity_bounded(
  env: ManagerBasedRlEnv,
  command_name: str = "base_velocity",
  std: float = 0.10,
) -> torch.Tensor:
  """Smooth stopped-joint cost in[0,1), scaled to a meaningful quiet-stance speed.

  This is mean(qdot²)/(mean(qdot²)+std²). The equivalent subtraction form stays
  finite even if an already-invalid, very large finite speed overflows squared
  float32 arithmetic. The separate physics guard still rejects that state.
  """
  if not math.isfinite(std) or std <= 0:
    raise ValueError("standing velocity std must be finite and positive")
  mean_square = env.scene["robot"].data.joint_vel.square().mean(dim=1)
  cost = 1.0 - std**2 / (mean_square + std**2)
  return cost * _standing(env, command_name)


def signed_translation_progress(
  env: ManagerBasedRlEnv,
  command_name: str = "base_velocity",
  min_command_speed: float = 0.05,
) -> torch.Tensor:
  """Bounded signed velocity along a meaningful planar command, in[-1,1].

  Credit saturates at the requested speed, so overspeed earns no extra credit.
  Small commands and pure yaw receive zero. Opposite symmetric oscillations
  cancel; stepwise clipping does not guarantee cancellation for every asymmetric
  zero-net oscillation. Gaussian tracking remains a separate reward.
  """
  if not math.isfinite(min_command_speed) or min_command_speed <= 0:
    raise ValueError("min_command_speed must be finite and positive")
  command = env.command_manager.get_command(command_name)[:, :2]
  velocity = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
  magnitude_squared = command.square().sum(dim=1)
  projection = (velocity * command).sum(dim=1)
  capped = torch.clamp(projection, -magnitude_squared, magnitude_squared)
  floor = min_command_speed**2
  return capped / magnitude_squared.clamp_min(floor) * (magnitude_squared >= floor)


class valid_foot_cycle_age:
  """Per-foot moving age since an observed, sufficiently large lift and landing.

  ``asset_cfg`` must resolve FOOT_SITES with preserve_order=True. Lift is measured
  above that foot's last contact-site height, a displacement proxy rather than a
  terrain-clearance measurement. No phase, diagonal pairing, or terrain ray is
  prescribed. At zero command, clocks and unfinished swing memory are cleared.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    self.age_s = torch.zeros((env.num_envs, 4), device=env.device)
    self.contact_height = torch.zeros_like(self.age_s)
    self.peak_lift = torch.zeros_like(self.age_s)
    self.has_reference = torch.zeros_like(self.age_s, dtype=torch.bool)
    self.valid_cycle = torch.zeros_like(self.has_reference)
    for name, default in (("min_air", 0.06), ("min_lift", 0.025), ("ramp_s", 0.8)):
      value = cfg.params.get(name, default)
      if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    grace = cfg.params.get("grace_s", 0.8)
    if not math.isfinite(grace) or grace < 0:
      raise ValueError("grace_s must be finite and nonnegative")

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.age_s[ids] = 0
    self.contact_height[ids] = 0
    self.peak_lift[ids] = 0
    self.has_reference[ids] = False
    self.valid_cycle[ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    sensor_name: str = "feet_ground_contact",
    command_name: str = "base_velocity",
    min_air: float = 0.06,
    min_lift: float = 0.025,
    grace_s: float = 0.8,
    ramp_s: float = 0.8,
  ) -> torch.Tensor:
    sensor = env.scene[sensor_name]
    names = [name.rsplit("/", 1)[-1] for name in sensor.primary_names]
    if len(names) != 4 or set(names) != set(FOOT_GEOMS):
      raise ValueError("Valid cycle age requires one contact primary per Pochi foot")
    order = [names.index(name) for name in FOOT_GEOMS]
    data = sensor.data
    if data.found is None or data.found.shape != (env.num_envs, 4):
      raise ValueError("Valid cycle age requires four contact slots")
    if data.last_air_time is None:
      raise ValueError("Valid cycle age requires contact track_air_time=True")
    contact = data.found[:, order] > 0
    touchdown = sensor.compute_first_contact(dt=env.step_dt)[:, order]
    foot_z = env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids, 2]
    peak = torch.maximum(self.peak_lift, (foot_z - self.contact_height).clamp_min(0))
    moving = ~_standing(env, command_name)
    valid = (
      moving[:, None]
      & self.has_reference
      & contact
      & touchdown
      & (data.last_air_time[:, order] >= min_air - 1.0e-6)
      & (peak >= min_lift - 1.0e-6)
    )
    self.valid_cycle.copy_(valid)
    self.age_s.copy_(
      torch.where(moving[:, None] & ~valid, self.age_s + env.step_dt, 0.0)
    )
    cost = ((self.age_s - grace_s) / ramp_s).clamp(0, 1).mean(dim=1)
    self.contact_height.copy_(torch.where(contact, foot_z, self.contact_height))
    self.peak_lift.copy_(torch.where(contact | ~moving[:, None], 0.0, peak))
    self.has_reference.copy_(
      torch.where(moving[:, None], self.has_reference | contact, contact)
    )
    return cost
