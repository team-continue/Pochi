"""Prospective qualified-swing occupancy cost; no task registers this module.

The target is an engineering choice, not a gait guarantee. In particular,
synchronous and staggered versions of a periodic per-foot swing can have the
same long-run cost. Tracking, support, slip and independent gait checks remain
necessary. RewardManager applies the external weight (proposed -8) and dt once.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pochi_rl.mjlab.lp_acrl_balanced_rewards import _standing
from pochi_rl.robot import FOOT_GEOMS, FOOT_SITES

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg


def _validate_parameters(tau_s: float, target: float, min_air: float, min_lift: float):
  for name, value in (("tau_s", tau_s), ("min_air", min_air), ("min_lift", min_lift)):
    if not math.isfinite(value) or value <= 0:
      raise ValueError(f"{name} must be finite and positive")
  if not math.isfinite(target) or not 0 < target < 1:
    raise ValueError("target must be finite and strictly between zero and one")


def _validate_sites(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> None:
  if (
    not isinstance(asset_cfg, SceneEntityCfg)
    or not asset_cfg.preserve_order
    or asset_cfg.site_names is None
    or tuple(asset_cfg.site_names) != FOOT_SITES
  ):
    raise ValueError("Require FOOT_SITES in named order with preserve_order=True")
  entity = env.scene[asset_cfg.name]
  indices = asset_cfg.site_ids
  names = entity.site_names
  selected = (
    names[indices] if isinstance(indices, slice) else [names[i] for i in indices]
  )
  if tuple(selected) != FOOT_SITES:
    raise ValueError("Resolved site_ids must select FOOT_SITES in named order")


class moving_qualified_swing_occupancy:
  """Mean squared error of each foot's EMA qualified occupancy from ``target``.

  A foot qualifies after continuous air >= ``min_air`` and peak world-Z lift
  >= ``min_lift`` above its last observed contact site. This displacement is not
  terrain clearance. Qualification persists while descending until touchdown.
  Sensor air time, when available, also catches contacts between reward calls;
  an unobserved touchdown invalidates the old height reference conservatively.

  Stops and episode resets clear every swing/reference state and initialize the
  EMA at the neutral target. A foot initially in air cannot qualify. All state
  tensors are allocated in the constructor as normal tensors, even if its caller
  uses inference mode; subsequent calls update those same buffers in place.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    p = cfg.params
    tau_s, target = p.get("tau_s", 1.0), p.get("target", 0.20)
    min_air, min_lift = p.get("min_air", 0.06), p.get("min_lift", 0.025)
    _validate_parameters(tau_s, target, min_air, min_lift)
    if not math.isfinite(env.step_dt) or env.step_dt <= 0:
      raise ValueError("control dt must be finite and positive")
    _validate_sites(env, p.get("asset_cfg"))
    self.target = target
    with torch.inference_mode(False):
      self.occupancy = torch.full((env.num_envs, 4), target, device=env.device)
      self.air_s = torch.zeros_like(self.occupancy)
      self.contact_height = torch.zeros_like(self.occupancy)
      self.peak_lift = torch.zeros_like(self.occupancy)
      self.has_reference = torch.zeros_like(self.occupancy, dtype=torch.bool)
      self.qualified = torch.zeros_like(self.has_reference)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.occupancy[ids] = self.target
    self.air_s[ids] = 0
    self.contact_height[ids] = 0
    self.peak_lift[ids] = 0
    self.has_reference[ids] = False
    self.qualified[ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    sensor_name: str = "feet_ground_contact",
    command_name: str = "base_velocity",
    tau_s: float = 1.0,
    target: float = 0.20,
    min_air: float = 0.06,
    min_lift: float = 0.025,
  ) -> torch.Tensor:
    _validate_parameters(tau_s, target, min_air, min_lift)
    if target != self.target:
      raise ValueError("target cannot change without constructing a new reward")
    _validate_sites(env, asset_cfg)
    sensor = env.scene[sensor_name]
    names = [name.rsplit("/", 1)[-1] for name in sensor.primary_names]
    if len(names) != 4 or set(names) != set(FOOT_GEOMS):
      raise ValueError("Require one contact primary per named Pochi foot")
    order = [names.index(name) for name in FOOT_GEOMS]
    data = sensor.data
    shape = (env.num_envs, 4)
    if data.found is None or data.found.shape != shape:
      raise ValueError("Require exactly four foot contact slots")
    contact = data.found[:, order] > 0
    moving = (~_standing(env, command_name))[:, None]
    foot_z = env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids, 2]
    air = torch.where(contact | ~moving, 0.0, self.air_s + env.step_dt)
    sensor_air = getattr(data, "current_air_time", None)
    missed_touchdown = torch.zeros_like(contact)
    if sensor_air is not None:
      if sensor_air.shape != shape:
        raise ValueError("Require four per-foot current_air_time values")
      sensor_air = sensor_air[:, order]
      # Air must advance by a full control step after the first observed air.
      # A reset can leave the same small duration at two successive samples.
      # Ten microseconds accommodates float32 elapsed-time arithmetic at 20 s.
      missed_touchdown = (
        ~contact
        & (self.air_s > 0)
        & (sensor_air < self.air_s + env.step_dt - 1.0e-5)
      )
      air = torch.minimum(air, sensor_air.clamp_min(0))
    referenced = self.has_reference & ~missed_touchdown
    peak = torch.maximum(self.peak_lift, (foot_z - self.contact_height).clamp_min(0))
    peak = torch.where(contact | ~moving | missed_touchdown, 0.0, peak)
    # The 1 us / 1 um tolerance preserves threshold equality under float32 sums.
    acquired = referenced & (air >= min_air - 1.0e-6) & (peak >= min_lift - 1.0e-6)
    qualified = moving & ~contact & ~missed_touchdown & (self.qualified | acquired)
    alpha = -math.expm1(-env.step_dt / tau_s)
    updated = self.occupancy + alpha * (
      qualified.to(self.occupancy.dtype) - self.occupancy
    )
    self.occupancy.copy_(torch.where(moving, updated, target))
    self.air_s.copy_(air)
    self.peak_lift.copy_(peak)
    self.qualified.copy_(qualified)
    self.contact_height.copy_(
      torch.where(moving, torch.where(contact, foot_z, self.contact_height), 0.0)
    )
    self.has_reference.copy_(moving & (contact | referenced))
    return (self.occupancy - target).square().mean(dim=1)
