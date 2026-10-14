"""Optional low-clearance transfer cost; this module registers no task.

The cost is horizontal foot speed times squared relative lift deficit, averaged
over all four feet. RewardManager applies any external negative weight and dt.
It can also be reduced by planting or slowing a foot, so it is not a swing or
gait guarantee. World-Z displacement from last observed contact is not terrain
clearance and can penalize a low descending step before its next touchdown.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pochi_rl.mjlab.lp_acrl_balanced_rewards import _standing
from pochi_rl.mjlab.lp_acrl_qualified_swing_rewards import _validate_sites
from pochi_rl.robot import FOOT_GEOMS

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg


def _validate_target(target_lift: float) -> None:
  if not math.isfinite(target_lift) or target_lift <= 0:
    raise ValueError("target_lift must be finite and positive")


class moving_dense_clearance:
  """Penalize horizontal transfer below the current lift target while airborne.

  Every actual contact refreshes that foot's height reference. An initially
  airborne foot has no reference and incurs no cost. Stops and episode resets
  erase the reference; a contact observed after restarting is required. Where
  sensor air time is available, an intervening unseen touchdown invalidates the
  old reference. Without that clock only sampled contacts can be detected.

  There is no minimum air duration, peak-height memory, qualification latch or
  phase clock. Persistent state is allocated as normal tensors in the constructor
  and remains safe across inference-mode stepping and ordinary in-place resets.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    _validate_target(cfg.params.get("target_lift", 0.035))
    if not math.isfinite(env.step_dt) or env.step_dt <= 0:
      raise ValueError("control dt must be finite and positive")
    _validate_sites(env, cfg.params.get("asset_cfg"))
    with torch.inference_mode(False):
      self.contact_height = torch.zeros((env.num_envs, 4), device=env.device)
      self.air_s = torch.zeros_like(self.contact_height)
      self.has_reference = torch.zeros_like(self.contact_height, dtype=torch.bool)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.contact_height[ids] = 0
    self.air_s[ids] = 0
    self.has_reference[ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    sensor_name: str = "feet_ground_contact",
    command_name: str = "base_velocity",
    target_lift: float = 0.035,
  ) -> torch.Tensor:
    _validate_target(target_lift)
    _validate_sites(env, asset_cfg)
    sensor = env.scene[sensor_name]
    names = [name.rsplit("/", 1)[-1] for name in sensor.primary_names]
    if len(names) != 4 or set(names) != set(FOOT_GEOMS):
      raise ValueError("Require one contact primary per named Pochi foot")
    order = [names.index(name) for name in FOOT_GEOMS]
    shape = (env.num_envs, 4)
    data = sensor.data
    if data.found is None or data.found.shape != shape:
      raise ValueError("Require exactly four foot contact slots")
    contact = data.found[:, order] > 0
    moving = (~_standing(env, command_name))[:, None]
    robot_data = env.scene[asset_cfg.name].data
    foot_z = robot_data.site_pos_w[:, asset_cfg.site_ids, 2]
    velocity_xy = robot_data.site_lin_vel_w[:, asset_cfg.site_ids, :2]
    if foot_z.shape != shape or velocity_xy.shape != (*shape, 2):
      raise ValueError("Require four ordered per-foot positions and velocities")

    air = torch.where(contact | ~moving, 0.0, self.air_s + env.step_dt)
    sensor_air = getattr(data, "current_air_time", None)
    missed_touchdown = torch.zeros_like(contact)
    if sensor_air is not None:
      if sensor_air.shape != shape:
        raise ValueError("Require four per-foot current_air_time values")
      sensor_air = sensor_air[:, order]
      missed_touchdown = (
        ~contact & (self.air_s > 0) & (sensor_air < self.air_s + env.step_dt - 1.0e-5)
      )
      air = torch.minimum(air, sensor_air.clamp_min(0))
    referenced = self.has_reference & ~missed_touchdown
    lift = foot_z - self.contact_height
    deficit = ((target_lift - lift) / target_lift).clamp(0, 1)
    cost = torch.linalg.vector_norm(velocity_xy, dim=-1) * deficit.square()
    cost = torch.where(moving & ~contact & referenced, cost, 0.0)

    self.air_s.copy_(air)
    self.contact_height.copy_(
      torch.where(moving, torch.where(contact, foot_z, self.contact_height), 0.0)
    )
    self.has_reference.copy_(moving & (contact | referenced))
    return cost.mean(dim=1)
