"""Opt-in blind stride-center shaping and supported-stop damping.

These are geometric engineering costs, not a prescribed phase or proof of gait.
RewardManager applies the external weight and control dt exactly once.
"""

from __future__ import annotations

import math
from functools import lru_cache
from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pochi_rl.mjlab.lp_acrl_balanced_rewards import _contacts, _standing
from pochi_rl.mjlab.lp_acrl_blind_v1_rewards import standing_joint_velocity_bounded
from pochi_rl.mjlab.lp_acrl_qualified_swing_rewards import _validate_sites
from pochi_rl.robot import BASE_BODY, DEFAULT_JOINT_POS, FOOT_SITES, JOINT_NAMES

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.managers.reward_manager import RewardTermCfg

# FL, FR, RL, RR: nominal asset/default-pose foot-site X minus whole-robot COM X.
# Kept explicit in task configuration/checkpoint identity, and checked against
# native MuJoCo FK in each reward constructor (the CPU derivation is cached).
NOMINAL_FOOT_COM_X = (
  0.2556439192223622,
  0.2556548709230452,
  -0.25633672816459196,
  -0.25633866081765366,
)


@lru_cache(maxsize=1)
def nominal_stride_center_offsets() -> tuple[float, ...]:
  """Derive named offsets from actual nominal assets, with no simulation steps.

  Payload and link masses come from pochi.xml; all named joints use
  DEFAULT_JOINT_POS. Root orientation is identity. The reference is the root
  body's *subtree* COM, not that body's own COM. Training mass randomization is
  intentionally absent from the nominal target and present in runtime COM.
  """
  import mujoco

  from pochi_rl.mjlab.entity_cfg import POCHI_XML

  model = mujoco.MjModel.from_xml_path(str(POCHI_XML))
  data = mujoco.MjData(model)
  joint_ids = [model.joint(name).id for name in JOINT_NAMES]
  site_ids = [model.site(name).id for name in FOOT_SITES]
  root = model.body(BASE_BODY).id
  data.qpos[:] = model.qpos0
  data.qpos[:3] = 0.0
  data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
  data.qpos[model.jnt_qposadr[joint_ids]] = [
    DEFAULT_JOINT_POS[name] for name in JOINT_NAMES
  ]
  mujoco.mj_kinematics(model, data)
  mujoco.mj_comPos(model, data)
  return tuple(
    float(x) for x in data.site_xpos[site_ids, 0] - data.subtree_com[root, 0]
  )


def _validate_parameters(tau_s: float, offsets: tuple[float, ...]) -> None:
  if not math.isfinite(tau_s) or tau_s <= 0:
    raise ValueError("stride-center tau_s must be finite and positive")
  if len(offsets) != 4 or not all(math.isfinite(x) for x in offsets):
    raise ValueError("Require four finite nominal fore-aft offsets")


class moving_top_two_stride_center:
  """Mean of the largest two squared per-foot EMA fore-aft offset errors.

  First project each current foot-minus-wholeCOM vector into the normalized
  world-horizontal root-forward direction, then subtract the named nominal
  target and update an exact exponential EMA. This ordering is yaw/translation
  invariant. Stops and episode resets clear error memory to zero (nominal).

  No contact gate, phase, commanded foot trajectory or height scan enters this
  term. It does not by itself demand lifting or balanced forces. Necessary
  foothold adaptations on terrain can conflict with a fixed nominal target.
  The EMA is reward memory, not an added observation.
  """

  def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRlEnv):
    params = cfg.params
    self.tau_s = params.get("tau_s", 0.25)
    self.nominal_offsets = tuple(params.get("nominal_offsets_m", NOMINAL_FOOT_COM_X))
    _validate_parameters(self.tau_s, self.nominal_offsets)
    if not math.isfinite(env.step_dt) or env.step_dt <= 0:
      raise ValueError("control dt must be finite and positive")
    _validate_sites(env, params.get("asset_cfg"))
    derived = nominal_stride_center_offsets()
    if not all(
      math.isclose(given, actual, rel_tol=0, abs_tol=1e-9)
      for given, actual in zip(self.nominal_offsets, derived, strict=True)
    ):
      raise ValueError("Nominal targets must match current robot assets/default pose")
    with torch.inference_mode(False):
      self.target_m = torch.tensor(self.nominal_offsets, device=env.device)
      self.ema_error_m = torch.zeros((env.num_envs, 4), device=env.device)
      self.current_error_m = torch.zeros_like(self.ema_error_m)
      self.geometry_valid = torch.zeros(
        env.num_envs, dtype=torch.bool, device=env.device
      )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    ids = slice(None) if env_ids is None else env_ids
    self.ema_error_m[ids] = 0
    self.current_error_m[ids] = 0
    self.geometry_valid[ids] = False

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    asset_cfg: SceneEntityCfg,
    tau_s: float = 0.25,
    nominal_offsets_m: tuple[float, ...] = NOMINAL_FOOT_COM_X,
    command_name: str = "base_velocity",
  ) -> torch.Tensor:
    if tau_s != self.tau_s or tuple(nominal_offsets_m) != self.nominal_offsets:
      raise ValueError("Stride-center parameters cannot change after construction")
    if not math.isfinite(env.step_dt) or env.step_dt <= 0:
      raise ValueError("control dt must be finite and positive")
    _validate_sites(env, asset_cfg)
    robot = env.scene[asset_cfg.name]
    feet = robot.data.site_pos_w[:, asset_cfg.site_ids]
    # root_com_pos_w is only the base body's COM and must not be substituted.
    com = env.sim.data.subtree_com[:, robot.indexing.root_body_id]
    quat = robot.data.root_link_quat_w
    if feet.shape != (env.num_envs, 4, 3) or com.shape != (env.num_envs, 3):
      raise ValueError("Require four named sites and per-world whole-subtree COM")
    if quat.shape != (env.num_envs, 4):
      raise ValueError("Require per-world wxyz root quaternion")
    norm = torch.linalg.vector_norm(quat, dim=1)
    valid_quat = torch.isfinite(quat).all(1) & torch.isfinite(norm) & (norm > 1e-8)
    identity = quat.new_tensor((1.0, 0.0, 0.0, 0.0))
    safe = torch.where(
      valid_quat[:, None], quat / norm.clamp_min(1e-8)[:, None], identity
    )
    w, x, y, z = safe.unbind(1)
    forward = torch.stack((1 - 2 * (y * y + z * z), 2 * (x * y + w * z)), dim=1)
    horizontal_norm = torch.linalg.vector_norm(forward, dim=1)
    valid = (
      valid_quat
      & (horizontal_norm > 1e-6)
      & torch.isfinite(feet).all(dim=(1, 2))
      & torch.isfinite(com).all(1)
    )
    direction = forward / horizontal_norm.clamp_min(1e-6)[:, None]
    relative = torch.where(valid[:, None, None], feet - com[:, None, :], 0.0)
    offset = (relative[:, :, :2] * direction[:, None, :]).sum(2)
    raw_error = offset - self.target_m
    # Finite but nonsensical coordinates must not overflow the squared cost.
    valid = valid & torch.isfinite(raw_error.square()).all(dim=1)
    error = torch.where(valid[:, None], raw_error, 0.0)
    moving_valid = (~_standing(env, command_name)) & valid
    alpha = -math.expm1(-env.step_dt / tau_s)
    updated = self.ema_error_m + alpha * (error - self.ema_error_m)
    self.ema_error_m.copy_(torch.where(moving_valid[:, None], updated, 0.0))
    self.current_error_m.copy_(error)
    self.geometry_valid.copy_(valid)
    # Degenerate/nonfinite geometry clears only that world's memory and returns
    # zero, avoiding NaN propagation during evaluation. Physics guard and fall
    # termination remain responsible for invalid/fallen states.
    return self.ema_error_m.square().topk(2, dim=1).values.mean(dim=1)


def standing_joint_velocity_bounded_all_contact(
  env: ManagerBasedRlEnv,
  command_name: str = "base_velocity",
  std: float = 0.10,
  sensor_name: str = "feet_ground_contact",
) -> torch.Tensor:
  """Existing quiet-stop cost, active only when all four feet have contact.

  Entire-body damping is released during a missing-foot recovery; the unchanged
  missing-contact and posture terms still apply. All-contact output is exactly
  the inherited bounded kernel, including its std and exact-zero command gate.
  """
  original = standing_joint_velocity_bounded(env, command_name, std)
  return original * _contacts(env, sensor_name).all(dim=1)
