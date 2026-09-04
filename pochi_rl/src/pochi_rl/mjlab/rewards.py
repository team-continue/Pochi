"""Small Pochi-specific reward helpers."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pochi_rl.control.standup import COLLAPSED_BASE_HEIGHT
from pochi_rl.robot import NOMINAL_BASE_HEIGHT

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

ROBOT = SceneEntityCfg("robot")


def base_lin_vel_z_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  return torch.square(asset.data.root_link_lin_vel_b[:, 2])


def base_ang_vel_xy_l2(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  return torch.sum(torch.square(asset.data.root_link_ang_vel_b[:, :2]), dim=1)


def _standing(env: ManagerBasedRlEnv, command_name: str, threshold: float):
  """1.0 for environments whose velocity command is (near) zero."""
  command = env.command_manager.get_command(command_name)
  magnitude = torch.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
  return (magnitude < threshold).float()


def stand_still_joint_deviation(
  env: ManagerBasedRlEnv,
  command_name: str = "base_velocity",
  command_threshold: float = 0.5,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  """Penalize drifting off the default stance while commanded to stand.

  ``feet_air_time`` already stops paying out below ``command_threshold``, so a
  standing robot gains nothing by stepping -- but it loses nothing either, and
  it marches in place.  This gives that idling a cost, measured as L1 deviation
  from the default joint angles so a single badly placed leg is penalized as
  much as a whole gait cycle of small ones.
  """
  asset = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  deviation = torch.sum(
    torch.abs(
      asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    ),
    dim=1,
  )
  return deviation * _standing(env, command_name, command_threshold)


# --- Stand-up task ------------------------------------------------------------


def _base_height(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  asset = env.scene[asset_cfg.name]
  return asset.data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]


def standup_reference_height(
  env: ManagerBasedRlEnv,
  settle_s: float,
  rise_s: float,
) -> torch.Tensor:
  """The commanded base height for every environment, from its own clock.

  Torch mirror of :func:`pochi_rl.control.standup.reference_height`; the shape
  and the constants both come from there.
  """
  t = env.episode_length_buf.float() * env.step_dt
  u = ((t - settle_s) / rise_s).clamp(0.0, 1.0)
  smoothstep = u * u * (3.0 - 2.0 * u)
  span = NOMINAL_BASE_HEIGHT - COLLAPSED_BASE_HEIGHT
  return COLLAPSED_BASE_HEIGHT + span * smoothstep


def standup_phase(
  env: ManagerBasedRlEnv,
  settle_s: float,
  rise_s: float,
) -> torch.Tensor:
  """How far through the height ramp each environment is, in [0, 1]."""
  t = env.episode_length_buf.float() * env.step_dt
  return ((t - settle_s) / rise_s).clamp(0.0, 1.0)


def standup_reference_obs(
  env: ManagerBasedRlEnv,
  settle_s: float,
  rise_s: float,
) -> torch.Tensor:
  """What the policy is told about the ramp: the target, and its progress.

  Without a clock the task is not Markov -- "how high should I be right now"
  is not a function of the robot's state -- so the policy would have to guess,
  and the only guess with any value is "as high as possible, immediately".
  Hardware has a clock too, so this costs nothing at deployment.
  """
  height = standup_reference_height(env, settle_s, rise_s)
  phase = standup_phase(env, settle_s, rise_s)
  return torch.stack((height, phase), dim=-1)


def base_height_tracking(
  env: ManagerBasedRlEnv,
  settle_s: float,
  rise_s: float,
  std: float,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  """Reward following the height ramp, with an L1-exponential kernel.

  ``exp(-|error| / std)`` rather than the usual ``exp(-error^2 / std^2)``: a
  squared kernel goes flat once the robot is more than a few centimetres off,
  which is where it spends the start of every episode, and a reward with no
  gradient there is no reward at all.
  """
  reference = standup_reference_height(env, settle_s, rise_s)
  height = _base_height(env, asset_cfg)
  return torch.exp(-torch.abs(height - reference) / std)


def upright(
  env: ManagerBasedRlEnv,
  std: float,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  """Reward the base z axis pointing at the sky."""
  asset = env.scene[asset_cfg.name]
  tilt = torch.norm(asset.data.projected_gravity_b[:, :2], dim=1)
  return torch.exp(-tilt / std)


def posture_along_ramp(
  env: ManagerBasedRlEnv,
  settle_s: float,
  rise_s: float,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  """L1 distance from the default stance, faded in over the ramp.

  Ungated, this term is what breaks the manoeuvre.  Collapsed on the floor the
  robot is ~7.5 rad of L1 away from the standing stance, so an ungated penalty
  bills it about as much per step as the tracking reward is worth -- and the
  cheapest way out is to stand up immediately and wait for the reference to
  catch up, which is exactly the behaviour the ramp exists to prevent.  Scaling
  by the ramp phase asks for the default stance only where the reference is
  actually asking the robot to be standing in it.
  """
  asset = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  deviation = torch.sum(
    torch.abs(
      asset.data.joint_pos[:, joint_ids] - asset.data.default_joint_pos[:, joint_ids]
    ),
    dim=1,
  )
  return deviation * standup_phase(env, settle_s, rise_s)


def joint_speed_over_limit(
  env: ManagerBasedRlEnv,
  soft_limit: float,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  """Squared joint speed in excess of ``soft_limit`` [rad/s].

  The actuator's torque-speed curve already makes a hard cap unreachable; this
  is the soft one, so the policy settles well inside the envelope instead of
  learning to ride against it.
  """
  asset = env.scene[asset_cfg.name]
  excess = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids]) - soft_limit
  return torch.sum(torch.square(excess.clamp(min=0.0)), dim=1)


def feet_on_ground(
  env: ManagerBasedRlEnv,
  sensor_name: str,
) -> torch.Tensor:
  """Fraction of the four feet currently touching the ground."""
  sensor = env.scene[sensor_name]
  found = sensor.data.found
  assert found is not None
  return found.float().mean(dim=-1)


def height_tracking_error(
  env: ManagerBasedRlEnv,
  settle_s: float,
  rise_s: float,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  """Absolute distance from the height reference [m].

  Paired with :func:`base_height_tracking`, which is the exponential kernel.
  The kernel is what makes close tracking worth being precise about, but its
  gradient dies with its value, so on its own it leaves the policy free to sit
  20 cm below the reference on a plateau -- which is exactly the local optimum
  a run without this term settles into.  A linear error has the same gradient
  everywhere, so there is always a reason to close the gap.
  """
  reference = standup_reference_height(env, settle_s, rise_s)
  return torch.abs(_base_height(env, asset_cfg) - reference)


def contact_while_lifted(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  min_height: float,
  force_threshold: float = 5.0,
  asset_cfg: SceneEntityCfg = ROBOT,
) -> torch.Tensor:
  """Penalize anything but the feet touching down once the robot is up.

  Lying on the floor, the belly and the front thighs rest on it -- that is what
  lying down *is*, and penalizing it would only teach the policy to squirm.
  What has to be discouraged is levering the robot up on its knees, so the term
  is gated on the base already being ``min_height`` off the ground.
  """
  sensor = env.scene[sensor_name]
  data = sensor.data
  if data.force_history is not None:
    force = torch.norm(data.force_history, dim=-1)  # [B, N, H]
    hits = (force > force_threshold).any(dim=1).float().mean(dim=-1)  # [B]
  else:
    assert data.found is not None
    hits = data.found.float().sum(dim=-1)

  return hits * (_base_height(env, asset_cfg) > min_height).float()
