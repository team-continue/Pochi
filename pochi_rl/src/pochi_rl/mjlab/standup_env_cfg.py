"""Flat-ground stand-up environment for Pochi in mjlab.

The robot is laid on the floor in the pose it collapses into when the motors
are cut, and has to follow a height ramp back up onto its feet.  The scripted
controller in :mod:`pochi_rl.control.standup` already solves this; the learned
version exists because it comes with a motor envelope -- see
``POCHI_SLOW_ROBOT_CFG``, whose torque-speed curve stops a motor driving a
joint past ~2 rad/s and caps torque at the RS02's continuous rating -- which
makes it safe to stand next to while debugging hardware.

The ramp is the point of the design.  Rewarding the standing height on its own
produces a policy that is up in 0.8 s and looks nothing like the scripted
manoeuvre: no speed penalty small enough to leave the rest of the task intact
outweighs the discounted value of standing early.  Tracking a reference over
time makes the timing part of the objective instead of something traded against
it, and the reference is the same one the scripted controller follows.

Observations are the velocity task's with the three velocity-command numbers
swapped for two describing the ramp, so the two policies see the same robot and
hand off to each other at the same default stance.
"""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from pochi_rl.control.standup import COLLAPSED_BASE_HEIGHT, COLLAPSED_JOINT_POS
from pochi_rl.mjlab import events as pochi_events
from pochi_rl.mjlab import rewards as pochi_rewards
from pochi_rl.mjlab.actions import CenteredJointPositionActionCfg
from pochi_rl.mjlab.entity_cfg import POCHI_SLOW_ROBOT_CFG
from pochi_rl.robot import BASE_BODY, DEFAULT_JOINT_POS, FOOT_GEOMS, JOINT_NAMES
from pochi_rl.robot.pochi_constants import SHANK_GEOMS, THIGH_GEOMS
from pochi_rl.task_spec import POCHI_STANDUP_SPEC as S

ROBOT = SceneEntityCfg("robot")
JOINTS = SceneEntityCfg("robot", joint_names=JOINT_NAMES)
ORDERED_JOINTS = SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)


def _uniform_noise(std: float) -> Unoise:
  return Unoise(n_min=-std, n_max=std)


def _observation_terms(*, add_noise: bool) -> dict[str, ObservationTermCfg]:
  noise = S.observations.noise
  return {
    "base_lin_vel": ObservationTermCfg(
      func=env_mdp.base_lin_vel,
      noise=_uniform_noise(noise.lin_vel) if add_noise else None,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=env_mdp.base_ang_vel,
      noise=_uniform_noise(noise.ang_vel) if add_noise else None,
    ),
    "projected_gravity": ObservationTermCfg(
      func=env_mdp.projected_gravity,
      noise=_uniform_noise(noise.projected_gravity) if add_noise else None,
    ),
    # Stands where the velocity task puts its velocity command: the thing the
    # policy is being asked for right now.
    "standup_reference": ObservationTermCfg(
      func=pochi_rewards.standup_reference_obs,
      params={
        "settle_s": S.reference.settle_s,
        "rise_s": S.reference.rise_s,
      },
    ),
    "joint_pos_rel": ObservationTermCfg(
      func=env_mdp.joint_pos_rel,
      params={"asset_cfg": JOINTS},
      noise=_uniform_noise(noise.joint_pos) if add_noise else None,
    ),
    "joint_vel_rel": ObservationTermCfg(
      func=env_mdp.joint_vel_rel,
      params={"asset_cfg": JOINTS},
      noise=_uniform_noise(noise.joint_vel) if add_noise else None,
    ),
    "last_action": ObservationTermCfg(func=env_mdp.last_action),
  }


def pochi_standup_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create the Pochi flat-ground stand-up configuration."""

  foot_contact_sensor = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
  )
  limb_contact_sensor = ContactSensorCfg(
    name="limb_ground_contact",
    primary=ContactMatch(
      mode="geom",
      pattern=THIGH_GEOMS + SHANK_GEOMS + ("base_collision",),
      entity="robot",
    ),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    history_length=S.control.decimation,
  )

  observations = {
    "actor": ObservationGroupCfg(
      terms=_observation_terms(add_noise=True),
      concatenate_terms=True,
      enable_corruption=not play,
    ),
    "critic": ObservationGroupCfg(
      terms=_observation_terms(add_noise=False),
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  # Centred halfway between the pose the robot starts in and the one it has to
  # finish in, so neither end of the manoeuvre sits in the tail of the policy's
  # action distribution.  See CenteredJointPositionAction for the failure mode
  # this avoids.
  centre_pose = {
    name: 0.5 * (COLLAPSED_JOINT_POS[name] + DEFAULT_JOINT_POS[name])
    for name in JOINT_NAMES
  }
  actions: dict[str, ActionTermCfg] = {
    "joint_pos": CenteredJointPositionActionCfg(
      entity_name="robot",
      actuator_names=JOINT_NAMES,
      scale=S.control.action_scale,
      centre_pose=centre_pose,
    )
  }

  rewards = {
    "track_height": RewardTermCfg(
      func=pochi_rewards.base_height_tracking,
      weight=S.rewards.track_height,
      params={
        "settle_s": S.reference.settle_s,
        "rise_s": S.reference.rise_s,
        "std": S.height_std,
        "asset_cfg": ROBOT,
      },
    ),
    "height_error": RewardTermCfg(
      func=pochi_rewards.height_tracking_error,
      weight=S.rewards.height_error,
      params={
        "settle_s": S.reference.settle_s,
        "rise_s": S.reference.rise_s,
        "asset_cfg": ROBOT,
      },
    ),
    "upright": RewardTermCfg(
      func=pochi_rewards.upright,
      weight=S.rewards.upright,
      params={"std": S.upright_std, "asset_cfg": ROBOT},
    ),
    "posture": RewardTermCfg(
      func=pochi_rewards.posture_along_ramp,
      weight=S.rewards.posture,
      params={
        "settle_s": S.reference.settle_s,
        "rise_s": S.reference.rise_s,
        "asset_cfg": JOINTS,
      },
    ),
    "feet_contact": RewardTermCfg(
      func=pochi_rewards.feet_on_ground,
      weight=S.rewards.feet_contact,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "non_foot_contact": RewardTermCfg(
      func=pochi_rewards.contact_while_lifted,
      weight=S.rewards.non_foot_contact,
      params={
        "sensor_name": "limb_ground_contact",
        "min_height": COLLAPSED_BASE_HEIGHT + S.control.lifted_margin,
        "asset_cfg": ROBOT,
      },
    ),
    "joint_speed_limit": RewardTermCfg(
      func=pochi_rewards.joint_speed_over_limit,
      weight=S.rewards.joint_speed_limit,
      params={"soft_limit": S.safety.soft_speed_limit, "asset_cfg": JOINTS},
    ),
    "joint_vel_l2": RewardTermCfg(
      func=env_mdp.joint_vel_l2,
      weight=S.rewards.joint_vel_l2,
      params={"asset_cfg": JOINTS},
    ),
    "joint_acc_l2": RewardTermCfg(
      func=env_mdp.joint_acc_l2,
      weight=S.rewards.joint_acc_l2,
      params={"asset_cfg": JOINTS},
    ),
    "joint_torques_l2": RewardTermCfg(
      func=env_mdp.joint_torques_l2,
      weight=S.rewards.joint_torques_l2,
      params={"asset_cfg": JOINTS},
    ),
    "action_rate_l2": RewardTermCfg(
      func=env_mdp.action_rate_l2,
      weight=S.rewards.action_rate_l2,
    ),
    "lin_vel_z_l2": RewardTermCfg(
      func=pochi_rewards.base_lin_vel_z_l2,
      weight=S.rewards.lin_vel_z_l2,
      params={"asset_cfg": ROBOT},
    ),
    "ang_vel_xy_l2": RewardTermCfg(
      func=pochi_rewards.base_ang_vel_xy_l2,
      weight=S.rewards.ang_vel_xy_l2,
      params={"asset_cfg": ROBOT},
    ),
    "termination": RewardTermCfg(
      func=env_mdp.is_terminated,
      weight=S.rewards.termination,
    ),
  }

  events: dict[str, EventTermCfg] = {
    "reset_collapsed": EventTermCfg(
      func=pochi_events.reset_collapsed,
      mode="reset",
      params={
        "joint_pos": tuple(COLLAPSED_JOINT_POS[name] for name in JOINT_NAMES),
        "base_height": COLLAPSED_BASE_HEIGHT,
        "joint_pos_range": S.reset.joint_pos,
        "joint_vel_range": S.reset.joint_vel,
        "height_range": S.reset.base_height,
        "roll_pitch_range": S.reset.base_roll_pitch,
        "asset_cfg": ORDERED_JOINTS,
      },
    ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=FOOT_GEOMS),
        "operation": "abs",
        "ranges": S.events.foot_friction,
        "shared_random": True,
      },
    ),
    "base_mass": EventTermCfg(
      mode="startup",
      func=dr.body_mass,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=(BASE_BODY,)),
        "operation": "add",
        "ranges": S.events.base_mass_add_kg,
      },
    ),
  }

  terminations = {
    "time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True),
    # Only for a robot that has genuinely rolled over.  The usual 70 degree
    # limit would fire on the start state, which is the whole task.
    "flipped": TerminationTermCfg(
      func=env_mdp.bad_orientation,
      params={"limit_angle": math.radians(110.0), "asset_cfg": ROBOT},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=1 if play else 4096,
      env_spacing=2.5,
      terrain=TerrainEntityCfg(terrain_type="plane", env_spacing=2.5),
      entities={"robot": POCHI_SLOW_ROBOT_CFG},
      sensors=(foot_contact_sensor, limb_contact_sensor),
      extent=2.5,
    ),
    observations=observations,
    actions=actions,
    commands={},
    events=events,
    curriculum={},
    rewards=rewards,
    terminations=terminations,
    sim=SimulationCfg(
      nconmax=64,
      njmax=300,
      mujoco=MujocoCfg(
        timestep=S.control.sim_dt,
        iterations=50,
        cone="elliptic",
      ),
    ),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name=BASE_BODY,
      distance=1.5,
      elevation=-10.0,
      azimuth=90.0,
    ),
    decimation=S.control.decimation,
    episode_length_s=S.control.episode_length_s,
  )
