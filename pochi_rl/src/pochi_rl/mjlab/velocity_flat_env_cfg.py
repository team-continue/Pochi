"""Flat-ground velocity-tracking environment for Pochi in mjlab."""

from __future__ import annotations

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg, curriculums
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from pochi_rl.mjlab import rewards as pochi_rewards
from pochi_rl.mjlab.entity_cfg import POCHI_ROBOT_CFG
from pochi_rl.robot import BASE_BODY, FOOT_GEOMS, JOINT_NAMES
from pochi_rl.robot.pochi_constants import SHANK_GEOMS, THIGH_GEOMS
from pochi_rl.task_spec import POCHI_TASK_SPEC as S

ROBOT = SceneEntityCfg("robot")
JOINTS = SceneEntityCfg("robot", joint_names=JOINT_NAMES)


def _uniform_noise(std: float) -> Unoise:
  return Unoise(n_min=-std, n_max=std)


def _observation_terms(*, add_noise: bool) -> dict[str, ObservationTermCfg]:
  return {
    "base_lin_vel": ObservationTermCfg(
      func=env_mdp.base_lin_vel,
      noise=_uniform_noise(S.observations.noise.lin_vel) if add_noise else None,
    ),
    "base_ang_vel": ObservationTermCfg(
      func=env_mdp.base_ang_vel,
      noise=_uniform_noise(S.observations.noise.ang_vel) if add_noise else None,
    ),
    "projected_gravity": ObservationTermCfg(
      func=env_mdp.projected_gravity,
      noise=_uniform_noise(S.observations.noise.projected_gravity)
      if add_noise
      else None,
    ),
    "velocity_commands": ObservationTermCfg(
      func=env_mdp.generated_commands,
      params={"command_name": "base_velocity"},
    ),
    "joint_pos_rel": ObservationTermCfg(
      func=env_mdp.joint_pos_rel,
      params={"asset_cfg": JOINTS},
      noise=_uniform_noise(S.observations.noise.joint_pos) if add_noise else None,
    ),
    "joint_vel_rel": ObservationTermCfg(
      func=env_mdp.joint_vel_rel,
      params={"asset_cfg": JOINTS},
      noise=_uniform_noise(S.observations.noise.joint_vel) if add_noise else None,
    ),
    "last_action": ObservationTermCfg(func=env_mdp.last_action),
  }


def pochi_velocity_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Create Pochi flat velocity tracking configuration."""

  foot_contact_sensor = ContactSensorCfg(
    name="feet_ground_contact",
    primary=ContactMatch(mode="geom", pattern=FOOT_GEOMS, entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found", "force"),
    reduce="netforce",
    num_slots=1,
    track_air_time=True,
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

  actions: dict[str, ActionTermCfg] = {
    "joint_pos": JointPositionActionCfg(
      entity_name="robot",
      actuator_names=JOINT_NAMES,
      scale=S.control.action_scale,
      use_default_offset=True,
    )
  }

  _first = S.curriculum.stages[0]
  commands: dict[str, CommandTermCfg] = {
    "base_velocity": UniformVelocityCommandCfg(
      entity_name="robot",
      resampling_time_range=(10.0, 10.0),
      rel_standing_envs=0.1,
      rel_heading_envs=0.3,
      heading_command=True,
      heading_control_stiffness=0.5,
      debug_vis=True,
      # Start at the curriculum's first stage; the curriculum term widens it
      # from there.  Playback skips the curriculum, so it gets the full range.
      ranges=UniformVelocityCommandCfg.Ranges(
        lin_vel_x=S.commands.lin_vel_x if play else _first.lin_vel_x,
        lin_vel_y=S.commands.lin_vel_y if play else _first.lin_vel_y,
        ang_vel_z=S.commands.ang_vel_z if play else _first.ang_vel_z,
        heading=S.commands.heading,
      ),
    )
  }

  rewards = {
    "track_lin_vel_xy_exp": RewardTermCfg(
      func=mdp.track_linear_velocity,
      weight=S.rewards.track_lin_vel_xy_exp,
      params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
    ),
    "track_ang_vel_z_exp": RewardTermCfg(
      func=mdp.track_angular_velocity,
      weight=S.rewards.track_ang_vel_z_exp,
      params={"command_name": "base_velocity", "std": math.sqrt(0.5)},
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
    "joint_torques_l2": RewardTermCfg(
      func=env_mdp.joint_torques_l2,
      weight=S.rewards.joint_torques_l2,
      params={"asset_cfg": JOINTS},
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
    "action_rate_l2": RewardTermCfg(
      func=env_mdp.action_rate_l2,
      weight=S.rewards.action_rate_l2,
    ),
    "stand_still": RewardTermCfg(
      func=pochi_rewards.stand_still_joint_deviation,
      weight=S.rewards.stand_still,
      params={
        "command_name": "base_velocity",
        "command_threshold": S.thresholds.stand_still,
        "asset_cfg": JOINTS,
      },
    ),
    "feet_air_time": RewardTermCfg(
      func=mdp.feet_air_time,
      weight=S.rewards.feet_air_time,
      params={
        "sensor_name": "feet_ground_contact",
        "threshold_min": 0.05,
        "threshold_max": 0.5,
        "command_name": "base_velocity",
        "command_threshold": S.thresholds.feet_air_time,
      },
    ),
    "undesired_contacts": RewardTermCfg(
      func=mdp.self_collision_cost,
      weight=S.rewards.undesired_contacts,
      params={"sensor_name": "limb_ground_contact"},
    ),
    "termination": RewardTermCfg(
      func=env_mdp.is_terminated,
      weight=S.rewards.termination,
    ),
  }

  events = {
    "reset_root": EventTermCfg(
      func=env_mdp.reset_root_state_uniform,
      mode="reset",
      params={
        "pose_range": {
          "x": (-0.5, 0.5),
          "y": (-0.5, 0.5),
          "yaw": (-math.pi, math.pi),
        },
        "velocity_range": {
          "x": (-0.2, 0.2),
          "y": (-0.2, 0.2),
          "z": (-0.2, 0.2),
          "roll": (-0.2, 0.2),
          "pitch": (-0.2, 0.2),
          "yaw": (-0.2, 0.2),
        },
        "asset_cfg": ROBOT,
      },
    ),
    "reset_joints": EventTermCfg(
      func=env_mdp.reset_joints_by_offset,
      mode="reset",
      params={
        "position_range": (-0.05, 0.05),
        "velocity_range": (-0.05, 0.05),
        "asset_cfg": JOINTS,
      },
    ),
    "push_robot": EventTermCfg(
      func=env_mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=S.events.push_interval_s,
      params={
        "velocity_range": {
          "x": S.events.push_vel_xy,
          "y": S.events.push_vel_xy,
        },
        "asset_cfg": ROBOT,
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

  curriculum: dict[str, CurriculumTermCfg] = {
    "command_ranges": CurriculumTermCfg(
      func=curriculums.commands_vel,
      params={
        "command_name": "base_velocity",
        "velocity_stages": [
          {
            "step": stage.step,
            "lin_vel_x": stage.lin_vel_x,
            "lin_vel_y": stage.lin_vel_y,
            "ang_vel_z": stage.ang_vel_z,
          }
          for stage in S.curriculum.stages
        ],
      },
    )
  }

  if play:
    events.pop("push_robot", None)
    curriculum = {}

  terminations = {
    "time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=env_mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0), "asset_cfg": ROBOT},
    ),
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(
      num_envs=1 if play else 4096,
      env_spacing=2.5,
      terrain=TerrainEntityCfg(
        terrain_type="plane",
        env_spacing=2.5,
      ),
      entities={"robot": POCHI_ROBOT_CFG},
      sensors=(foot_contact_sensor, limb_contact_sensor),
      extent=2.5,
    ),
    observations=observations,
    actions=actions,
    commands=commands,
    events=events,
    curriculum=curriculum,
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
    episode_length_s=1.0e9 if play else S.control.episode_length_s,
  )
