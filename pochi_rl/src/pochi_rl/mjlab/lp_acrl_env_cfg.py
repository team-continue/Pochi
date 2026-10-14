"""Independent Pochi LP-ACRL flat and terrain-aware locomotion tasks.

The signed learning-progress curriculum follows arXiv:2601.17428. The task
uses Pochi's geometry, motor envelope and hardware PD gains. Terrain sizes and
acceleration regularization are explicitly adapted to this smaller robot.
"""

from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass, replace

import mujoco
import numpy as np
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as env_mdp
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.sensor import GridPatternCfg, ObjRef, RayCastSensorCfg
from mjlab.terrains import (
  BoxInvertedPyramidStairsTerrainCfg,
  BoxPyramidStairsTerrainCfg,
  HfPyramidSlopedTerrainCfg,
  HfRandomUniformTerrainCfg,
  TerrainEntityCfg,
  TerrainGeneratorCfg,
)
from mjlab.terrains.terrain_generator import SubTerrainCfg, TerrainOutput
from mjlab.utils.noise import UniformNoiseCfg

from pochi_rl.lp_acrl import ROUGH_TASK_CATALOG
from pochi_rl.mjlab import lp_acrl_rewards as rewards
from pochi_rl.mjlab import rewards as base_rewards
from pochi_rl.mjlab.lp_acrl_commands import LPACRLVelocityCommandCfg, pre_reset_task
from pochi_rl.mjlab.velocity_flat_env_cfg import (
  JOINTS,
  ROBOT,
  pochi_velocity_flat_env_cfg,
)
from pochi_rl.robot import BASE_BODY, NOMINAL_BASE_HEIGHT


@dataclass(kw_only=True)
class DiscreteTerrainCfg(SubTerrainCfg):
  """Turn mjlab's jittered row difficulty into four reproducible task levels."""

  terrain: SubTerrainCfg
  levels: int = 4

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    level = min(int(difficulty * self.levels), self.levels - 1)
    terrain = replace(self.terrain, size=self.size)
    return terrain.function((level + 1) / self.levels, spec, rng)


@dataclass(kw_only=True)
class ProgressivelyRoughTerrainCfg(HfRandomUniformTerrainCfg):
  """Native random heightfield with amplitude increasing with task level.

  mjlab's stock random terrain ignores difficulty, which would make all four
  roughness levels equivalent. Rescale before delegating to its generator.
  """

  max_amplitude: float = 0.04

  def function(
    self, difficulty: float, spec: mujoco.MjSpec, rng: np.random.Generator
  ) -> TerrainOutput:
    amplitude = max(self.vertical_scale, self.max_amplitude * difficulty)
    cfg = HfRandomUniformTerrainCfg(
      size=self.size,
      noise_range=(0.0, amplitude),
      noise_step=self.noise_step,
      horizontal_scale=self.horizontal_scale,
      downsampled_scale=self.downsampled_scale,
      vertical_scale=self.vertical_scale,
      border_width=self.border_width,
    )
    output = cfg.function(difficulty, spec, rng)
    # Spawn above the highest sample; spline interpolation may overshoot the
    # requested amplitude. This avoids starting with a foot inside a bump.
    output.origin[2] = max(
      geometry.geom.pos[2] + geometry.hfield.size[2]
      for geometry in output.geometries
      if geometry.geom is not None and geometry.hfield is not None
    )
    return output


def lp_acrl_terrain_cfg() -> TerrainEntityCfg:
  """Five terrain families × four levels, scaled for Pochi's 0.317 m stance."""
  stairs = dict(
    step_height_range=(0.0, 0.08),
    step_width=0.30,
    platform_width=1.2,
    border_width=0.2,
  )
  slopes = dict(
    slope_range=(0.0, 0.30),
    platform_width=1.2,
    border_width=0.2,
    horizontal_scale=0.1,
    vertical_scale=0.002,
  )
  # The robot spawns at the center: an inverted pyramid climbs OUT from its
  # low center; a regular pyramid descends OUT from its raised center.
  families = {
    "ascending_stairs": BoxInvertedPyramidStairsTerrainCfg(**stairs),
    "descending_stairs": BoxPyramidStairsTerrainCfg(**stairs),
    "rough": ProgressivelyRoughTerrainCfg(
      noise_range=(0.0, 0.04),
      noise_step=0.002,
      horizontal_scale=0.1,
      downsampled_scale=0.2,
      vertical_scale=0.002,
      border_width=0.2,
    ),
    "uphill": HfPyramidSlopedTerrainCfg(**slopes, inverted=True),
    "downhill": HfPyramidSlopedTerrainCfg(**slopes),
  }
  return TerrainEntityCfg(
    terrain_type="generator",
    terrain_generator=TerrainGeneratorCfg(
      seed=42,
      curriculum=True,
      size=(24.0, 24.0),
      num_rows=ROUGH_TASK_CATALOG.num_levels,
      num_cols=len(ROUGH_TASK_CATALOG.terrains),
      border_width=2.0,
      sub_terrains={
        name: DiscreteTerrainCfg(terrain=families[name])
        for name in ROUGH_TASK_CATALOG.terrains
      },
    ),
  )


def _lp_acrl_env_cfg(*, rough: bool, play: bool) -> ManagerBasedRlEnvCfg:
  # deepcopy isolates even the shared robot constants used by the base factory.
  cfg = deepcopy(pochi_velocity_flat_env_cfg(play=play))
  robot = cfg.scene.entities["robot"]
  assert robot.articulation is not None
  robot.articulation = replace(
    robot.articulation,
    actuators=tuple(
      replace(actuator, stiffness=140.0, damping=1.5)
      for actuator in robot.articulation.actuators
    ),
  )
  cfg.scene.num_envs = 1 if play else 2048
  # Stair impacts exposed a non-finite physics state at 5 ms in long GPU
  # training. Keep the policy at 50 Hz while integrating rough contact at 400 Hz.
  cfg.decimation = 8 if rough else 4
  cfg.sim.mujoco.timestep = 0.0025 if rough else 0.005
  if rough:
    # Saved finite failure states produced NaNs under Warp Newton with both
    # cone types, but stayed finite under CG (128 worlds × 200 substeps).
    # Avoid its float32 Hessian factorization; retain elliptic friction.
    cfg.sim.mujoco.solver = "cg"
    cfg.sim.mujoco.cone = "elliptic"
    for sensor in cfg.scene.sensors:
      if getattr(sensor, "history_length", 0) == 4:
        sensor.history_length = cfg.decimation
  cfg.episode_length_s = 20.0
  cfg.scale_rewards_by_dt = True
  cfg.actions["joint_pos"].scale = 0.25
  cfg.commands = {
    "base_velocity": LPACRLVelocityCommandCfg(
      entity_name="robot",
      rough=rough,
      adaptive=not play,
      resampling_time_range=(1.0e9, 1.0e9),
      heading_command=False,
      rel_heading_envs=0.0,
      rel_standing_envs=0.0,
      debug_vis=True,
      ranges=LPACRLVelocityCommandCfg.Ranges(
        lin_vel_x=(-0.5, 0.5),
        lin_vel_y=(-0.1, 0.1) if rough else (0.0, 0.0),
        ang_vel_z=(-1.2, 1.2) if rough else (0.0, 0.0),
      ),
    )
  }
  cfg.curriculum = {"learning_progress": CurriculumTermCfg(func=pre_reset_task)}
  cfg.events.pop("push_robot", None)
  cfg.events["reset_root"].params["pose_range"] = {
    "x": (-0.1, 0.1),
    "y": (-0.1, 0.1),
    "yaw": (-math.pi, math.pi),
  }
  cfg.events["reset_root"].params["velocity_range"] = {}
  cfg.events["base_mass"].params["ranges"] = (-0.5, 0.5)
  cfg.events["foot_friction"].params["ranges"] = (0.6, 1.2)

  cfg.rewards = {
    "track_lin_vel_xy_exp": RewardTermCfg(
      func=rewards.track_linear_velocity,
      weight=1.0,
    ),
    "track_ang_vel_z_exp": RewardTermCfg(
      func=rewards.track_yaw_velocity,
      weight=0.5,
    ),
    "lin_vel_z_l2": RewardTermCfg(
      func=base_rewards.base_lin_vel_z_l2,
      weight=-4.0,
    ),
    "ang_vel_xy_l2": RewardTermCfg(
      func=base_rewards.base_ang_vel_xy_l2,
      weight=-0.05,
    ),
    "joint_torques_l2": RewardTermCfg(
      func=env_mdp.joint_torques_l2,
      weight=-2.0e-5,
      params={"asset_cfg": JOINTS},
    ),
    "joint_vel_l2": RewardTermCfg(
      func=env_mdp.joint_vel_l2,
      weight=-0.001,
      params={"asset_cfg": JOINTS},
    ),
    # Table II prints -0.001 for accelerations. Applied to MuJoCo's rad/s²
    # acceleration that overwhelms tracking; use the established Pochi scale.
    "joint_acc_l2": RewardTermCfg(
      func=env_mdp.joint_acc_l2,
      weight=-2.5e-7,
      params={"asset_cfg": JOINTS},
    ),
    "target_position_rate": RewardTermCfg(
      func=rewards.target_position_rate,
      weight=-0.25,
    ),
    "feet_air_time": RewardTermCfg(
      func=rewards.touchdown_air_time,
      weight=2.0,
    ),
    "undesired_contacts": RewardTermCfg(
      func=rewards.undesired_contact_count,
      weight=-0.001,
    ),
  }
  cfg.terminations = {
    "time_out": TerminationTermCfg(func=env_mdp.time_out, time_out=True),
    "fell_over": TerminationTermCfg(
      func=env_mdp.bad_orientation,
      params={"limit_angle": math.radians(70.0), "asset_cfg": ROBOT},
    ),
  }
  if rough:
    cfg.rewards = {
      "physics_validity": RewardTermCfg(func=rewards.assert_valid_physics, weight=1.0),
      **cfg.rewards,
    }
    cfg.scene.terrain = lp_acrl_terrain_cfg()
    cfg.scene.extent = 20.0
    cfg.scene.sensors += (
      RayCastSensorCfg(
        name="terrain_scan",
        frame=ObjRef(type="body", name=BASE_BODY, entity="robot"),
        ray_alignment="yaw",
        pattern=GridPatternCfg(size=(2.4, 1.0), resolution=0.1),
        max_distance=3.0,
        include_geom_groups=(0,),
        exclude_parent_body=True,
      ),
    )
    for name, group in cfg.observations.items():
      group.terms["height_scan"] = ObservationTermCfg(
        func=env_mdp.height_scan,
        params={"sensor_name": "terrain_scan", "offset": NOMINAL_BASE_HEIGHT},
        clip=(-1.0, 1.0),
        noise=UniformNoiseCfg(n_min=-0.02, n_max=0.02) if name == "actor" else None,
      )
    cfg.terminations["task_terrain_edge"] = TerminationTermCfg(
      func=rewards.outside_task_terrain,
      time_out=True,
    )
  return cfg


def pochi_lp_acrl_flat_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """Pochi adaptation: five forward-speed bins on a plane, from random weights."""
  return _lp_acrl_env_cfg(rough=False, play=play)


def pochi_lp_acrl_rough_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
  """600 velocity/terrain tasks with height observations for the teacher policy."""
  return _lp_acrl_env_cfg(rough=True, play=play)
