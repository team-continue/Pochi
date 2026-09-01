"""Backend-neutral velocity task specification for Pochi."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass(frozen=True)
class CommandRanges:
  lin_vel_x: tuple[float, float] = (-1.0, 1.0)
  lin_vel_y: tuple[float, float] = (-0.5, 0.5)
  ang_vel_z: tuple[float, float] = (-1.0, 1.0)
  heading: tuple[float, float] = (-math.pi, math.pi)


@dataclass(frozen=True)
class CurriculumStage:
  """Command ranges to switch to once ``step`` env steps have elapsed."""

  step: int
  lin_vel_x: tuple[float, float]
  lin_vel_y: tuple[float, float]
  ang_vel_z: tuple[float, float]


@dataclass(frozen=True)
class CommandCurriculum:
  """Widen the velocity command range as training progresses.

  Training the full range from the start leaves the slow end poorly covered:
  most sampled commands are fast, so the policy optimises for those and merely
  scales the same gait down when asked to crawl.  Starting narrow forces it to
  learn a genuine low-speed gait first, then extends outward.  The final stage
  has to match ``CommandRanges`` -- ``test_task_spec`` checks that.

  Steps are per-environment env steps; a 1500-iteration run at 24 steps per
  iteration is 36000, so the range is at full width for the last ~45%.
  """

  stages: tuple[CurriculumStage, ...] = (
    CurriculumStage(0, (-0.3, 0.3), (-0.2, 0.2), (-0.5, 0.5)),
    CurriculumStage(6_000, (-0.5, 0.5), (-0.3, 0.3), (-0.7, 0.7)),
    CurriculumStage(12_000, (-0.7, 0.7), (-0.4, 0.4), (-1.0, 1.0)),
    CurriculumStage(20_000, (-1.0, 1.0), (-0.5, 0.5), (-1.0, 1.0)),
  )


@dataclass(frozen=True)
class RewardWeights:
  track_lin_vel_xy_exp: float = 1.5
  track_ang_vel_z_exp: float = 0.75
  lin_vel_z_l2: float = -2.0
  ang_vel_xy_l2: float = -0.05
  joint_torques_l2: float = -1.0e-4
  # Slow the legs down: penalise how fast and how abruptly the joints move, so
  # the gait is deliberate rather than a scramble.
  joint_vel_l2: float = -2.0e-3
  joint_acc_l2: float = -2.5e-7
  action_rate_l2: float = -0.01
  # Hold the default stance when no velocity is commanded.  feet_air_time
  # already pays nothing below the standing threshold; this makes marching in
  # place actively cost something.
  stand_still: float = -0.5
  feet_air_time: float = 0.5
  undesired_contacts: float = -1.0
  termination: float = -200.0


@dataclass(frozen=True)
class CommandThresholds:
  """Command magnitude below which the robot counts as standing.

  These have to stay well under the slowest command the robot is asked to walk
  at.  mjlab defaults both to 0.5, which silently makes every command under
  0.5 m/s a *standing* one: the stand-still penalty fires, the air-time reward
  does not, and standing still through a 0.2 m/s command still earns ~92 % of
  the tracking reward.  The robot then refuses to move below 0.5 m/s at all.
  """

  stand_still: float = 0.1
  feet_air_time: float = 0.1


@dataclass(frozen=True)
class ObservationNoise:
  lin_vel: float = 0.1
  ang_vel: float = 0.2
  projected_gravity: float = 0.05
  joint_pos: float = 0.01
  joint_vel: float = 1.5


@dataclass(frozen=True)
class ObservationSpec:
  noise: ObservationNoise = field(default_factory=ObservationNoise)

  @property
  def policy_dim(self) -> int:
    # base lin vel, base ang vel, projected gravity, commands, joint pos/vel,
    # last action.
    return 3 + 3 + 3 + 3 + 12 + 12 + 12


@dataclass(frozen=True)
class EventSpec:
  push_interval_s: tuple[float, float] = (10.0, 15.0)
  push_vel_xy: tuple[float, float] = (-0.5, 0.5)
  foot_friction: tuple[float, float] = (0.4, 1.2)
  base_mass_add_kg: tuple[float, float] = (-1.0, 3.0)


@dataclass(frozen=True)
class ControlSpec:
  decimation: int = 4
  sim_dt: float = 0.005
  episode_length_s: float = 20.0
  action_scale: float = 0.25

  @property
  def policy_hz(self) -> float:
    return 1.0 / (self.sim_dt * self.decimation)


@dataclass(frozen=True)
class PochiTaskSpec:
  commands: CommandRanges = field(default_factory=CommandRanges)
  curriculum: CommandCurriculum = field(default_factory=CommandCurriculum)
  thresholds: CommandThresholds = field(default_factory=CommandThresholds)
  rewards: RewardWeights = field(default_factory=RewardWeights)
  observations: ObservationSpec = field(default_factory=ObservationSpec)
  events: EventSpec = field(default_factory=EventSpec)
  control: ControlSpec = field(default_factory=ControlSpec)


POCHI_TASK_SPEC = PochiTaskSpec()
