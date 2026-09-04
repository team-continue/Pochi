"""Backend-neutral velocity task specification for Pochi."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from pochi_rl.robot import RS02_RATED_TORQUE_NM


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


# --- Stand-up task -----------------------------------------------------------
# A second, much smaller task: get up off the floor, slowly, on the feet.  It
# exists to be run on the hardware while debugging, so the defining constraint
# is not performance but that nothing moves fast.  The scripted controller in
# `pochi_rl.control.standup` solves the same problem analytically; this spec is
# what a learned policy has to be shaped by to end up doing the same thing.


@dataclass(frozen=True)
class StandUpReferenceSpec:
  """The height ramp the policy is asked to follow.

  Rewarding the standing height alone gets the robot up, but in 0.8 s: the
  discounted value of standing early beats any speed penalty that is not itself
  large enough to distort the rest of the task.  Timing is therefore made part
  of the objective rather than left to be traded against.  The ramp is the same
  shape the scripted controller uses -- smoothstep from the collapsed height to
  the nominal stance, after a pause to settle -- so a policy that tracks it is
  reproducing that manoeuvre rather than merely arriving at the same pose.

  ``pochi_rl.control.standup.reference_height`` is the one implementation, so
  the scripted controller and the reward cannot drift apart.
  """

  settle_s: float = 0.75
  rise_s: float = 6.0


@dataclass(frozen=True)
class StandUpRewardWeights:
  # Following the ramp is the task, so this has to outweigh every other term.
  # The kernel is exp(-|error| / std), not the usual squared one: squared goes
  # flat 25 cm away, which is exactly where the robot starts, and a reward with
  # no gradient at the start state is no reward at all.
  track_height: float = 5.0
  # The long-range half of the tracking objective; see
  # `rewards.height_tracking_error`.  Without it the exponential kernel alone
  # has a plateau the policy is happy to sit on, 20 cm below the reference.
  height_error: float = -8.0
  upright: float = 1.0
  # Hold the stance the velocity task expects, so a policy hand-off is clean.
  # Faded in over the ramp -- see `rewards.posture_along_ramp` for why an
  # ungated version wrecks the manoeuvre.
  posture: float = -0.4
  feet_contact: float = 0.3
  # Push with the feet, not the knees.  Only counted once the robot is off the
  # floor -- while it is lying down, the belly and thighs *are* resting on the
  # ground and there is nothing wrong with that.
  non_foot_contact: float = -2.0
  # Slowness.  These are an order of magnitude heavier than the velocity task's:
  # there is no gait to keep up with, and the whole point is a sedate motion.
  joint_vel_l2: float = -0.02
  joint_acc_l2: float = -2.5e-7
  action_rate_l2: float = -0.05
  joint_torques_l2: float = -2.0e-4
  joint_speed_limit: float = -2.0
  lin_vel_z_l2: float = -1.0
  ang_vel_xy_l2: float = -0.1
  termination: float = -50.0


@dataclass(frozen=True)
class StandUpSafetySpec:
  """Motor envelope for the stand-up policy.

  This exists so the policy is safe to stand next to on hardware, so the caps
  are a property of the motor model rather than a reward penalty: a penalty is a
  preference the policy may decide to pay, a torque-speed curve is not.

  ``motor_speed_limit`` is the no-load speed handed to mjlab's DC-motor model.
  Torque falls linearly to zero there and reverses beyond it, so the motor
  cannot *drive* a joint faster than this.  It is not an absolute speed cap:
  gravity and the momentum of the rest of the machine can still overhaul a
  joint, and the motor's ability to brake that is itself bounded by
  ``motor_effort_limit``.  Measured worst case, slamming the action between its
  extremes every 0.6 s -- far more violent than anything a trained policy
  emits -- the joints reach about 3.9 rad/s, against 18 rad/s for the stock RS02
  envelope.  The scripted manoeuvre in `pochi_rl.control.standup`, for scale,
  peaks at 0.43 rad/s.

  ``motor_effort_limit`` is the RS02's *continuous* rating rather than its
  17 N.m stall torque.  Standing up needs 3.6 N.m at the worst moment, so this
  is ample, and halving the torque roughly halves the worst-case overspeed above
  as well.
  """

  motor_speed_limit: float = 2.0  # [rad/s] ~19 rpm at the output shaft
  soft_speed_limit: float = 1.0  # [rad/s] where the speed penalty starts
  motor_effort_limit: float = RS02_RATED_TORQUE_NM


@dataclass(frozen=True)
class StandUpControlSpec:
  decimation: int = 4
  sim_dt: float = 0.005
  # The scripted manoeuvre takes 8.25 s end to end.  Twice that leaves room to
  # get up and then be judged on holding the stance.
  episode_length_s: float = 16.0
  # The velocity task's scale.  This task spans far more of the joint range, but
  # the fix for that is where the action range is centred, not how wide it is --
  # see `pochi_rl.mjlab.actions.CenteredJointPositionAction`.  Widening it to
  # 0.7 was tried and thrashes the robot over: reward collapses to -5 with the
  # episodes ending on the `flipped` termination.
  action_scale: float = 0.25
  # How far above the collapsed height the robot counts as "off the floor", and
  # so where the non-foot contact penalty switches on.
  lifted_margin: float = 0.03


@dataclass(frozen=True)
class StandUpResetSpec:
  """How much the collapsed start pose is randomised."""

  joint_pos: tuple[float, float] = (-0.1, 0.1)
  joint_vel: tuple[float, float] = (-0.05, 0.05)
  base_height: tuple[float, float] = (0.0, 0.03)
  base_roll_pitch: tuple[float, float] = (-0.15, 0.15)


@dataclass(frozen=True)
class PochiStandUpSpec:
  reference: StandUpReferenceSpec = field(default_factory=StandUpReferenceSpec)
  rewards: StandUpRewardWeights = field(default_factory=StandUpRewardWeights)
  safety: StandUpSafetySpec = field(default_factory=StandUpSafetySpec)
  control: StandUpControlSpec = field(default_factory=StandUpControlSpec)
  reset: StandUpResetSpec = field(default_factory=StandUpResetSpec)
  observations: ObservationSpec = field(default_factory=ObservationSpec)
  events: EventSpec = field(default_factory=EventSpec)
  # Width of the height-tracking kernel [m].  Wide enough that the term still
  # has a usable slope when the robot is most of the way off the reference,
  # narrow enough to be worth actually tracking.
  height_std: float = 0.10
  upright_std: float = 0.3

  @property
  def policy_dim(self) -> int:
    # base lin vel, base ang vel, projected gravity, the commanded height and
    # how far through the ramp it is, joint pos/vel, last action.  No velocity
    # command: this task has none.
    return 3 + 3 + 3 + 2 + 12 + 12 + 12


POCHI_STANDUP_SPEC = PochiStandUpSpec()
