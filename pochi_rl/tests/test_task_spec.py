from pochi_rl.robot import DEFAULT_JOINT_POS, FOOT_BODIES, JOINT_NAMES
from pochi_rl.task_spec import POCHI_TASK_SPEC


def test_joint_constants_match_expected_robot_shape() -> None:
  assert len(JOINT_NAMES) == 12
  assert tuple(DEFAULT_JOINT_POS) == JOINT_NAMES
  assert len(FOOT_BODIES) == 4


def test_policy_observation_dim_arithmetic() -> None:
  assert POCHI_TASK_SPEC.observations.policy_dim == 48
  assert POCHI_TASK_SPEC.control.policy_hz == 50.0


def test_curriculum_ends_at_the_full_command_range() -> None:
  """The last stage must reach the ranges the task actually targets.

  The command term is initialised at the *first* stage, so if the final stage
  fell short the robot would silently never train at full speed.
  """
  from pochi_rl.task_spec import POCHI_TASK_SPEC as spec

  final = spec.curriculum.stages[-1]
  assert final.lin_vel_x == spec.commands.lin_vel_x
  assert final.lin_vel_y == spec.commands.lin_vel_y
  assert final.ang_vel_z == spec.commands.ang_vel_z


def test_curriculum_stages_widen_monotonically() -> None:
  """Each stage must start later and cover at least as much as the last."""
  from pochi_rl.task_spec import POCHI_TASK_SPEC as spec

  stages = spec.curriculum.stages
  assert stages[0].step == 0, "the first stage has to apply from step 0"
  for previous, stage in zip(stages, stages[1:]):
    assert stage.step > previous.step
    for axis in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
      lo, hi = getattr(stage, axis)
      prev_lo, prev_hi = getattr(previous, axis)
      assert lo <= prev_lo and hi >= prev_hi, (axis, stage.step)


def test_standing_thresholds_are_below_the_slowest_command() -> None:
  """A command the robot must walk at may never count as "standing".

  With mjlab's 0.5 default, every command under 0.5 m/s got the stand-still
  penalty and no air-time reward, so the policy learned to stand still through
  them -- measured at 0.00 rad/s of joint motion under a 0.2 m/s command.
  """
  from pochi_rl.task_spec import POCHI_TASK_SPEC as spec

  slowest_walk = min(stage.lin_vel_x[1] for stage in spec.curriculum.stages)
  for name in ("stand_still", "feet_air_time"):
    threshold = getattr(spec.thresholds, name)
    assert threshold < slowest_walk / 2.0, (
      f"{name} threshold {threshold} swallows commands the robot must walk at "
      f"(slowest curriculum stage tops out at {slowest_walk})"
    )
