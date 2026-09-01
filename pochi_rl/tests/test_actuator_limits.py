"""The RS02 torque envelope has to be a constraint, not a reward.

These tests drive the env hard and assert that no torque ever leaves the motor's
torque-speed curve, which is what makes a learned gait executable on hardware.
"""

from __future__ import annotations

import pytest
import torch

from pochi_rl.robot import (
  RS02_NO_LOAD_SPEED_RAD_S,
  RS02_PEAK_TORQUE_NM,
)


def test_mjlab_spec_drops_the_xml_actuators() -> None:
  """mjlab adds its own <motor> elements; the XML ones must not survive.

  If both sets are present the model silently ends up with 24 actuators and the
  torque-speed curve only governs half of them.
  """
  pytest.importorskip("mujoco")
  from pochi_rl.mjlab.entity_cfg import pochi_spec

  assert len(list(pochi_spec().actuators)) == 0


def test_torque_stays_inside_the_rs02_envelope() -> None:
  """No commanded torque may leave the motor's torque-speed curve.

  Run at ``decimation=1`` so one env step is one physics step: the actuator
  re-samples joint velocity before every substep, so this is the only setting
  where the velocity it used is exactly the one observable before the step.
  ``actuator_force`` for a <motor> is ``gear * ctrl``, i.e. precisely the value
  the DC-motor model clamped, so the comparison is exact rather than a bound.

  ``TOL`` absorbs the sub-millisecond skew between the velocity observable here
  and the one the actuator sampled: the curve loses 0.4 N.m per rad/s, so a
  0.03 rad/s skew is worth ~0.01 N.m.  An actuator that failed to clamp would
  miss by whole newton-metres, so this stays a sharp test.
  """
  TOL = 0.05
  # mjlab clips the velocity fed to the curve at
  # velocity_limit * (1 + effort_limit / saturation_effort); the Pochi actuator
  # sets both efforts to the peak torque, so that is simply twice no-load speed.
  VEL_AT_EFFORT_LIM = 2.0 * RS02_NO_LOAD_SPEED_RAD_S
  pytest.importorskip("mjlab")
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  import pochi_rl  # noqa: F401

  cfg = load_env_cfg("Pochi-Velocity-Flat-v0")
  cfg.scene.num_envs = 4
  cfg.decimation = 1
  cfg.events.pop("base_mass", None)
  env = ManagerBasedRlEnv(cfg=cfg, device="cpu")
  torch.manual_seed(0)
  try:
    env.reset()
    robot = env.scene["robot"]
    assert robot.data.actuator_force.shape[-1] == 12

    saturated = 0
    for _ in range(60):
      vel = robot.data.joint_vel.clone()
      # Saturating actions on purpose: the point is to command more torque than
      # the motor can deliver and check that the sim refuses to deliver it.
      env.step(torch.sign(torch.randn(env.action_space.shape)))
      torque = robot.data.actuator_force

      # The curve is signed, not symmetric in |qd|: it derates torque only in
      # the direction the joint is already turning.  A joint spinning backwards
      # can still be driven forwards at the full peak torque, so bound the
      # signed torque from both sides the way mjlab's _clip_effort does.
      vel = vel.clamp(-VEL_AT_EFFORT_LIM, VEL_AT_EFFORT_LIM)
      ratio = vel / RS02_NO_LOAD_SPEED_RAD_S
      upper = (RS02_PEAK_TORQUE_NM * (1.0 - ratio)).clamp(max=RS02_PEAK_TORQUE_NM)
      lower = (RS02_PEAK_TORQUE_NM * (-1.0 - ratio)).clamp(min=-RS02_PEAK_TORQUE_NM)

      assert torque.abs().max() <= RS02_PEAK_TORQUE_NM + TOL
      over = torch.maximum(torque - upper, lower - torque).max()
      assert float(over) <= TOL, (
        f"torque left the RS02 torque-speed curve by {float(over):.4f} N.m"
      )
      saturated += int(((torque >= upper - TOL) | (torque <= lower + TOL)).sum())

    # Guard against the test passing vacuously on a policy that never pushes
    # hard enough to reach the limit.
    assert saturated > 0, "no joint ever reached the limit; test proves nothing"
  finally:
    env.close()
