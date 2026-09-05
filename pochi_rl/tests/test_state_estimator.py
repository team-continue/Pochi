"""The hardware base-velocity Kalman filter: quaternion math and end-to-end
convergence, entirely off-hardware (synthetic IMU/encoder streams).
"""

from __future__ import annotations

import numpy as np
import pytest

from pochi_rl.control import leg_kinematics as lk
from pochi_rl.control.state_estimator import (
  BodyVelocityEstimator,
  StateEstimatorConfig,
  projected_gravity,
  quat_conjugate,
  quat_rotate,
)
from pochi_rl.robot import DEFAULT_JOINT_POS, JOINT_KINDS, JOINT_NAMES, LEGS, MOTOR_SIGN

GRAVITY = 9.81


def test_quat_rotate_identity_and_conjugate() -> None:
  identity = np.array([1.0, 0.0, 0.0, 0.0])
  v = np.array([1.0, 2.0, 3.0])
  assert quat_rotate(identity, v) == pytest.approx(v)
  assert quat_conjugate(identity) == pytest.approx(identity)


def test_projected_gravity_matches_a_known_tilt() -> None:
  """90 deg roll about +x should read world -z entirely on the body -y axis."""
  half = np.pi / 4.0
  q = np.array([np.cos(half), np.sin(half), 0.0, 0.0])
  g = projected_gravity(q)
  assert g == pytest.approx([0.0, -GRAVITY, 0.0], abs=1e-9)


def _default_joint_pos() -> np.ndarray:
  return np.array([DEFAULT_JOINT_POS[name] for name in JOINT_NAMES])


def test_filter_converges_to_zero_velocity_when_stationary() -> None:
  """Wrong initial velocity, but every measurement says the robot isn't
  moving (flat ground, all four feet planted, no joint motion) -- the filter
  has to walk itself back to ~0."""
  estimator = BodyVelocityEstimator()
  estimator.reset(initial_velocity=np.array([1.0, -0.5, 0.2]))

  pos = _default_joint_pos()
  zero_vel = np.zeros(12)
  torque = np.full(12, 5.0)  # every leg reads as "in contact"
  quaternion = np.array([1.0, 0.0, 0.0, 0.0])
  accel = np.array([0.0, 0.0, GRAVITY])  # stationary: exactly cancels g_body
  gyro = np.zeros(3)

  v = estimator.velocity
  for _ in range(200):
    v = estimator.step(
      accel_body=accel,
      gyro_body=gyro,
      quaternion_wxyz=quaternion,
      joint_pos=pos,
      joint_vel=zero_vel,
      joint_torque=torque,
      dt=0.02,
    )

  assert v == pytest.approx([0.0, 0.0, 0.0], abs=1e-2)


def test_filter_tracks_a_constructed_constant_velocity() -> None:
  """Construct FL joint velocities that make its no-slip measurement read
  exactly ``true_velocity`` (the other three legs report no contact), and
  check the filter converges to it. This exercises the real wiring --
  MOTOR_SIGN conversion, per-leg contact gating, the Jacobian -- rather than
  just the filter math in isolation."""
  estimator = BodyVelocityEstimator(StateEstimatorConfig(measurement_noise=1e-6))

  pos = _default_joint_pos()
  quaternion = np.array([1.0, 0.0, 0.0, 0.0])
  gyro = np.zeros(3)
  accel = np.array([0.0, 0.0, GRAVITY])
  true_velocity = np.array([0.3, 0.05, 0.0])

  leg = "FL"
  leg_index = LEGS.index(leg)
  names = [f"{leg}_{kind}" for kind in JOINT_KINDS]
  signs = np.array([MOTOR_SIGN[name] for name in names])
  q_canonical = pos[3 * leg_index : 3 * leg_index + 3] * signs
  jacobian = lk.forward3d_jacobian(leg, *q_canonical)
  # measured_v = -(omega x r_foot + J @ qdot_canonical); omega == 0 here, so
  # solving J @ qdot_canonical == -true_velocity makes measured_v ==
  # true_velocity exactly.
  qdot_canonical = np.linalg.solve(jacobian, -true_velocity)
  qdot_motor = qdot_canonical * signs  # signs are +/-1, so this is self-inverse

  joint_vel = np.zeros(12)
  joint_vel[3 * leg_index : 3 * leg_index + 3] = qdot_motor
  torque = np.zeros(12)
  torque[3 * leg_index : 3 * leg_index + 3] = 5.0  # only FL "in contact"

  v = estimator.velocity
  for _ in range(300):
    v = estimator.step(
      accel_body=accel,
      gyro_body=gyro,
      quaternion_wxyz=quaternion,
      joint_pos=pos,
      joint_vel=joint_vel,
      joint_torque=torque,
      dt=0.02,
    )

  assert v == pytest.approx(true_velocity, abs=1e-2)


def test_no_leg_in_contact_falls_back_to_imu_dead_reckoning() -> None:
  """No telemetered torque anywhere -> no measurement updates -> the filter
  just integrates the IMU, which for a real (non-gravity-cancelling) push
  should show a nonzero velocity change after 1 s."""
  estimator = BodyVelocityEstimator()
  pos = _default_joint_pos()
  zero_vel = np.zeros(12)
  no_contact_torque = np.zeros(12)
  quaternion = np.array([1.0, 0.0, 0.0, 0.0])
  gyro = np.zeros(3)
  accel = np.array([1.0, 0.0, GRAVITY])  # 1 m/s^2 of real forward acceleration

  dt = 0.02
  v = estimator.velocity
  for _ in range(int(1.0 / dt)):
    v = estimator.step(
      accel_body=accel,
      gyro_body=gyro,
      quaternion_wxyz=quaternion,
      joint_pos=pos,
      joint_vel=zero_vel,
      joint_torque=no_contact_torque,
      dt=dt,
    )

  assert v[0] == pytest.approx(1.0, abs=0.05)
