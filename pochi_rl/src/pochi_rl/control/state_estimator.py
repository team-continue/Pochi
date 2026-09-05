"""Body-frame base linear-velocity estimator for hardware deployment.

The velocity task's policy observes ``base_lin_vel`` (see
``pochi_rl.task_spec.ObservationSpec``), which in simulation is read straight
off the physics state. Real hardware has no sensor for it: the IMU
(``pochi_hardware``'s ``ImuState``) reports orientation, angular velocity and
acceleration only. This estimates it instead, by fusing the IMU with
leg-odometry from the joint encoders under a no-slip assumption -- the
standard approach for legged robots without a body-velocity sensor (see e.g.
Bloesch et al., "State Estimation for Legged Robots", and the MIT Cheetah
estimator it inspired).

Both halves of the model are linear in the filter's own state for a given IMU
reading, so this is an ordinary time-varying linear Kalman filter, not an
EKF -- nothing here needs the filter's own dynamics linearized. The only
Jacobian involved is a geometric one, `leg_kinematics.forward3d_jacobian`,
used to turn joint velocities into a foot velocity.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pochi_rl.control import leg_kinematics as lk
from pochi_rl.robot import JOINT_KINDS, LEGS, MOTOR_SIGN

GRAVITY = 9.81


def quat_conjugate(q: np.ndarray) -> np.ndarray:
  w, x, y, z = q
  return np.array([w, -x, -y, -z])


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  """Rotate ``v`` by the quaternion ``q = (w, x, y, z)``."""
  w = q[0]
  qv = q[1:]
  t = 2.0 * np.cross(qv, v)
  return v + w * t + np.cross(qv, t)


def projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
  """True gravitational acceleration, in the body frame -- what the filter's
  own process model needs (``v_dot = ... + g_body``, a real Newton's-second-law
  term, so it has to be in m/s^2).

  This is *not* what the policy's ``projected_gravity`` observation wants --
  mjlab's ``EntityData.gravity_vec_w`` is the unit vector ``[0, 0, -1]``
  (`mjlab.entity.entity.EntityArticulationInfoCfg` /
  `Entity.__init__`), not one scaled by 9.81. Feeding this function's output
  to the policy overstates the tilt signal by 9.81x -- use
  `gravity_direction_body` for that instead.

  ``quaternion_wxyz`` is the IMU's body-to-world orientation, the convention
  both ``ImuState`` and mjlab use.
  """
  world_gravity = np.array([0.0, 0.0, -GRAVITY])
  return quat_rotate(
    quat_conjugate(np.asarray(quaternion_wxyz, dtype=float)), world_gravity
  )


def gravity_direction_body(quaternion_wxyz: np.ndarray) -> np.ndarray:
  """Unit gravity direction in the body frame -- the actual quantity behind
  the policy's ``projected_gravity`` observation term. See `projected_gravity`
  for why that name is reserved for the (differently scaled) physical one the
  filter needs instead."""
  return projected_gravity(quaternion_wxyz) / GRAVITY


def skew(v: np.ndarray) -> np.ndarray:
  return np.array(
    [
      [0.0, -v[2], v[1]],
      [v[2], 0.0, -v[0]],
      [-v[1], v[0], 0.0],
    ]
  )


@dataclass(frozen=True)
class StateEstimatorConfig:
  """Filter tuning. Starting points only -- ``contact_torque_threshold_nm``
  in particular has to be checked against real telemetry (there is no force
  sensor to calibrate it against in sim)."""

  process_noise_velocity: float = 0.5  # (m/s)^2 / s, IMU-driven drift
  process_noise_bias: float = 1.0e-4  # (m/s^2)^2 / s, accel-bias random walk
  measurement_noise: float = 0.02  # (m/s)^2, per contact-leg pseudo-measurement
  contact_torque_threshold_nm: float = 3.0
  initial_velocity_variance: float = 1.0
  initial_bias_variance: float = 0.1


class BodyVelocityEstimator:
  """Strapdown IMU + leg-odometry Kalman filter for ``base_lin_vel``.

  State is ``[v_body (3), accel_bias (3)]``. Process model:
  ``v_dot = (a_meas - bias) - omega x v + g_body``, ``bias_dot = 0``.
  Measurement, once per leg currently in contact: the foot's world-frame
  velocity should be zero (no slip), which rearranges to a direct
  pseudo-measurement of ``v_body``.
  """

  def __init__(self, cfg: StateEstimatorConfig | None = None) -> None:
    self.cfg = cfg or StateEstimatorConfig()
    self.reset()

  def reset(self, initial_velocity: np.ndarray | None = None) -> None:
    self.x = np.zeros(6)
    if initial_velocity is not None:
      self.x[:3] = initial_velocity
    self.P = np.diag(
      [self.cfg.initial_velocity_variance] * 3 + [self.cfg.initial_bias_variance] * 3
    )

  @property
  def velocity(self) -> np.ndarray:
    return self.x[:3].copy()

  @property
  def bias(self) -> np.ndarray:
    return self.x[3:].copy()

  def predict(
    self,
    accel_body: np.ndarray,
    gyro_body: np.ndarray,
    gravity_body: np.ndarray,
    dt: float,
  ) -> None:
    v = self.x[:3]
    bias = self.x[3:]
    v_dot = (accel_body - bias) - np.cross(gyro_body, v) + gravity_body
    self.x[:3] = v + dt * v_dot
    # bias_dot == 0: only its covariance grows, via Q below.

    jacobian = np.eye(6)
    jacobian[:3, :3] -= dt * skew(gyro_body)
    jacobian[:3, 3:] -= dt * np.eye(3)
    process_noise = np.diag(
      [self.cfg.process_noise_velocity * dt] * 3
      + [self.cfg.process_noise_bias * dt] * 3
    )
    self.P = jacobian @ self.P @ jacobian.T + process_noise

  def update_leg(self, foot_velocity_measured: np.ndarray) -> None:
    """One no-slip pseudo-measurement of ``v_body`` from a leg in contact."""
    H = np.zeros((3, 6))
    H[:, :3] = np.eye(3)
    R = np.eye(3) * self.cfg.measurement_noise
    innovation = foot_velocity_measured - H @ self.x
    innovation_cov = H @ self.P @ H.T + R
    kalman_gain = self.P @ H.T @ np.linalg.inv(innovation_cov)
    self.x = self.x + kalman_gain @ innovation
    self.P = (np.eye(6) - kalman_gain @ H) @ self.P

  def step(
    self,
    *,
    accel_body: np.ndarray,
    gyro_body: np.ndarray,
    quaternion_wxyz: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    joint_torque: np.ndarray,
    dt: float,
  ) -> np.ndarray:
    """Advance the filter one control tick and return the updated ``v_body``.

    ``joint_pos``/``joint_vel``/``joint_torque`` are motor-frame arrays in
    ``JOINT_NAMES`` order (hardware encoder convention); this converts to the
    canonical frame `leg_kinematics` expects internally.
    """
    accel_body = np.asarray(accel_body, dtype=float)
    gyro_body = np.asarray(gyro_body, dtype=float)
    g_body = projected_gravity(quaternion_wxyz)
    self.predict(accel_body, gyro_body, g_body, dt)

    for leg_index, leg in enumerate(LEGS):
      s = slice(3 * leg_index, 3 * leg_index + 3)
      names = [f"{leg}_{kind}" for kind in JOINT_KINDS]
      signs = np.array([MOTOR_SIGN[name] for name in names])
      q = joint_pos[s] * signs
      qdot = joint_vel[s] * signs
      torque = joint_torque[s]

      in_contact = (
        abs(torque[1]) + abs(torque[2])
      ) > self.cfg.contact_torque_threshold_nm
      if not in_contact:
        continue

      r_foot = np.array(lk.forward3d(leg, *q))
      jacobian = lk.forward3d_jacobian(leg, *q)
      foot_velocity_body = jacobian @ qdot
      measured_v = -(np.cross(gyro_body, r_foot) + foot_velocity_body)
      self.update_leg(measured_v)

    return self.velocity
