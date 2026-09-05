"""Check ``BodyVelocityEstimator`` against real physics, not just algebra.

``tests/test_state_estimator.py`` only exercises the filter's math in
isolation (hand-constructed synthetic streams). This instead drives the real
``Pochi-Deliberate-Walk-v0`` checkpoint in closed loop against plain MuJoCo
(``assets/pochi/scene_flat.xml``, the same standalone model
``pochi_rl.control.mujoco_driver`` uses for stand-up) -- the policy's own
``base_lin_vel`` input comes from the estimator, fed only IMU sensors and
joint encoders, exactly as it would be on hardware. Ground truth comes from
MuJoCo's ``imu_vel`` velocimeter, which the estimator never sees.

This is not a full sim-to-real check: ``scene_flat.xml``'s plain
``<position>`` actuators have the same kp/kv as training but not the RS02
torque-speed clamp mjlab's actuator model applies, so joint tracking is a bit
stiffer here than during training. It is, however, real rigid-body contact
dynamics -- a much closer proxy for "does the estimator work under a real
gait" than any hand-built synthetic stream can be.

Usage (needs mujoco + torch; neither needs the rest of the mjlab/rsl-rl
training stack):

  uv run python scripts/verify_deliberate_walk_estimator.py \\
      --checkpoint checkpoints/pochi_deliberate_walk_model_950.pt
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import mujoco
import numpy as np
import torch

from pochi_rl.control.state_estimator import (
  BodyVelocityEstimator,
  StateEstimatorConfig,
  projected_gravity,
)
from pochi_rl.policy_io import load_actor
from pochi_rl.robot import DEFAULT_JOINT_POS, JOINT_NAMES, NOMINAL_BASE_HEIGHT
from pochi_rl.task_spec import POCHI_TASK_SPEC

SCENE_XML = Path(__file__).resolve().parents[1] / "assets" / "pochi" / "scene_flat.xml"

# Mirrors pochi_hardware/src/run_walk_hardware.py -- see that file for why
# these specific numbers (Pochi-Deliberate-Walk-v0's own trained envelope,
# not the general velocity task's).
GAIT_PERIOD_S = 2.8
COMMAND_THRESHOLD = 0.025
COMMAND = np.array([0.10, 0.0, 0.0])  # the validated "walk slowly" demo

_DEFAULT_JOINT_POS = np.array([DEFAULT_JOINT_POS[name] for name in JOINT_NAMES])


def gait_phase(elapsed_s: float, command: np.ndarray) -> np.ndarray:
  moving = (math.hypot(command[0], command[1]) + abs(command[2])) > COMMAND_THRESHOLD
  if not moving:
    return np.zeros(2)
  angle = 2.0 * math.pi * elapsed_s / GAIT_PERIOD_S
  return np.array([math.sin(angle), math.cos(angle)])


class Rollout:
  """Standing-start MuJoCo instance, stepped like ``mujoco_driver.StandUpSim``."""

  def __init__(self) -> None:
    self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    self.data = mujoco.MjData(self.model)
    self.decimation = POCHI_TASK_SPEC.control.decimation
    self.control_dt = self.decimation * self.model.opt.timestep

    self._qpos_adr = np.array(
      [
        self.model.jnt_qposadr[
          mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ]
        for name in JOINT_NAMES
      ]
    )
    self._qvel_adr = np.array(
      [
        self.model.jnt_dofadr[
          mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        ]
        for name in JOINT_NAMES
      ]
    )
    self._act_ids = np.array(
      [
        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_act")
        for name in JOINT_NAMES
      ]
    )
    self._base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "base_link")

    mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[0:3] = (0.0, 0.0, NOMINAL_BASE_HEIGHT + 0.01)
    self.data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
    self.data.qpos[self._qpos_adr] = _DEFAULT_JOINT_POS
    self.data.ctrl[self._act_ids] = _DEFAULT_JOINT_POS
    mujoco.mj_forward(self.model, self.data)
    for _ in range(int(round(0.2 / self.model.opt.timestep))):
      mujoco.mj_step(self.model, self.data)

  @property
  def joint_pos(self) -> np.ndarray:
    return self.data.qpos[self._qpos_adr].copy()

  @property
  def joint_vel(self) -> np.ndarray:
    return self.data.qvel[self._qvel_adr].copy()

  @property
  def joint_torque(self) -> np.ndarray:
    return self.data.actuator_force[self._act_ids].copy()

  @property
  def base_height(self) -> float:
    return float(self.data.xpos[self._base_id][2])

  def imu(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(quaternion_wxyz, gyro, accel, true_velocity) -- the last one is the
    ground truth the estimator never sees."""
    quat = self.data.sensor("imu_quat").data.copy()
    gyro = self.data.sensor("imu_gyro").data.copy()
    accel = self.data.sensor("imu_accel").data.copy()
    true_velocity = self.data.sensor("imu_vel").data.copy()
    return quat, gyro, accel, true_velocity

  def set_targets(self, targets: np.ndarray) -> None:
    self.data.ctrl[self._act_ids] = targets

  def substep(self) -> None:
    mujoco.mj_step(self.model, self.data)


def build_observation(
  *, base_lin_vel, base_ang_vel, gravity_body, command, joint_pos, joint_vel,
  last_action, elapsed_s,
) -> np.ndarray:
  phase = gait_phase(elapsed_s, command)
  return np.concatenate(
    [
      base_lin_vel,
      base_ang_vel,
      gravity_body,
      command,
      joint_pos - _DEFAULT_JOINT_POS,
      joint_vel,
      last_action,
      phase,
    ]
  ).astype(np.float32)


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--checkpoint",
    default=str(
      Path(__file__).resolve().parents[1]
      / "checkpoints"
      / "pochi_deliberate_walk_model_950.pt"
    ),
  )
  parser.add_argument("--duration", type=float, default=14.0)
  parser.add_argument("--contact-torque-threshold", type=float, default=3.0)
  args = parser.parse_args()

  actor = load_actor(args.checkpoint)
  estimator = BodyVelocityEstimator(
    StateEstimatorConfig(contact_torque_threshold_nm=args.contact_torque_threshold)
  )
  sim = Rollout()

  ticks = int(round(args.duration / sim.control_dt))
  last_action = np.zeros(12)
  elapsed = 0.0
  errors = []
  min_height = sim.base_height
  start_xy = sim.data.qpos[0:2].copy()

  print(f"{'t':>6s} {'v_true_x':>9s} {'v_est_x':>9s} {'v_true_y':>9s} "
        f"{'v_est_y':>9s} {'height':>7s}")

  for tick in range(ticks):
    quat, gyro, accel, true_velocity = sim.imu()
    v_est = estimator.step(
      accel_body=accel,
      gyro_body=gyro,
      quaternion_wxyz=quat,
      joint_pos=sim.joint_pos,
      joint_vel=sim.joint_vel,
      joint_torque=sim.joint_torque,
      dt=sim.control_dt,
    )
    g_body = projected_gravity(quat)
    obs = build_observation(
      base_lin_vel=v_est,
      base_ang_vel=gyro,
      gravity_body=g_body,
      command=COMMAND,
      joint_pos=sim.joint_pos,
      joint_vel=sim.joint_vel,
      last_action=last_action,
      elapsed_s=elapsed,
    )
    with torch.inference_mode():
      action = actor(torch.from_numpy(obs).unsqueeze(0)).squeeze(0).numpy()
    last_action = action
    targets = _DEFAULT_JOINT_POS + POCHI_TASK_SPEC.control.action_scale * action
    sim.set_targets(targets)

    for _ in range(sim.decimation):
      sim.substep()

    errors.append(v_est - true_velocity)
    min_height = min(min_height, sim.base_height)
    if tick % 25 == 0:
      print(
        f"{elapsed:6.2f} {true_velocity[0]:9.3f} {v_est[0]:9.3f} "
        f"{true_velocity[1]:9.3f} {v_est[1]:9.3f} {sim.base_height:7.3f}"
      )
    elapsed += sim.control_dt

  errors = np.array(errors)
  drift = sim.data.qpos[0:2] - start_xy
  rmse = np.sqrt((errors**2).mean(axis=0))
  print()
  print(f"velocity estimate RMSE (x, y, z) [m/s]: {rmse}")
  print(f"max abs error (x, y, z) [m/s]:          {np.abs(errors).max(axis=0)}")
  print(f"min base height during rollout [m]:     {min_height:.3f}")
  print(f"net xy displacement [m]:                {drift}")
  fell = min_height < NOMINAL_BASE_HEIGHT * 0.6
  print(f"fell over: {fell}")
  return 1 if fell else 0


if __name__ == "__main__":
  raise SystemExit(main())
