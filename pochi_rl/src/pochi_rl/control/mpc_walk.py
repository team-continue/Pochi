"""MuJoCo closed-loop MPC crawl; independent of the RL task/driver.

Only joint torques drive the free-floating robot. Contact forces planned by
the centroidal MPC become stance torques through the measured foot Jacobians;
swing feet track smooth Cartesian trajectories using impedance control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from pochi_rl.control.centroidal_mpc import CentroidalMPC, MPCConfig
from pochi_rl.robot import (
  BASE_BODY,
  DEFAULT_JOINT_POS,
  FOOT_GEOMS,
  FOOT_SITES,
  JOINT_NAMES,
  NOMINAL_BASE_HEIGHT,
  RS02_NO_LOAD_SPEED_RAD_S,
  RS02_PEAK_TORQUE_NM,
)

SCENE_XML = Path(__file__).resolve().parents[3] / "assets/pochi/scene_flat.xml"


@dataclass(frozen=True)
class WalkConfig:
  speed: float = 0.06
  period: float = 2.4
  swing_duration: float = 0.4
  step_height: float = 0.045
  start_delay: float = 1.0
  ramp_duration: float = 2.0
  mpc: MPCConfig = field(default_factory=MPCConfig)

  def __post_init__(self) -> None:
    if not np.isfinite(self.speed) or abs(self.speed) > 0.09:
      raise ValueError("speed must be finite and within +/-0.09 m/s")
    for name in ("period", "swing_duration", "step_height", "ramp_duration"):
      if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
        raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(self.start_delay) or self.start_delay < 0:
      raise ValueError("start_delay must be finite and nonnegative")
    if self.swing_duration >= self.period / 4:
      raise ValueError("crawl requires swing_duration < period / 4")


def swing_trajectory(
  start: np.ndarray, end: np.ndarray, phase: float, duration: float, height: float
) -> tuple[np.ndarray, np.ndarray]:
  """Quintic horizontal transfer and a vertical bump with zero endpoint speed."""
  s = float(np.clip(phase, 0, 1))
  blend = 10 * s**3 - 15 * s**4 + 6 * s**5
  rate = (30 * s**2 - 60 * s**3 + 30 * s**4) / duration
  pos = start + blend * (end - start)
  vel = rate * (end - start)
  pos[2] += height * 16 * s**2 * (1 - s) ** 2
  vel[2] += height * 32 * s * (1 - s) * (1 - 2 * s) / duration
  return pos, vel


class MPCWalkSim:
  """One robot, CPU physics, and a receding-horizon force planner."""

  def __init__(self, cfg: WalkConfig | None = None) -> None:
    self.cfg = cfg or WalkConfig()
    self.model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    self.data = mujoco.MjData(self.model)
    m = self.model
    self.base_id = m.body(BASE_BODY).id
    self.site_ids = np.array([m.site(name).id for name in FOOT_SITES])
    self.foot_geom_ids = {m.geom(name).id for name in FOOT_GEOMS}
    self.ground_id = m.geom("ground").id
    self.joint_ids = np.array([m.joint(name).id for name in JOINT_NAMES])
    self.qpos_ids = m.jnt_qposadr[self.joint_ids]
    self.dof_ids = m.jnt_dofadr[self.joint_ids]
    self.act_ids = np.array([m.actuator(f"{name}_act").id for name in JOINT_NAMES])
    self.nominal_q = np.array([DEFAULT_JOINT_POS[name] for name in JOINT_NAMES])
    # Convert only this in-memory model's position servos to torque motors.
    m.actuator_gainprm[:] = 0
    m.actuator_gainprm[:, 0] = 1
    m.actuator_biasprm[:] = 0
    m.actuator_ctrlrange[:] = [-RS02_PEAK_TORQUE_NM, RS02_PEAK_TORQUE_NM]
    self.mpc_steps = max(1, round(self.cfg.mpc.dt / m.opt.timestep))
    if not np.isclose(self.mpc_steps * m.opt.timestep, self.cfg.mpc.dt):
      raise ValueError("MPC dt must be a multiple of the physics timestep")
    self.reset()

  @property
  def feet(self) -> np.ndarray:
    return self.data.site_xpos[self.site_ids].copy()

  @property
  def base_height(self) -> float:
    return float(self.data.xpos[self.base_id, 2])

  def contacts_at(self, time: float) -> tuple[np.ndarray, np.ndarray]:
    contacts = np.ones(4, dtype=bool)
    phases = np.zeros(4)
    if time < self.cfg.start_delay or abs(self.cfg.speed) < 1e-8:
      return contacts, phases
    # FR -> RL -> FL -> RR; one swing per quarter-cycle, then all-foot support.
    for slot, leg in enumerate((1, 2, 0, 3)):
      local = (
        time - self.cfg.start_delay - slot * self.cfg.period / 4
      ) % self.cfg.period
      if local < self.cfg.swing_duration:
        contacts[leg] = False
        phases[leg] = local / self.cfg.swing_duration
    return contacts, phases

  def command_at(self, time: float) -> tuple[float, float]:
    t = max(0.0, time - self.cfg.start_delay)
    ramp = self.cfg.ramp_duration
    speed = self.cfg.speed * min(t / ramp, 1.0)
    x = self.cfg.speed * (t * t / (2 * ramp) if t < ramp else t - ramp / 2)
    return x, speed

  def reset(self, model=None, data=None) -> None:
    m, d = self.model, self.data
    mujoco.mj_resetData(m, d)
    d.qpos[self.qpos_ids] = self.nominal_q
    d.qpos[2] = NOMINAL_BASE_HEIGHT
    mujoco.mj_forward(m, d)
    self.origin = d.subtree_com[self.base_id].copy()
    self.nominal_feet = self.feet - d.xpos[self.base_id]
    self.ground_z = self.feet[:, 2].copy()
    self.swing_start = self.feet
    self.swing_end = self.feet
    self.foot_targets = self.feet
    self.contacts = np.ones(4, dtype=bool)
    self.force = np.zeros((4, 3))
    # Composite inertia with all links locked at the nominal stance, about COM.
    inertia = np.zeros((3, 3))
    for b in range(1, m.nbody):
      rot = d.ximat[b].reshape(3, 3)
      r = d.xipos[b] - self.origin
      inertia += rot @ np.diag(m.body_inertia[b]) @ rot.T
      inertia += m.body_mass[b] * (np.dot(r, r) * np.eye(3) - np.outer(r, r))
    self.planner = CentroidalMPC(float(m.body_mass.sum()), inertia, self.cfg.mpc)
    self.steps = 0
    self.max_torque = 0.0
    self.min_height = self.base_height
    self.max_tilt = 0.0
    self.nonfoot_contact_force = 0.0
    self.liftoffs = np.zeros(4, dtype=int)
    self.max_clearance = np.zeros(4)
    self.failed = False

  def state(self) -> np.ndarray:
    m, d = self.model, self.data
    jac = np.zeros((3, m.nv))
    mujoco.mj_jacSubtreeCom(m, d, jac, self.base_id)
    rot = d.xmat[self.base_id].reshape(3, 3)
    velocity = np.zeros(6)
    mujoco.mj_objectVelocity(m, d, mujoco.mjtObj.mjOBJ_BODY, self.base_id, velocity, 0)
    return np.r_[
      d.subtree_com[self.base_id],
      Rotation.from_matrix(rot).as_euler("xyz"),
      jac @ d.qvel,
      velocity[:3],
    ]

  def _landing(self, leg: int, state: np.ndarray, speed: float) -> np.ndarray:
    end = self.data.xpos[self.base_id] + self.nominal_feet[leg]
    stance = self.cfg.period - self.cfg.swing_duration
    end[0] += speed * stance / 2 + 0.12 * (state[6] - speed)
    end[1] += 0.1 * state[7]
    end[2] = self.ground_z[leg]
    return end

  def step(self, model=None, data=None) -> None:
    if self.failed:
      return
    m, d = self.model, self.data
    time = self.steps * m.opt.timestep
    state = self.state()
    contacts, phases = self.contacts_at(time)
    _, speed = self.command_at(time + self.cfg.swing_duration)
    for leg in range(4):
      if self.contacts[leg] and not contacts[leg]:
        self.swing_start[leg] = self.feet[leg]
        self.swing_end[leg] = self._landing(leg, state, speed)
        self.liftoffs[leg] += 1
    changed = not np.array_equal(contacts, self.contacts)
    self.contacts = contacts
    if self.steps % self.mpc_steps == 0 or changed:
      n = self.cfg.mpc.horizon
      reference = np.zeros((n, 12))
      planned_feet = np.tile(self.feet, (n, 1, 1))
      schedule = np.zeros((n, 4), dtype=bool)
      for k in range(n):
        future = time + k * self.cfg.mpc.dt
        schedule[k], _ = self.contacts_at(future)
        x, v = self.command_at(future + self.cfg.mpc.dt)
        reference[k, :3] = self.origin + [x, 0, 0]
        reference[k, 6] = v
        for leg in range(4):
          if not contacts[leg]:
            planned_feet[k, leg] = self.swing_end[leg]
      self.force = self.planner.solve(
        state, reference, planned_feet, schedule, d.xmat[self.base_id].reshape(3, 3)
      )
    torque = d.qfrc_bias[self.dof_ids].copy()
    for leg, site in enumerate(self.site_ids):
      jac = np.zeros((3, m.nv))
      mujoco.mj_jacSite(m, d, jac, None, site)
      cols = self.dof_ids[3 * leg : 3 * leg + 3]
      j = jac[:, cols]
      if contacts[leg]:
        foot_force = -self.force[leg]
        self.foot_targets[leg] = self.feet[leg]
      else:
        target, target_vel = swing_trajectory(
          self.swing_start[leg],
          self.swing_end[leg],
          phases[leg],
          self.cfg.swing_duration,
          self.cfg.step_height,
        )
        self.foot_targets[leg] = target
        foot_force = 500 * (target - d.site_xpos[site])
        foot_force += 16 * (target_vel - jac @ d.qvel)
      torque[3 * leg : 3 * leg + 3] += j.T @ foot_force
    torque -= 0.15 * d.qvel[self.dof_ids]
    # RS02 peak envelope; this is a simulation demo, not a hardware safety mode.
    limit = RS02_PEAK_TORQUE_NM * np.clip(
      1 - np.abs(d.qvel[self.dof_ids]) / RS02_NO_LOAD_SPEED_RAD_S, 0, 1
    )
    d.ctrl[self.act_ids] = np.clip(torque, -limit, limit)
    mujoco.mj_step(m, d)
    self.steps += 1
    self.max_torque = max(self.max_torque, float(np.max(np.abs(d.actuator_force))))
    self.min_height = min(self.min_height, self.base_height)
    tilt = np.arccos(np.clip(d.xmat[self.base_id].reshape(3, 3)[2, 2], -1, 1))
    self.max_tilt = max(self.max_tilt, float(tilt))
    self.max_clearance = np.maximum(self.max_clearance, self.feet[:, 2] - self.ground_z)
    force = np.zeros(6)
    for c in range(d.ncon):
      pair = {int(d.contact.geom1[c]), int(d.contact.geom2[c])}
      if self.ground_id in pair and not pair & self.foot_geom_ids:
        mujoco.mj_contactForce(m, d, c, force)
        self.nonfoot_contact_force = max(self.nonfoot_contact_force, abs(force[0]))
    self.failed = (
      not np.isfinite(d.qpos).all()
      or not np.isfinite(d.qvel).all()
      or self.base_height < 0.16
      or tilt > 0.8
    )

  def run(self, duration: float) -> dict:
    for _ in range(round(duration / self.model.opt.timestep)):
      self.step()
      if self.failed:
        break
    return self.metrics()

  def metrics(self) -> dict:
    return {
      "time_s": self.steps * self.model.opt.timestep,
      "distance_m": float(self.data.xpos[self.base_id, 0]),
      "lateral_drift_m": float(self.data.xpos[self.base_id, 1]),
      "height_m": self.base_height,
      "min_height_m": self.min_height,
      "max_tilt_deg": float(np.rad2deg(self.max_tilt)),
      "max_torque_nm": self.max_torque,
      "nonfoot_ground_force_n": self.nonfoot_contact_force,
      "liftoffs": self.liftoffs.tolist(),
      "max_foot_clearance_m": self.max_clearance.tolist(),
      "solver_status": self.planner.status,
      "solver_failures": self.planner.failures,
      "last_solve_ms": self.planner.solve_ms,
      "failed": bool(self.failed),
    }
