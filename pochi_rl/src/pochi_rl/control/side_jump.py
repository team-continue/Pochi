"""Contact-driven lateral jump with Cartesian trajectories and analytic leg IK.

Standalone MuJoCo demonstration: only the twelve torque-limited joint motors
move the floating base. No policy checkpoint or optimizer is required.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np

from pochi_rl.control.leg_kinematics import inverse_3d
from pochi_rl.robot import (
  BASE_BODY,
  DEFAULT_JOINT_POS,
  FOOT_GEOMS,
  FOOT_SITES,
  JOINT_NAMES,
  LEGS,
  NOMINAL_BASE_HEIGHT,
  RS02_NO_LOAD_SPEED_RAD_S,
  RS02_PEAK_TORQUE_NM,
)
from pochi_rl.robot.pochi_constants import FOOT_OFFSET_Y

JUMP_SIZES = ("small", "medium", "large")

SCENE_XML = Path(__file__).resolve().parents[3] / "assets/pochi/scene_flat.xml"


@dataclass(frozen=True)
class SideJumpConfig:
  """Flat-ground demo settings; lateral_speed is a takeoff reference, not distance.

  Other trajectory settings are tunable; successful landing is checked by the
  evaluator, not guaranteed for every combination of positive parameters.
  """

  direction: int = 1
  lateral_speed: float = 2.4
  crouch_height: float = 0.22
  thrust_duration: float = 0.18
  vertical_acceleration: float = 18.0
  settle_duration: float = 0.8
  crouch_duration: float = 0.7

  def __post_init__(self) -> None:
    if self.direction not in (-1, 1):
      raise ValueError("direction must be -1 (right) or +1 (left)")
    for name in (
      "lateral_speed",
      "thrust_duration",
      "vertical_acceleration",
      "settle_duration",
      "crouch_duration",
    ):
      if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
        raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(self.crouch_height) or not 0.22 <= self.crouch_height <= 0.28:
      raise ValueError("crouch_height must be within [0.22, 0.28] m")
    if self.lateral_speed > 2.4:
      raise ValueError("lateral_speed must be at most 2.4 m/s")

  @classmethod
  def for_size(
    cls,
    size: str = "large",
    *,
    direction: int = 1,
    lateral_speed: float | None = None,
  ) -> SideJumpConfig:
    """Matched takeoff trajectories, rather than increasing sideways speed alone.

    Approximate total lateral travel on the nominal scene: small 13 cm,
    medium 33 cm, large 52 cm. These are presets, not distance tracking.
    """
    profiles = {
      "small": (0.45, 0.24, 0.16, 10.0),
      "medium": (1.2, 0.22, 0.18, 18.0),
      "large": (2.4, 0.22, 0.18, 18.0),
    }
    if size not in profiles:
      raise ValueError(f"size must be one of {JUMP_SIZES}")
    speed, height, duration, acceleration = profiles[size]
    return cls(
      direction=direction,
      lateral_speed=speed if lateral_speed is None else lateral_speed,
      crouch_height=height,
      thrust_duration=duration,
      vertical_acceleration=acceleration,
    )


def _skew(v: np.ndarray) -> np.ndarray:
  x, y, z = v
  return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


def _blend(t: float) -> float:
  s = float(np.clip(t, 0, 1))
  return s**3 * (10 - 15 * s + 6 * s**2)


class SideJumpSim:
  """Jump, absorb touchdown, then replant each foot and hold the standard pose."""

  def __init__(self, cfg: SideJumpConfig | None = None) -> None:
    self.cfg = cfg or SideJumpConfig()
    self.model = m = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    self.data = mujoco.MjData(m)
    self.base_id = m.body(BASE_BODY).id
    self.site_ids = np.array([m.site(name).id for name in FOOT_SITES])
    self.foot_geom_ids = np.array([m.geom(name).id for name in FOOT_GEOMS])
    self.ground_id = m.geom("ground").id
    joints = np.array([m.joint(name).id for name in JOINT_NAMES])
    self.qpos_ids = m.jnt_qposadr[joints]
    self.dof_ids = m.jnt_dofadr[joints]
    self.joint_limits = m.jnt_range[joints].copy()
    self.act_ids = np.array([m.actuator(f"{name}_act").id for name in JOINT_NAMES])
    self.hips = m.body_pos[[m.body(f"{leg}_hip").id for leg in LEGS]].copy()
    self.mass = float(m.body_mass.sum())
    # Change only this instance; leave the generated asset and other tasks intact.
    m.actuator_gainprm[:] = 0
    m.actuator_gainprm[:, 0] = 1
    m.actuator_biasprm[:] = 0
    m.actuator_ctrlrange[:] = [-RS02_PEAK_TORQUE_NM, RS02_PEAK_TORQUE_NM]
    self.reset()

  @property
  def feet(self) -> np.ndarray:
    return self.data.site_xpos[self.site_ids].copy()

  @property
  def base_height(self) -> float:
    return float(self.data.xpos[self.base_id, 2])

  def reset(self, model=None, data=None) -> None:
    m, d = self.model, self.data
    mujoco.mj_resetData(m, d)
    d.qpos[self.qpos_ids] = [DEFAULT_JOINT_POS[name] for name in JOINT_NAMES]
    d.qpos[2] = NOMINAL_BASE_HEIGHT
    mujoco.mj_forward(m, d)
    self.origin = d.xpos[self.base_id].copy()
    self.anchors = self.feet
    self.nominal_feet = self.feet - self.origin
    self.foot_targets = self.feet
    self.target_q = d.qpos[self.qpos_ids].copy()
    self.target_base = self.origin.copy()
    self.phase = "settle"
    self.phase_start = 0.0
    self.steps = 0
    self.failed = False
    self.success = False
    self.contacts = np.ones(4, dtype=bool)
    self.previous_contacts = self.contacts.copy()
    self.airtime = 0.0
    self.flight_start: float | None = None
    self.touchdown_time: float | None = None
    self.takeoff_y: float | None = None
    self.touchdown_y: float | None = None
    self.max_height = self.base_height
    self.max_tilt = 0.0
    self.max_torque = 0.0
    self.nonfoot_force = 0.0
    self.stable_time = 0.0
    self.recovery_leg = -1
    self.recovery_slot = -1
    self.recovery_lifts = np.zeros(4, dtype=int)
    self.recovery_clearance = np.zeros(4)
    self.nominal_q = np.array([DEFAULT_JOINT_POS[name] for name in JOINT_NAMES])

  def _contacts(self) -> np.ndarray:
    contacts = np.zeros(4, dtype=bool)
    force = np.zeros(6)
    for i in range(self.data.ncon):
      c = self.data.contact[i]
      if self.ground_id not in (c.geom1, c.geom2):
        continue
      mujoco.mj_contactForce(self.model, self.data, i, force)
      if force[0] <= 0.1:
        continue
      other = c.geom2 if c.geom1 == self.ground_id else c.geom1
      matches = self.foot_geom_ids == other
      contacts |= matches
      if not matches.any():
        self.nonfoot_force = max(self.nonfoot_force, float(force[0]))
    return contacts

  def _transition(self, phase: str) -> None:
    self.phase = phase
    self.phase_start = self.data.time

  def _ik(self, relative_feet: np.ndarray) -> np.ndarray:
    q = np.empty((4, 3))
    for leg in range(4):
      x, y, z = relative_feet[leg] - self.hips[leg]
      q[leg] = inverse_3d(
        x,
        y,
        z,
        -1 if leg < 2 else 1,
        FOOT_OFFSET_Y * (1 if leg % 2 == 0 else -1),
      )
    return np.clip(q.ravel(), self.joint_limits[:, 0], self.joint_limits[:, 1])

  def _reference(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d, cfg = self.data, self.cfg
    t = d.time - self.phase_start
    pos = self.origin.copy()
    vel, acc = np.zeros(3), np.zeros(3)
    if self.phase == "settle":
      if t >= cfg.settle_duration:
        self._transition("crouch")
        return self._reference()
    elif self.phase == "crouch":
      pos[2] += (cfg.crouch_height - self.origin[2]) * _blend(t / cfg.crouch_duration)
      if t >= cfg.crouch_duration:
        self._transition("thrust")
        return self._reference()
    elif self.phase == "thrust":
      acc[1] = cfg.direction * cfg.lateral_speed / cfg.thrust_duration
      acc[2] = cfg.vertical_acceleration
      vel = acc * min(t, cfg.thrust_duration)
      pos[2] = cfg.crouch_height
      pos += 0.5 * acc * min(t, cfg.thrust_duration) ** 2
      if not self.contacts.any() and t > 0.04:
        self.flight_start = d.time
        self.takeoff_y = float(d.qpos[1])
        self._transition("flight")
        return self._reference()
      elif t > cfg.thrust_duration + 0.2:
        self.failed = True
    elif self.phase == "flight":
      # Move the feet ahead of lateral motion to catch and brake the body.
      relative = self.nominal_feet.copy()
      relative[:, 2] += 0.04 * (1 - _blend(t / 0.15))
      relative[:, 1] += 0.12 * d.qvel[1]
      rot = d.xmat[self.base_id].reshape(3, 3)
      self.target_q = self._ik(relative)
      self.foot_targets = d.xpos[self.base_id] + relative @ rot.T
      if self.contacts.any() and t > 0.025:
        self.touchdown_time = d.time
        self.touchdown_y = float(d.qpos[1])
        self.anchors = self.feet
        self.anchors[:, 2] = self.origin[2] + self.nominal_feet[:, 2]
        self.landing_origin = d.xpos[self.base_id].copy()
        self.landing_target = self.landing_origin.copy()
        self.landing_target[1] += 0.10 * d.qvel[1]
        self.landing_target[2] = self.origin[2]
        self._transition("land")
        return self._reference()
      elif t > 1.0:
        self.failed = True
    elif self.phase == "land":
      pos = self.landing_target.copy()
      pos[2] = self.landing_origin[2] + (
        self.origin[2] - self.landing_origin[2]
      ) * _blend(t / 0.5)
      # Feet that touch down later acquire their own contact anchor.
      for leg in range(4):
        if not self.previous_contacts[leg] and self.contacts[leg]:
          self.anchors[leg, :2] = self.feet[leg, :2]
      if t > 0.8 and self.contacts.all() and np.linalg.norm(d.qvel[:6]) < 0.1:
        self.standard_base = pos.copy()
        self.standard_feet = self.standard_base + self.nominal_feet
        self._transition("recover")
        return self._reference()
    elif self.phase == "recover":
      # Shift over the three supports before lifting one foot to its standard
      # stance location. The floating base continues to move through motor forces.
      period, shift_time, swing_time = 1.2, 0.6, 0.5
      slot = min(int(t / period), 4)
      local = t - slot * period
      order = (1, 2, 0, 3)
      if slot < 4:
        leg = order[slot]
        if slot != self.recovery_slot:
          self.recovery_slot = slot
          self.recovery_shift_start = self.target_base.copy()
          self.recovery_foot_start = self.feet[leg].copy()
          supports = [i for i in range(4) if i != leg]
          self.recovery_support_center = np.mean(self.anchors[supports], axis=0)
          self.recovery_support_center[2] = self.origin[2]
        pos = self.recovery_shift_start + _blend(local / shift_time) * (
          self.recovery_support_center - self.recovery_shift_start
        )
        if local >= shift_time:
          self.recovery_leg = leg if local < shift_time + swing_time else -1
          phase = float(np.clip((local - shift_time) / swing_time, 0, 1))
          self.anchors[leg] = self.recovery_foot_start + _blend(phase) * (
            self.standard_feet[leg] - self.recovery_foot_start
          )
          self.anchors[leg, 2] += 0.04 * 16 * phase**2 * (1 - phase) ** 2
          if not self.contacts[leg]:
            self.recovery_lifts[leg] = 1
      else:
        self.recovery_leg = -1
        pos = self.recovery_support_center + _blend(local / 0.8) * (
          self.standard_base - self.recovery_support_center
        )
        if local >= 0.8:
          self._transition("hold")
          return self._reference()
    elif self.phase == "hold":
      pos = self.standard_base.copy()
    if self.phase != "flight":
      self.target_q = self._ik(self.anchors - pos)
      self.foot_targets = self.anchors.copy()
    self.target_base = pos
    return pos, vel, acc

  def step(self, model=None, data=None) -> None:
    if self.failed:
      return
    m, d = self.model, self.data
    self.previous_contacts = self.contacts.copy()
    self.contacts = self._contacts()
    pos, vel, acc = self._reference()
    q = d.qpos[self.qpos_ids]
    qd = d.qvel[self.dof_ids]
    # IK sets joint targets; PD provides compliance during landing.
    torque = 65 * (self.target_q - q) - 2.0 * qd
    if self.phase != "flight":
      rot = d.xmat[self.base_id].reshape(3, 3)
      angular = rot @ d.qvel[3:6]
      orient_error = 0.5 * np.array(
        [rot[2, 1] - rot[1, 2], rot[0, 2] - rot[2, 0], rot[1, 0] - rot[0, 1]]
      )
      force = self.mass * (
        acc + [0, 0, 9.81] + 70 * (pos - d.xpos[self.base_id]) + 12 * (vel - d.qvel[:3])
      )
      moment = -65 * orient_error - 8 * angular
      support_mask = self.contacts.copy()
      if self.recovery_leg >= 0:
        support_mask[self.recovery_leg] = False
      supports = np.flatnonzero(support_mask)
      if len(supports):
        r = self.feet - d.subtree_com[self.base_id]
        # Distribute desired body force/moment over the measured support feet.
        # This small least-squares allocation is not an MPC or a contact solver.
        allocation = np.vstack(
          (
            np.tile(np.eye(3), (1, len(supports))),
            np.hstack([_skew(r[i]) for i in supports]),
          )
        )
        forces = np.linalg.lstsq(allocation, np.r_[force, moment], rcond=None)[
          0
        ].reshape(-1, 3)
        for leg, f in zip(supports, forces, strict=True):
          f[2] = np.clip(f[2], 0, 120)
          f[:2] = np.clip(f[:2], -0.7 * f[2], 0.7 * f[2])
          jac = np.zeros((3, m.nv))
          mujoco.mj_jacSite(m, d, jac, None, self.site_ids[leg])
          sl = slice(leg * 3, leg * 3 + 3)
          # Opposite foot force produces the requested ground reaction.
          torque[sl] -= jac[:, self.dof_ids[sl]].T @ f
    limit = RS02_PEAK_TORQUE_NM * np.clip(
      1 - np.abs(qd) / RS02_NO_LOAD_SPEED_RAD_S, 0, 1
    )
    d.ctrl[self.act_ids] = np.clip(torque, -limit, limit)
    mujoco.mj_step(m, d)
    self.steps += 1
    tilt = float(np.arccos(np.clip(d.xmat[self.base_id].reshape(3, 3)[2, 2], -1, 1)))
    self.max_tilt = max(self.max_tilt, tilt)
    self.max_height = max(self.max_height, self.base_height)
    self.max_torque = max(self.max_torque, float(np.max(np.abs(d.actuator_force))))
    if self.phase == "flight" and not self.contacts.any():
      self.airtime += m.opt.timestep
    self.failed |= (
      not np.isfinite(d.qpos).all()
      or not np.isfinite(d.qvel).all()
      or self.base_height < 0.17
      or tilt > 0.8
      or self.nonfoot_force > 0.5
    )
    if self.phase == "recover":
      self.recovery_clearance = np.maximum(
        self.recovery_clearance,
        self.feet[:, 2] - (self.origin[2] + self.nominal_feet[:, 2]),
      )
    # A stopped, splayed landing is not the final pose. Require the measured
    # joint angles and full body orientation (including yaw) to be restored.
    joint_error = float(np.max(np.abs(d.qpos[self.qpos_ids] - self.nominal_q)))
    rot = d.xmat[self.base_id].reshape(3, 3)
    orientation_error = float(np.arccos(np.clip((np.trace(rot) - 1) / 2, -1, 1)))
    if (
      self.phase == "hold"
      and self.contacts.all()
      and np.linalg.norm(d.qvel[:6]) < 0.08
      and orientation_error < np.deg2rad(1.0)
      and joint_error < np.deg2rad(1.5)
    ):
      self.stable_time += m.opt.timestep
    else:
      self.stable_time = 0.0
    self.success = (
      self.stable_time >= 0.5
      and self.airtime >= 0.05
      and self.cfg.direction * (d.qpos[1] - self.origin[1]) > 0.02
      and not self.failed
    )

  def run(self, duration: float = 10.0) -> dict:
    if not np.isfinite(duration) or duration <= 0:
      raise ValueError("duration must be finite and positive")
    for _ in range(round(duration / self.model.opt.timestep)):
      self.step()
      if self.failed:
        break
    return self.metrics()

  def metrics(self) -> dict:
    airborne_distance = None
    if self.takeoff_y is not None and self.touchdown_y is not None:
      airborne_distance = self.touchdown_y - self.takeoff_y
    rot = self.data.xmat[self.base_id].reshape(3, 3)
    joint_error = np.max(np.abs(self.data.qpos[self.qpos_ids] - self.nominal_q))
    return {
      "time_s": float(self.data.time),
      "phase": self.phase,
      "lateral_distance_m": float(self.data.qpos[1] - self.origin[1]),
      "forward_drift_m": float(self.data.qpos[0] - self.origin[0]),
      "airtime_s": self.airtime,
      "airborne_lateral_distance_m": airborne_distance,
      "max_height_m": self.max_height,
      "height_m": self.base_height,
      "max_tilt_deg": float(np.rad2deg(self.max_tilt)),
      "final_tilt_deg": float(np.rad2deg(np.arccos(np.clip(rot[2, 2], -1, 1)))),
      "final_yaw_deg": float(np.rad2deg(np.arctan2(rot[1, 0], rot[0, 0]))),
      "standard_joint_error_deg": float(np.rad2deg(joint_error)),
      "recovery_foot_clearance_m": self.recovery_clearance.tolist(),
      "max_torque_nm": self.max_torque,
      "nonfoot_ground_force_n": self.nonfoot_force,
      "final_speed_m_s": float(np.linalg.norm(self.data.qvel[:3])),
      "success": bool(self.success),
      "failed": bool(self.failed),
    }
