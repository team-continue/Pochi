"""Run the stand-up policy against the plain MuJoCo model.

This is the standalone path: ``assets/pochi/scene_flat.xml`` with the
``<position>`` actuators that ship in the MJCF, no mjlab and no torch.  The
policy is stepped at the task spec's control rate and everything in between is
plain physics, so both the viser demo (``scripts/play_standup.py``) and the
regression test drive the manoeuvre through this one class.
"""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from pochi_rl.control.standup import (
  FOLDED_JOINT_POS,
  MIN_CROUCH_HEIGHT,
  StandUpConfig,
  StandUpController,
)
from pochi_rl.robot import BASE_BODY, JOINT_NAMES
from pochi_rl.task_spec import POCHI_TASK_SPEC

SCENE_XML = Path(__file__).resolve().parents[3] / "assets" / "pochi" / "scene_flat.xml"

# Spawn height for the folded pose: a hair above the crouch the folded legs
# hold, so the robot drops onto its feet rather than being born interpenetrating
# the floor.
FOLDED_DROP_HEIGHT = MIN_CROUCH_HEIGHT + 0.005

# How long to hold the motors off while the robot collapses onto its belly.
# It is down and still inside 2.5 s; the rest is margin.
LIMP_SETTLE_S = 3.0


class StandUpSim:
  """A single MuJoCo instance of Pochi standing up from the folded crouch."""

  def __init__(
    self,
    cfg: StandUpConfig | None = None,
    limp_settle_s: float = LIMP_SETTLE_S,
    xml_path: Path | str = SCENE_XML,
  ) -> None:
    self.model = mujoco.MjModel.from_xml_path(str(xml_path))
    self.data = mujoco.MjData(self.model)
    self.controller = StandUpController(cfg)
    self.limp_steps = int(round(limp_settle_s / self.model.opt.timestep))

    # Gains to switch off and back on again to go limp.  A <position> actuator
    # makes torque as gainprm[0] * ctrl + biasprm[1] * q + biasprm[2] * qd, so
    # zeroing those three zeroes the motor while leaving the joints' own
    # damping and armature -- which the real hardware has too -- in place.
    self._gainprm = self.model.actuator_gainprm.copy()
    self._biasprm = self.model.actuator_biasprm.copy()

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
    self._act_ids = np.array(
      [
        mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"{name}_act")
        for name in JOINT_NAMES
      ]
    )
    self._base_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
    self._folded = np.array([FOLDED_JOINT_POS[name] for name in JOINT_NAMES])
    self._step_in_period = 0

    self.reset()

  # -- state -------------------------------------------------------------------

  @property
  def joint_pos(self) -> np.ndarray:
    return self.data.qpos[self._qpos_adr].copy()

  @property
  def base_height(self) -> float:
    return float(self.data.xpos[self._base_id][2])

  @property
  def phase(self) -> str:
    return self.controller.phase

  # -- simulation --------------------------------------------------------------

  def reset(self, model=None, data=None) -> None:
    """Collapse the robot onto its belly, then arm the manoeuvre.

    Rather than posing the start by hand, the robot is put in the folded crouch
    and the motors are switched off.  It sinks: the hip rolls splay out to their
    stops and the knees stay curled, so it ends up flat on its belly in a frog
    sprawl.  That settled pose -- whatever physics makes of it -- is the pose
    the manoeuvre starts from.

    The signature takes the ``(model, data)`` that mjviser's reset callback
    passes; both are ignored in favour of this instance's own.
    """
    mujoco.mj_resetData(self.model, self.data)
    self.data.qpos[self._qpos_adr] = self._folded
    self.data.qpos[2] = FOLDED_DROP_HEIGHT
    self.data.ctrl[self._act_ids] = self._folded
    mujoco.mj_forward(self.model, self.data)

    self._set_motors(on=False)
    for _ in range(self.limp_steps):
      mujoco.mj_step(self.model, self.data)
    self._set_motors(on=True)

    # Hold the pose it landed in, so switching the motors back on is not itself
    # a step input.
    self.data.ctrl[self._act_ids] = self.joint_pos
    self.controller.reset(self.joint_pos)
    self._step_in_period = 0

  def _set_motors(self, on: bool) -> None:
    self.model.actuator_gainprm[:] = self._gainprm if on else 0.0
    self.model.actuator_biasprm[:] = self._biasprm if on else 0.0

  def step(self, model=None, data=None) -> None:
    """Advance one physics step, refreshing the policy every ``decimation``.

    Same story as :meth:`reset` on the ignored arguments: this is mjviser's
    ``step_fn``.
    """
    if self._step_in_period == 0:
      self.data.ctrl[self._act_ids] = self.controller.act(
        self.joint_pos, self.control_dt
      )
    self._step_in_period = (self._step_in_period + 1) % self.decimation
    mujoco.mj_step(self.model, self.data)

  def run(self, duration_s: float) -> None:
    for _ in range(int(round(duration_s / self.model.opt.timestep))):
      self.step()
