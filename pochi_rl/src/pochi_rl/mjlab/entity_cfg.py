"""Pochi robot entity configuration for mjlab."""

from __future__ import annotations

from pathlib import Path

import mujoco
from mjlab.actuator import DcMotorActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg

from pochi_rl.robot import (
  DEFAULT_JOINT_POS,
  NOMINAL_BASE_HEIGHT,
  RS02_NO_LOAD_SPEED_RAD_S,
  RS02_PEAK_TORQUE_NM,
  RS02_REFLECTED_INERTIA,
)

ASSETS_DIR = Path(__file__).resolve().parents[3] / "assets" / "pochi"
POCHI_XML = ASSETS_DIR / "pochi.xml"


def pochi_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(POCHI_XML))
  # pochi.xml ships <position> actuators so the standalone model stays drivable
  # in the MuJoCo viewer and on the ROS side.  Here mjlab owns actuation
  # instead: POCHI_ACTUATOR adds <motor> elements and runs the PD loop in torch
  # so the RS02 torque-speed curve can clamp the command.  Drop the XML ones or
  # the model ends up with 24 actuators.
  for actuator in list(spec.actuators):
    spec.delete(actuator)
  return spec


# The RS02 is a real motor, so the sim gets its real envelope rather than an
# unbounded one.  mjlab's DC-motor model clamps torque to
# ``saturation_effort * (1 - |qd| / velocity_limit)``, capped by
# ``effort_limit``, in torch before the command reaches MuJoCo.  Peak torque is
# therefore only available near zero speed and falls to zero at the no-load
# speed, which is what the hardware does.
#
# ``effort_limit`` is mjlab's *continuous* cap.  It is set to the peak here, so
# the curve alone binds; lowering it to RS02_RATED_TORQUE_NM would additionally
# enforce the 7 N.m continuous rating.  It must never exceed saturation_effort.
#
# Gains match the kp/kv that pochi.xml used, so switching to this actuator
# changes the torque envelope and nothing else.
POCHI_ACTUATOR = DcMotorActuatorCfg(
  target_names_expr=(".*_hip_roll", ".*_hip_pitch", ".*_knee"),
  stiffness=60.0,
  damping=1.5,
  effort_limit=RS02_PEAK_TORQUE_NM,
  saturation_effort=RS02_PEAK_TORQUE_NM,
  velocity_limit=RS02_NO_LOAD_SPEED_RAD_S,
  armature=RS02_REFLECTED_INERTIA,
)

POCHI_ROBOT_CFG = EntityCfg(
  init_state=EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, NOMINAL_BASE_HEIGHT + 0.01),
    joint_pos=DEFAULT_JOINT_POS,
    joint_vel={".*": 0.0},
  ),
  spec_fn=pochi_spec,
  articulation=EntityArticulationInfoCfg(
    actuators=(POCHI_ACTUATOR,),
    soft_joint_pos_limit_factor=0.9,
  ),
)
