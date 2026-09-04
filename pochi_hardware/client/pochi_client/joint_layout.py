from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class JointSpec:
    name: str
    label: str
    leg: str
    joint: str
    motor_id: int


# Physical CAN mapping, listed foot-to-body (and therefore ID-ascending) within
# each leg so telemetry and GUI presentation follow the physical wiring order.
JOINTS: tuple[JointSpec, ...] = (
    JointSpec("front_left_calf", "FL Calf", "front_left", "calf", 0),
    JointSpec("front_left_thigh", "FL Thigh", "front_left", "thigh", 1),
    JointSpec("front_left_hip", "FL Hip", "front_left", "hip", 2),
    JointSpec("rear_left_calf", "RL Calf", "rear_left", "calf", 3),
    JointSpec("rear_left_thigh", "RL Thigh", "rear_left", "thigh", 4),
    JointSpec("rear_left_hip", "RL Hip", "rear_left", "hip", 5),
    JointSpec("rear_right_calf", "RR Calf", "rear_right", "calf", 6),
    JointSpec("rear_right_thigh", "RR Thigh", "rear_right", "thigh", 7),
    JointSpec("rear_right_hip", "RR Hip", "rear_right", "hip", 8),
    JointSpec("front_right_calf", "FR Calf", "front_right", "calf", 9),
    JointSpec("front_right_thigh", "FR Thigh", "front_right", "thigh", 10),
    JointSpec("front_right_hip", "FR Hip", "front_right", "hip", 11),
)

JOINT_BY_NAME = {joint.name: joint for joint in JOINTS}
JOINT_BY_ID = {joint.motor_id: joint for joint in JOINTS}
JOINTS_BY_LEG = {
    leg: tuple(joint for joint in JOINTS if joint.leg == leg)
    for leg in ("front_left", "front_right", "rear_left", "rear_right")
}
