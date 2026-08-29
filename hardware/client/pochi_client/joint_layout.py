from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True, slots=True)
class JointSpec:
    name: str
    label: str
    leg: str
    joint: str
    motor_id: int
    direction: float = 1.0
    zero_offset_rad: float = 0.0


# Provisional physical mapping. Motor IDs are always shown in the GUI so this
# table can be corrected without changing the wire protocol or client API.
JOINTS: tuple[JointSpec, ...] = (
    JointSpec("front_left_hip", "FL Hip", "front_left", "hip", 1),
    JointSpec("front_left_thigh", "FL Thigh", "front_left", "thigh", 2),
    JointSpec("front_left_calf", "FL Calf", "front_left", "calf", 3),
    JointSpec("front_right_hip", "FR Hip", "front_right", "hip", 4),
    JointSpec("front_right_thigh", "FR Thigh", "front_right", "thigh", 5),
    JointSpec("front_right_calf", "FR Calf", "front_right", "calf", 6),
    JointSpec("rear_left_hip", "RL Hip", "rear_left", "hip", 7),
    JointSpec("rear_left_thigh", "RL Thigh", "rear_left", "thigh", 8),
    JointSpec("rear_left_calf", "RL Calf", "rear_left", "calf", 9),
    JointSpec("rear_right_hip", "RR Hip", "rear_right", "hip", 10),
    JointSpec("rear_right_thigh", "RR Thigh", "rear_right", "thigh", 11),
    JointSpec("rear_right_calf", "RR Calf", "rear_right", "calf", 12),
)

JOINT_BY_NAME = {joint.name: joint for joint in JOINTS}
JOINT_BY_ID = {joint.motor_id: joint for joint in JOINTS}
JOINTS_BY_LEG = {
    leg: tuple(joint for joint in JOINTS if joint.leg == leg)
    for leg in ("front_left", "front_right", "rear_left", "rear_right")
}
