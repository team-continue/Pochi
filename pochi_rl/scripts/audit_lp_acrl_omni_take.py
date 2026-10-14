"""Grade actual fresh50Hz traces of first-take omni videos, without retakes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from eval_lp_acrl_omni_phase import summarize_omni_segment
from record_lp_acrl_omni_phase import SCENARIOS


def audit(report):
  records = report["records"]
  scenario = report["scenario"]
  if scenario not in SCENARIOS:
    raise ValueError("Unknown native scenario")
  dt = report["control_dt_s"]
  checks = {
    "complete20s": bool(report["complete"] and len(records) == 1000),
    "control_dt20ms": bool(np.isclose(dt, 0.02)),
    "native_infinite_plane": report["terrain"] == "native infinite plane",
    "task": report["task"] == "Pochi-LP-ACRL-Blind-v18" and report["task_id"] == 0,
    "registered_seed": (scenario, report["seed"])
    in (("translations", 9123), ("rotation_combined", 9223)),
    "obs50_actions12_no_heightscan": report["observation_dim"] == 50
    and report["action_dim"] == 12
    and report["height_scan"] is False,
    "no_fall_or_invalid": not report["fall"] and not report["invalid"],
    "first_episode_only": all(
      not r["terminated"] and r["valid"] and not r["physics_faults"] for r in records
    ),
    "ordered_control_steps": [r["step"] for r in records]
    == list(range(1, len(records) + 1)),
  }

  def array(name):
    return np.asarray([r[name] for r in records])

  finite = all(
    np.isfinite(array(name)).all()
    for name in (
      "foot_position_w_m",
      "foot_velocity_w_m_s",
      "foot_vertical_load_abs_N",
      "body_position_w_m",
      "body_quaternion_wxyz",
      "body_linear_velocity_b_m_s",
      "body_angular_velocity_b_rad_s",
      "joint_position_rad",
      "joint_velocity_rad_s",
    )
  )
  checks["finite_raw_measurements"] = bool(finite)
  if not records or not finite:
    return {"checks": checks, "segments": [], "passed": False}
  t = array("time_after_s")
  checks["exact50hz_trace"] = bool(
    np.allclose(t, np.arange(1, len(records) + 1) * dt, atol=1e-9)
  )
  checks["time_before_matches_control_step"] = bool(
    np.allclose(array("time_before_s"), t - dt, atol=1e-9)
  )
  checks["registered_commands"] = all(
    r["command_vx_vy_wz"] == list(command)
    for r in records
    for _, start, end, command in SCENARIOS[scenario]
    if start <= r["time_before_s"] < end
  )
  contacts = array("contact").astype(bool)
  slip_per_foot = np.linalg.norm(array("foot_velocity_w_m_s")[:, :, :2], axis=-1)
  slip = (slip_per_foot * contacts).sum(-1) / contacts.sum(-1).clip(min=1)
  q = array("body_quaternion_wxyz")
  tilt = np.degrees(np.arccos(np.clip(1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2), -1, 1)))
  samples = {
    "time_s": t,
    "valid": np.asarray([[r["valid"]] for r in records]),
    "velocity": np.column_stack(
      (
        array("body_linear_velocity_b_m_s")[:, :2],
        array("body_angular_velocity_b_rad_s")[:, 2],
      )
    )[:, None],
    "position": array("body_position_w_m")[:, None],
    "contact": contacts[:, None],
    "foot_z": array("foot_position_w_m")[:, None, :, 2],
    "foot_vertical_force": array("foot_vertical_load_abs_N")[:, None],
    "nonfoot_contact": array("nonfoot_contact")[:, None],
    "slip": slip[:, None],
    "tilt_deg": tilt[:, None],
    "joint_error": (
      array("joint_position_rad") - np.asarray(report["default_joint_position_rad"])
    )[:, None],
    "joint_velocity": array("joint_velocity_rad_s")[:, None],
  }
  segments = [
    summarize_omni_segment(
      samples, 0, start, end, command, dt, 1.0 if any(command) else 0.5
    )
    for _, start, end, command in SCENARIOS[scenario]
  ]
  return {
    "checks": checks,
    "segments": segments,
    "passed": all(checks.values()) and all(s["passed"] for s in segments),
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("trace", type=Path)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    parser.error("Preserve earlier audit results")
  result = audit(json.loads(args.trace.read_text()))
  with args.output.open("x") as f:
    json.dump(result, f, indent=2)
  print(json.dumps({"passed": result["passed"], "output": str(args.output)}))


if __name__ == "__main__":
  main()
