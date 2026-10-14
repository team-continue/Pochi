"""Grade immutable native traces with frozen stop/walk gates and real lift."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from eval_lp_acrl_behavior import STAND_LIMITS, WALK_LIMITS


def qualified_cycles(contact, feet, times, dt, *, start, end):
  counts = [0] * 4
  for foot in range(4):
    reference = None
    first_air = None
    for index, time in enumerate(times):
      if contact[index, foot]:
        if first_air is not None:
          duration = (index - first_air) * dt
          lift = feet[first_air:index, foot, 2].max() - feet[reference, foot, 2]
          if (
            times[reference] > start + 1.0 + 1e-9 and time <= end + 1e-9
            and duration >= .06 - 1e-9 and lift >= .025 - 1e-9
          ):
            counts[foot] += 1
        reference, first_air = index, None
      elif reference is not None and first_air is None:
        first_air = index
  return counts


def audit(report):
  records = report["records"]
  dt = report["control_dt_s"]
  if dt != .02 or report["task_id"] != 360:
    raise ValueError("Native task/time step differs")
  if report["height_scan"] or report["observation_dim"] != 48:
    raise ValueError("Blind observation contract differs")
  times = np.asarray([r["time_after_s"] for r in records])
  contact = np.asarray([r["contact"] for r in records], dtype=bool)
  feet = np.asarray([r["foot_position_w_m"] for r in records], dtype=float)
  phases = []
  for phase, start, end in (
    ("initial_stop", 0., 5.), ("forward", 5., 15.), ("final_stop", 15., 20.),
  ):
    chosen = [r for r in records if r["phase"] == phase and r["valid"]
              and r["time_after_s"] > start + 1.0 + 1e-9]
    expected = int(round((end - start - 1.) / dt))
    if len(chosen) != expected:
      phases.append({"phase": phase, "passed": False,
                     "reason": "Missing valid scored samples"})
      continue

    def array(key, rows=chosen):
      return np.asarray([r[key] for r in rows], dtype=float)

    position = array("body_position_w_m")
    velocity = array("body_linear_velocity_b_m_s")
    angular = array("body_angular_velocity_b_rad_s")
    quat = array("body_quaternion_wxyz")
    c = array("contact").astype(bool)
    duty = c.mean(0)
    slip = np.linalg.norm(array("foot_velocity_w_m_s")[..., :2], axis=-1)
    tilt = np.degrees(np.arccos(np.clip(
      1 - 2 * (quat[:, 1] ** 2 + quat[:, 2] ** 2), -1, 1,
    )))
    metrics = {
      "joint_default_rmse_rad": float(np.sqrt(np.mean(
        array("joint_default_rmse_rad") ** 2,
      ))),
      "joint_velocity_rms_rad_s": float(np.sqrt(np.mean(
        array("joint_velocity_rad_s") ** 2,
      ))),
      "all_four_contact_fraction": float(c.all(-1).mean()),
      "horizontal_speed_mean_m_s": float(np.linalg.norm(
        velocity[:, :2], axis=-1,
      ).mean()),
      "yaw_rate_absolute_mean_rad_s": float(np.abs(angular[:, 2]).mean()),
      "horizontal_drift_max_m": float(np.linalg.norm(
        position[:, :2] - position[0, :2], axis=-1,
      ).max()),
      "contact_slip_mean_m_s": float(((slip * c).sum(-1)
                                     / c.sum(-1).clip(min=1)).mean()),
      "tilt_p95_deg": float(np.quantile(tilt, .95)),
      "nonfoot_contact_fraction": float(array("nonfoot_contact").mean()),
    }
    checks = {}
    if phase != "forward":
      for key, limit in STAND_LIMITS.items():
        lower = key.endswith("_min")
        metric = key[:-4] if lower else key
        checks[key] = bool(metrics[metric] >= limit if lower
                           else metrics[metric] <= limit)
    else:
      cycles = qualified_cycles(contact, feet, times, dt, start=start, end=end)
      metrics.update(
        forward_error_mean_m_s=float(abs(velocity[:, 0].mean() - .25)),
        lateral_speed_absolute_mean_m_s=float(np.abs(velocity[:, 1]).mean()),
        foot_contact_duty_min=float(duty.min()),
        foot_contact_duty_max=float(duty.max()),
        foot_contact_duty_spread_max=float(np.ptp(duty)),
        footfalls_per_leg_min=min(cycles), qualified_cycles_per_foot=cycles,
        all_feet_airborne_fraction=float((c.sum(-1) == 0).mean()),
      )
      for key, limit in WALK_LIMITS.items():
        checks[key] = bool(metrics[key] >= limit if key.endswith("_min")
                           else metrics[key] <= limit)
      checks["no_observed_whole_body_flight"] = (
        metrics["all_feet_airborne_fraction"] == 0.
      )
    phases.append({"phase": phase, "scored_samples": len(chosen),
                   "metrics": metrics, "checks": checks,
                   "passed": all(checks.values())})
  integrity = bool(
    report["complete"] and not report["fall"] and not report["invalid"]
    and not report["failure_reasons"] and len(records) == 1000
    and all(r["valid"] and not r["terminated"] and not r["physics_faults"]
            for r in records)
    and all(np.isfinite(np.asarray([r[key] for r in records], dtype=float)).all()
            for key in ("foot_position_w_m", "foot_velocity_w_m_s",
                        "foot_force_w_N", "joint_position_rad",
                        "joint_velocity_rad_s", "body_position_w_m",
                        "body_quaternion_wxyz", "body_linear_velocity_b_m_s",
                        "body_angular_velocity_b_rad_s"))
    and np.allclose(times, np.arange(1, 1001) * dt, atol=1e-10)
  )
  return {"passed": integrity and all(p["passed"] for p in phases),
          "trace_integrity": integrity, "phases": phases,
          "method": "Frozen stop/walk gates; each foot >=2 complete air bouts "
          "of >=.06s with >=.025m lift over preceding contact, wholly in scored "
          "forward window. No observed whole-body flight. No simulation or retake.",
          "visual_review_required": True}


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--input", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  report = json.loads(args.input.read_text())
  result = audit(report)
  result["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
  result["checkpoint_sha256"] = report["checkpoint_sha256"]
  result["seed"] = report["seed"]
  with args.output.open("x") as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
  print(json.dumps(result, indent=2))


if __name__ == "__main__":
  main()
