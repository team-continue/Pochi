from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

from pochi_rl.mjlab.lp_acrl_phase_gait import foot_targets


def test_native_audit_checks_direction_switches_and_terminal_faults(monkeypatch):
  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
  from audit_lp_acrl_omni_take import audit
  from record_lp_acrl_omni_phase import SCENARIOS

  scenario = "rotation_combined"
  rows = []
  t = np.arange(1, 1001) * 0.02
  lift, contacts = foot_targets(torch.tensor(t * 1.5, dtype=torch.float32) % 1)
  for i in range(1000):
    before = i * 0.02
    _, _, _, command = next(s for s in SCENARIOS[scenario] if s[1] <= before < s[2])
    moving = any(command)
    z = lift[i].numpy() if moving else np.zeros(4)
    row = {
      "step": i + 1,
      "time_before_s": before,
      "time_after_s": t[i],
      "command_vx_vy_wz": list(command),
      "valid": True,
      "terminated": False,
      "physics_faults": [],
      "contact": contacts[i].tolist() if moving else [True] * 4,
      "foot_position_w_m": [[0, 0, float(v)] for v in z],
      "foot_velocity_w_m_s": [[0, 0, 0]] * 4,
      "foot_vertical_load_abs_N": [27] * 4,
      "body_position_w_m": [0, 0, 0.317],
      "body_quaternion_wxyz": [1, 0, 0, 0],
      "body_linear_velocity_b_m_s": [command[0], command[1], 0],
      "body_angular_velocity_b_rad_s": [0, 0, command[2]],
      "joint_position_rad": [0] * 12,
      "joint_velocity_rad_s": [0] * 12,
      "nonfoot_contact": False,
    }
    rows.append(row)
  report = {
    "records": rows,
    "scenario": scenario,
    "control_dt_s": 0.02,
    "complete": True,
    "terrain": "native infinite plane",
    "task": "Pochi-LP-ACRL-Blind-v18",
    "task_id": 0,
    "seed": 9223,
    "observation_dim": 50,
    "action_dim": 12,
    "height_scan": False,
    "fall": False,
    "invalid": False,
    "default_joint_position_rad": [0] * 12,
  }
  assert audit(report)["passed"]
  stopped = deepcopy(report)
  for r in stopped["records"]:
    r["body_angular_velocity_b_rad_s"] = [0, 0, 0]
  assert not audit(stopped)["passed"]  # Correct yaw commands require actual turning.
  corrupt = deepcopy(report)
  corrupt["records"][-1]["valid"] = False
  assert not audit(corrupt)["passed"]  # Terminal corruption cannot be dropped.
