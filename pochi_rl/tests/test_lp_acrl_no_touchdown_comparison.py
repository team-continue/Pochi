"""Cadence comparisons must not reward nonstarts or pair different worlds."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parents[1] / "scripts/compare_lp_acrl_no_touchdown.py"
_SPEC = importlib.util.spec_from_file_location("no_touchdown_comparison", _PATH)
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)


def report(task, cadence):
  rows = []
  for index in range(32):
    profile = _MODULE.PROFILES[index // 8]
    windows = [(0., 20., 2.)]
    commands = [0. if profile == "stand" else (.25 if profile == "forward" else -.25)]
    if profile == "stop_start":
      windows = [(i * 5., (i + 1) * 5., 1.) for i in range(4)]
      commands = [0., .25, 0., .25]
    segments = []
    for (start, end, settle), command in zip(windows, commands, strict=True):
      duration = end - start - settle
      segments.append({
        "start_s": start, "end_s": end, "excluded_transient_s": settle,
        "command_vx_vy_wz": [command, 0., 0.], "complete": True,
        "passed": True, "checks": {"forward_error_mean_m_s": True},
        "metrics": {
          "footfalls_per_leg": [int(cadence * duration)] * 4,
          "actual_mean_vx_vy_wz": [command, 0., 0.],
        },
      })
    rows.append({"environment_index": index, "profile": profile,
                 "fall": False, "invalid_state": False,
                 "passed": True, "segments": segments})
  return {
    "task": task, "checkpoint_sha256": _MODULE.SOURCE_SHA256,
    "protocol": "pochi-lp-acrl-behavior-plane-v2", "seed": 1456,
    "num_envs": 32, "height_scan_present": False,
    "control_dt_s": .02, "horizon_s": 20., "episodes": rows,
    "no_touchdown_state": {"completed_updates": 101},
    "summary": {p: {"episodes": 8, "passed": 8, "falls": 0, "invalid": 0}
                for p in _MODULE.PROFILES},
  }


def pair():
  return (report("Pochi-LP-ACRL-Blind-v13", 4.),
          report("Pochi-LP-ACRL-Blind-v15", 3.))


def test_paired_rate_uses_fixed_seconds_not_episode_length():
  old, new = pair()
  result = _MODULE.compare(old, new)
  assert result["candidate_behavior"]["eligible"]
  assert not result["cadence_used_for_selection"]
  for profile in ("forward", "backward", "stop_start"):
    primary = result["cadence"][profile]["speed_passing"]
    assert primary["paired_baseline"]["mean_cadence_per_foot_hz"] == [4.] * 4
    assert primary["paired_candidate"]["mean_cadence_per_foot_hz"] == [3.] * 4
    assert primary["paired_delta_hz"] == -1.
    assert primary["paired_relative_change"] == -.25


def test_nonstarter_is_excluded_from_effect_but_fails_behavior():
  old, new = pair()
  for row in new["episodes"][16:19]:
    row["passed"] = False
    segment = row["segments"][0]
    segment["passed"] = False
    segment["checks"]["forward_error_mean_m_s"] = False
    segment["metrics"]["footfalls_per_leg"] = [0] * 4
    segment["metrics"]["actual_mean_vx_vy_wz"] = [0., 0., 0.]
  new["summary"]["backward"]["passed"] = 5
  result = _MODULE.compare(old, new)
  assert not result["candidate_behavior"]["eligible"]
  effect = result["cadence"]["backward"]["speed_passing"]
  assert effect["paired_candidate"]["segments"] == 5
  assert effect["paired_candidate"]["mean_cadence_hz"] == 3.
  assert len(effect["jointly_passing_world_segments"]) == 5


def test_no_jointly_passing_worlds_reports_no_cadence_effect():
  old, new = pair()
  for row in new["episodes"][8:16]:
    row["passed"] = False
    row["segments"][0]["passed"] = False
    row["segments"][0]["checks"]["forward_error_mean_m_s"] = False
  new["summary"]["forward"]["passed"] = 0
  effect = _MODULE.compare(old, new)["cadence"]["forward"]["speed_passing"]
  assert effect["paired_delta_hz"] is None
  assert effect["paired_candidate"]["mean_cadence_hz"] is None


@pytest.mark.parametrize("change", ["seed", "hash", "slot", "size", "world",
                                   "duration", "command", "summary", "dt"])
def test_invalid_comparison_is_rejected(change):
  old, new = pair()
  if change == "seed":
    new["seed"] = 789
  elif change == "hash":
    old["checkpoint_sha256"] = "different"
  elif change == "slot":
    new["no_touchdown_state"]["completed_updates"] = 151
  elif change == "size":
    new["num_envs"] = 64
  elif change == "world":
    new["episodes"][8]["environment_index"] = 9
  elif change == "duration":
    new["episodes"][8]["segments"][0]["end_s"] = 19.
  elif change == "command":
    new["episodes"][8]["segments"][0]["command_vx_vy_wz"][0] = .4
  elif change == "summary":
    new["summary"]["forward"]["passed"] = 7
  else:
    new["control_dt_s"] = .01
  with pytest.raises(ValueError):
    _MODULE.compare(old, new)


def test_eligible_unchanged_cadence_is_not_rejected():
  old, _ = pair()
  new = deepcopy(old)
  new["task"] = "Pochi-LP-ACRL-Blind-v15"
  result = _MODULE.compare(old, new)
  assert result["candidate_behavior"]["eligible"]
  assert result["cadence"]["forward"]["speed_passing"]["paired_delta_hz"] == 0.
