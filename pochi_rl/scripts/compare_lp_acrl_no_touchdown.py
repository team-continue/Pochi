"""Paired fixed-seed behavior and cadence comparison, without outcome tuning.

Cadence is the frozen >= 40 ms observed footfall metric divided by the full
scored segment duration. A nonstarter is never counted as slower walking.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import mean

PROFILES = ("stand", "forward", "backward", "stop_start")
SOURCE_SHA256 = "c292c8cf99aa8e2e56295104c0efc8e388374c00a2685b747c98b283e1c0c09a"
SLOTS = (100, 200, 300, 399)


def behavior_gate(report: dict, *, seed: int, num_envs: int) -> dict:
  """Check the fixed plane contract and independently reconstruct pass counts."""
  if report["protocol"] != "pochi-lp-acrl-behavior-plane-v2":
    raise ValueError("Expected unchanged plane-v2 protocol")
  if report["seed"] != seed or report["num_envs"] != num_envs:
    raise ValueError("Wrong registered seed or environment count")
  if report["height_scan_present"] or report["control_dt_s"] != .02:
    raise ValueError("Height scan/control timestep differs")
  if report["horizon_s"] != 20.0:
    raise ValueError("Expected fixed 20 s first episode")
  episodes = report["episodes"]
  if len(episodes) != num_envs or {
    row["environment_index"] for row in episodes
  } != set(range(num_envs)):
    raise ValueError("Missing or repeated world identity")
  summary = {}
  for profile in PROFILES:
    rows = [row for row in episodes if row["profile"] == profile]
    if len(rows) != num_envs // 4:
      raise ValueError("Missing or repeated behavior profile")
    for row in rows:
      if row["passed"] != (
        not row["fall"] and not row["invalid_state"]
        and all(segment["passed"] for segment in row["segments"])
      ):
        raise ValueError("Row pass flag disagrees with segment results")
    summary[profile] = {
      "episodes": len(rows), "passed": sum(row["passed"] for row in rows),
      "falls": sum(row["fall"] for row in rows),
      "invalid": sum(row["invalid_state"] for row in rows),
    }
  if summary != report["summary"]:
    raise ValueError("Saved summary disagrees with world results")
  limits = {"stand": 7, "forward": 6, "backward": 6, "stop_start": 6}
  if num_envs == 64:
    limits = {"stand": 14, "forward": 12, "backward": 12, "stop_start": 12}
  elif num_envs != 32:
    raise ValueError("Only registered screen/holdout sizes are accepted")
  passed = all(
    summary[p]["passed"] >= limits[p]
    and summary[p]["falls"] == 0 and summary[p]["invalid"] == 0
    for p in PROFILES
  )
  return {"eligible": passed, "required_passes": limits, "summary": summary}


def moving_segments(report: dict) -> dict:
  result = {}
  for row in report["episodes"]:
    for segment_index, segment in enumerate(row["segments"]):
      if segment["command_vx_vy_wz"][0] == 0:
        continue
      key = (row["environment_index"], segment_index)
      duration = segment["end_s"] - segment["start_s"] - segment[
        "excluded_transient_s"
      ]
      expected_duration = 4. if row["profile"] == "stop_start" else 18.
      if duration != expected_duration:
        raise ValueError("Scored moving duration differs from frozen protocol")
      metrics = segment["metrics"]
      complete = (
        segment["complete"] and not row["fall"] and not row["invalid_state"]
        and metrics is not None
      )
      speed_pass = complete and segment["checks"]["forward_error_mean_m_s"]
      result[key] = {
        "profile": row["profile"], "segment_index": segment_index,
        "command": segment["command_vx_vy_wz"], "duration_s": duration,
        "complete": bool(complete), "speed_pass": bool(speed_pass),
        "behavior_pass": bool(complete and segment["passed"]),
        "cadence_per_foot_hz": (
          [value / duration for value in metrics["footfalls_per_leg"]]
          if complete else None
        ),
        "actual_mean_vx_m_s": metrics["actual_mean_vx_vy_wz"][0]
        if complete else None,
      }
  return result


def aggregate(rows: list[dict]) -> dict:
  if not rows:
    return {"segments": 0, "mean_cadence_per_foot_hz": None,
            "mean_cadence_hz": None, "mean_vx_m_s": None}
  per_foot = [mean(row["cadence_per_foot_hz"][i] for row in rows) for i in range(4)]
  return {
    "segments": len(rows), "mean_cadence_per_foot_hz": per_foot,
    "mean_cadence_hz": mean(per_foot),
    "mean_vx_m_s": mean(row["actual_mean_vx_m_s"] for row in rows),
  }


def compare(baseline: dict, candidate: dict) -> dict:
  if baseline["task"] != "Pochi-LP-ACRL-Blind-v13":
    raise ValueError("Baseline must be V13/700")
  if baseline["checkpoint_sha256"] != SOURCE_SHA256:
    raise ValueError("Wrong baseline checkpoint")
  if candidate["task"] != "Pochi-LP-ACRL-Blind-v15":
    raise ValueError("Candidate must be V15")
  slot = candidate["no_touchdown_state"]["completed_updates"] - 1
  if slot not in SLOTS:
    raise ValueError("Candidate is outside preregistered checkpoint slots")
  baseline_gate = behavior_gate(baseline, seed=1456, num_envs=32)
  candidate_gate = behavior_gate(candidate, seed=1456, num_envs=32)
  old, new = moving_segments(baseline), moving_segments(candidate)
  if old.keys() != new.keys():
    raise ValueError("Paired worlds/segments differ")
  for key in old:
    for field in ("profile", "segment_index", "command", "duration_s"):
      if old[key][field] != new[key][field]:
        raise ValueError("Paired command or segment differs")
  results = {}
  for profile in ("forward", "backward", "stop_start"):
    keys = [key for key in old if old[key]["profile"] == profile]
    item = {"total_moving_segments_per_policy": len(keys)}
    for label, field in (("speed_passing", "speed_pass"),
                         ("behavior_passing", "behavior_pass")):
      paired = [key for key in keys if old[key][field] and new[key][field]]
      before = aggregate([old[key] for key in paired])
      after = aggregate([new[key] for key in paired])
      delta = None if not paired else (
        after["mean_cadence_hz"] - before["mean_cadence_hz"]
      )
      item[label] = {
        "baseline_unpaired": aggregate([old[k] for k in keys if old[k][field]]),
        "candidate_unpaired": aggregate([new[k] for k in keys if new[k][field]]),
        "jointly_passing_world_segments": [list(key) for key in paired],
        "paired_baseline": before, "paired_candidate": after,
        "paired_delta_hz": delta,
        "paired_relative_change": None if delta is None or not before[
          "mean_cadence_hz"
        ] else delta / before["mean_cadence_hz"],
      }
    results[profile] = item
  return {
    "baseline_behavior": baseline_gate, "candidate_behavior": candidate_gate,
    "candidate_iteration": slot,
    "cadence_definition": (
      "Frozen observed landings after >=0.04 s air / fixed scored seconds "
      "(18 s constant, 4 s stop/start moving segment), averaged equally across "
      "four feet and listed complete segments. Frozen boundary censoring is "
      "retained; these counts do not prove >=25 mm lift. Incomplete/fallen/invalid "
      "segments excluded; speed failures never counted as slower walking."
    ),
    "primary_effect": "Paired cadence delta on jointly speed-passing worlds",
    "sensitivity_effect": "Paired cadence delta on jointly behavior-passing worlds",
    "cadence_used_for_selection": False,
    "no_claim_of_causal_isolation_from_further_training": True,
    "cadence": results,
  }


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--baseline", type=Path, required=True)
  parser.add_argument("--candidate", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  result = compare(json.loads(args.baseline.read_text()),
                   json.loads(args.candidate.read_text()))
  result["input_sha256"] = {
    str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
    for path in (args.baseline, args.candidate, Path(__file__))
  }
  args.output.parent.mkdir(parents=True, exist_ok=True)
  with args.output.open("x") as stream:
    json.dump(result, stream, indent=2, allow_nan=False)
    stream.write("\n")
  print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
  main()
