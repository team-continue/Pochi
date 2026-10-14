"""Recover JSON serialization failure from the first take's saved raw samples."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import eval_lp_acrl_omni_phase as base
import numpy as np


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--partial-report", required=True, type=Path)
  args = parser.parse_args()
  raw = args.partial_report.read_text()
  metadata = json.loads(
    raw[: raw.index('  "episodes": [')].rstrip().removesuffix(",") + "}"
  )
  recorded_script = args.partial_report.parent / "eval_lp_acrl_omni_transitions.py"
  spec = importlib.util.spec_from_file_location("recorded_diagnostic", recorded_script)
  diagnostic = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(diagnostic)
  samples_path = Path(metadata["samples"])
  assert (
    hashlib.sha256(samples_path.read_bytes()).hexdigest() == metadata["samples_sha256"]
  )
  for name, digest in metadata["source_sha256"].items():
    source = recorded_script if Path(name).name == recorded_script.name else Path(name)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == digest, name
  with np.load(samples_path) as archive:
    samples = {name: archive[name] for name in archive.files}
  assert samples["valid"].shape == (1000, 296)
  assert samples["valid"].all()
  assert all(s["falls"] == s["invalid"] == 0 for s in metadata["summary"].values())
  specs = diagnostic.profiles()
  assert json.loads(json.dumps(specs)) == metadata["profiles"]
  base.PROFILE_SEGMENTS = specs
  names = [tuple(specs)[i % len(specs)] for i in range(296)]
  rows = base.summarize_phase_profiles(
    samples,
    names,
    metadata["control_dt_s"],
    falls=np.zeros(296, bool),
    invalid=np.zeros(296, bool),
    horizon_steps=1000,
  )
  for row in rows:
    segments = specs[row["profile"]]
    changes = (
      [
        diagnostic.transition_metrics(
          samples,
          row["environment_index"],
          segments[j][0],
          segments[j - 1][2],
          segments[j][2],
          metadata["control_dt_s"],
        )
        for j in range(2, 5)
      ]
      if len(segments) > 1
      else []
    )
    row["steady_passed"] = row["passed"]
    row["transitions"] = changes
    row["passed"] = row["passed"] and all(e["passed"] for e in changes)
  recovered_summary = {}
  for name in specs:
    group = [r for r in rows if r["profile"] == name]
    recovered_summary[name] = {
      "episodes": len(group),
      "passed": sum(r["passed"] for r in group),
      "steady_passed": sum(r["steady_passed"] for r in group),
      "switches_passed": sum(all(t["passed"] for t in r["transitions"]) for r in group),
      "falls": sum(r["fall"] for r in group),
      "invalid": sum(r["invalid_state"] for r in group),
    }
  assert recovered_summary == metadata["summary"]
  metadata["episodes"] = rows
  metadata["serialization_recovery"] = {
    "cause": "NumPy boolean in transition checks could not be JSON serialized",
    "partial_report_preserved": str(args.partial_report),
    "raw_first_take_only_no_rerun": True,
    "same_frozen_metric_functions": True,
    "summary_identical_to_original_partial_report": True,
    "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
  }
  output = args.partial_report.with_suffix(".recovered.json")
  with output.open("x") as f:
    json.dump(metadata, f, indent=2, allow_nan=False, default=lambda x: x.item())
  print(
    json.dumps(
      {
        "output": str(output),
        "episodes": len(rows),
        "passed": sum(r["passed"] for r in rows),
      }
    )
  )


if __name__ == "__main__":
  main()
