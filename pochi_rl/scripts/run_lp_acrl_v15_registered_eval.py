"""Launch one explicitly selected V15 trial with exclusive provenance files."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from compare_lp_acrl_no_touchdown import SOURCE_SHA256, behavior_gate

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / (
  "logs/lp_acrl_blind_diagnostics/plane_v2/v15_no_touchdown_protocol_manifest.json"
)


def digest(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def launch(args) -> dict:
  manifest = json.loads(MANIFEST.read_text())
  for relative, expected in manifest["source_sha256"].items():
    if digest(ROOT / relative) != expected:
      raise ValueError(f"Frozen source changed: {relative}")
  if not args.checkpoint.is_file():
    raise ValueError("Missing checkpoint")
  checkpoint_sha = digest(args.checkpoint)
  if args.stage == "baseline":
    if checkpoint_sha != SOURCE_SHA256 or args.slot is not None:
      raise ValueError("Baseline must be the exact V13/700 source without slot")
    wrapper = "eval_lp_acrl_behavior_blind_v13.py"
    seed, size = 1456, 32
  else:
    wrapper = "eval_lp_acrl_behavior_blind_v15.py"
    if args.slot not in (100, 200, 300, 399):
      raise ValueError("V15 trial requires a registered checkpoint slot")
    seed, size = (1789, 64) if args.stage == "holdout" else (1456, 32)
  paths = {
    "report": args.output,
    "log": args.output.with_suffix(".log"),
    "provenance": args.output.with_suffix(".provenance.json"),
    "validation": args.output.with_suffix(".validation.json"),
  }
  if any(path.exists() for path in paths.values()):
    raise ValueError("Refusing to replace any previous trial artifact")
  args.output.parent.mkdir(parents=True, exist_ok=True)
  env = os.environ.copy()
  overrides = {
    "CUDA_VISIBLE_DEVICES": str(args.physical_gpu), "MUJOCO_GL": "egl",
    "MUJOCO_EGL_DEVICE_ID": "0", "PYTHONUNBUFFERED": "1",
    "WARP_CACHE_PATH": "/dev/shm/pochi_lp_acrl_warp_cache",
    "TMPDIR": "/dev/shm/pochi_lp_acrl_compile_tmp",
  }
  env.update(overrides)
  command = [
    sys.executable, str(ROOT / "scripts" / wrapper),
    "--checkpoint", str(args.checkpoint.resolve()), "--device", "cuda:0",
    "--seed", str(seed), "--num-envs", str(size),
    "--output", str(args.output.resolve()),
  ]
  provenance = {
    "status": "running", "stage": args.stage, "slot": args.slot,
    "started_at_utc": datetime.now(UTC).isoformat(), "command": command,
    "environment_overrides": overrides, "physical_gpu": args.physical_gpu,
    "checkpoint_sha256": checkpoint_sha,
    "protocol_manifest": str(MANIFEST), "protocol_manifest_sha256": digest(MANIFEST),
    "all_frozen_sources_checked_before_launch": True,
    "seed": seed, "num_envs": size,
    "automatic_selection_or_holdout_launch": False,
  }
  with paths["provenance"].open("x") as stream:
    json.dump(provenance, stream, indent=2)
    stream.write("\n")
  process = None
  try:
    with paths["log"].open("x") as stream:
      process = subprocess.Popen(command, cwd=ROOT, env=env,
                                 stdout=stream, stderr=subprocess.STDOUT)
      provenance["pid"] = process.pid
      paths["provenance"].write_text(json.dumps(provenance, indent=2) + "\n")
      result = process.wait()
    if result != 0:
      raise RuntimeError(f"Evaluation exited {result}; see preserved log")
    if digest(args.checkpoint) != checkpoint_sha:
      raise ValueError("Checkpoint changed during trial")
    for relative, expected in manifest["source_sha256"].items():
      if digest(ROOT / relative) != expected:
        raise ValueError(f"Frozen source changed during trial: {relative}")
    report = json.loads(args.output.read_text())
    if report["checkpoint_sha256"] != checkpoint_sha:
      raise ValueError("Report names a different checkpoint")
    if args.stage != "baseline" and report["no_touchdown_state"][
      "completed_updates"
    ] != args.slot + 1:
      raise ValueError("Checkpoint local iteration differs from registered slot")
    validation = behavior_gate(report, seed=seed, num_envs=size)
    with paths["validation"].open("x") as stream:
      json.dump(validation, stream, indent=2)
      stream.write("\n")
    provenance.update(
      status="completed", report_sha256=digest(args.output),
      validation_sha256=digest(paths["validation"]),
      all_frozen_sources_checked_after_completion=True,
    )
    return validation
  except BaseException as error:
    if process is not None and process.poll() is None:
      process.terminate()
      process.wait(timeout=30)
    provenance.update(
      status="failed", error_type=type(error).__name__, error=str(error),
    )
    raise
  finally:
    provenance["finished_at_utc"] = datetime.now(UTC).isoformat()
    paths["provenance"].write_text(json.dumps(provenance, indent=2) + "\n")


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--stage", choices=("baseline", "screen", "holdout"), required=True,
  )
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--slot", type=int)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--physical-gpu", type=int, default=2)
  args = parser.parse_args()
  print(json.dumps(launch(args), indent=2))


if __name__ == "__main__":
  main()
