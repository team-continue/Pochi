"""Monitor V19 speed-search checkpoints using frozen development profiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path


def sha(path):
  return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
  temp = path.with_suffix(".tmp")
  temp.write_text(json.dumps(value, indent=2))
  temp.replace(path)


def identity(pid):
  proc = Path(f"/proc/{pid}")
  try:
    return {
      "pid": pid,
      "process_start_ticks": proc.joinpath("stat")
      .read_text()
      .split(") ", 1)[1]
      .split()[19],
      "argv": proc.joinpath("cmdline").read_bytes().decode().rstrip("\0").split("\0"),
    }
  except FileNotFoundError:
    return None


def checkpoint_count(path):
  import torch

  from pochi_rl.mjlab.lp_acrl_speed_runner import validate_speed_checkpoint

  torch.set_num_threads(1)
  return validate_speed_checkpoint(
    torch.load(path, map_location="cpu", weights_only=False)
  )


def report_counts(path):
  report = json.loads(path.read_text())
  rows = report["episodes"]
  return {
    "report": str(path),
    "completed_updates": report["completed_updates"],
    "episodes": len(rows),
    "passed": sum(r["passed"] for r in rows),
    "falls": sum(r["fall"] for r in rows),
    "invalid": sum(r["invalid_state"] for r in rows),
    "summary": report["summary"],
    "retained_passed": sum(
      r["passed"]
      for r in rows
      if r["profile"] in report["profile_segments"]
      and not r["profile"].startswith("moving_")
      and any(r["profile"] == name for name in plan_retained_profiles())
    ),
    "pure_axis_all_trials_passed": {
      name: value
      for name, value in report["summary"].items()
      if name.startswith(("x_", "y_", "yaw_")) and value["passed"] == value["episodes"]
    },
  }


def plan_retained_profiles():
  from eval_lp_acrl_omni_phase import PROFILES

  return PROFILES


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--plan", type=Path, required=True)
  args = parser.parse_args()
  plan = json.loads(args.plan.read_text())
  output, run = Path(plan["diagnostics"]), Path(plan["run"])
  status_path = output / "monitor_status.json"
  state = {
    "status": "waiting",
    "stages": [],
    "held_out_seeds_used": False,
    "viewer_auto_started": False,
    "physical_gpu": 3,
    "started_utc": datetime.now(UTC).isoformat(),
  }
  environment = os.environ.copy()
  environment.update(
    CUDA_VISIBLE_DEVICES="3",
    MUJOCO_GL="egl",
    OMP_NUM_THREADS="4",
    OPENBLAS_NUM_THREADS="1",
    PYTHONUNBUFFERED="1",
    TMPDIR="/dev/shm/pochi_lp_acrl_compile_tmp",
    WARP_CACHE_PATH="/dev/shm/pochi_lp_acrl_warp_cache",
  )

  def verify_sources():
    for source, expected in plan["source_sha256"].items():
      assert sha(source) == expected, source

  try:
    verify_sources()
    write(status_path, state)
    for stage in plan["checkpoints"]:
      checkpoint = run / stage["filename"]
      while not checkpoint.exists():
        trainer = json.loads((output / "training_process.json").read_text())
        current = identity(trainer["pid"])
        if not current or any(current[k] != trainer[k] for k in current):
          state.update(
            status="trainer_stopped_before_pending_checkpoint",
            pending_checkpoint=str(checkpoint),
          )
          write(status_path, state)
          return
        time.sleep(30)
      before = (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
      time.sleep(2)
      assert before == (checkpoint.stat().st_size, checkpoint.stat().st_mtime_ns)
      # Wait for initialization/save to finish and preserve first reports.
      count = checkpoint_count(checkpoint)
      assert count == stage["completed_updates"]
      digest = sha(checkpoint)
      entry = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": digest,
        "completed_updates": count,
        "tests": {},
      }
      state["stages"].append(entry)
      for name in ("speed_search",):
        verify_sources()
        report = output / f"model_{checkpoint.stem.split('_')[-1]}_{name}.json"
        argv = [
          plan["python"],
          plan["scripts"][name],
          "--checkpoint",
          str(checkpoint),
          "--output",
          str(report),
        ]
        if report.exists():
          raise FileExistsError("Preserve earlier diagnostics")
        log_path = report.with_suffix(".log")
        with log_path.open("xb", buffering=0) as log:
          child = subprocess.Popen(
            argv, stdout=log, stderr=log, stdin=subprocess.DEVNULL, env=environment
          )
          state.update(
            status="evaluating",
            active_child=identity(child.pid),
            active_test=name,
            active_checkpoint=str(checkpoint),
          )
          write(status_path, state)
          result = child.wait()
        state.pop("active_child", None)
        state.pop("active_test", None)
        if result:
          raise RuntimeError(f"{name} diagnostic failed, code {result}; see {log_path}")
        entry["tests"][name] = report_counts(report)
        assert sha(checkpoint) == digest
        verify_sources()
        write(status_path, state)
      state.update(status="waiting")
      write(status_path, state)
    state.update(
      status="diagnostics_completed", finished_utc=datetime.now(UTC).isoformat()
    )
    write(status_path, state)
  except BaseException as error:
    state.update(status="failed", error=f"{type(error).__name__}: {error}")
    write(status_path, state)
    raise


if __name__ == "__main__":
  main()
