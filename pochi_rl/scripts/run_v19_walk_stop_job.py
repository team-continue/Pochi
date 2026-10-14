"""Persistent, task-owned training/export/video supervisor with stage journals."""
import argparse
import fcntl
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from v19_hardware_ablation import write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--preflight", action="store_true")
    args = p.parse_args()
    project = Path(__file__).resolve().parents[1]
    root = project.parent
    output = root / ("preflight" if args.preflight else "training")
    output.mkdir(exist_ok=False)
    lock = (root / "gpu0.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": "0", "MUJOCO_GL": "egl",
           "PYOPENGL_PLATFORM": "egl", "PYTHONUNBUFFERED": "1",
           "OMP_NUM_THREADS": "4", "OPENBLAS_NUM_THREADS": "4",
           "PYTHONPATH": str(project/"src")+os.pathsep+str(project/"scripts"),
           "PATH": str(root / "tools") + os.pathsep + os.environ["PATH"]}
    python = str(project / ".venv/bin/python")
    source = root / "inputs/noise_penalties_model_3999.pt"
    checkpoint = output / "run" / ("model_2.pt" if args.preflight else "model_3999.pt")
    stages = []
    if args.preflight:
        stages.append(("unit_tests", [python, "-m", "pytest", "-q",
            "tests/test_walk_stop_robust.py", "tests/test_torque_margin.py"]))
    if args.preflight:
        stages.append(("physics_smoke", [python, "scripts/smoke_v19_walk_stop.py",
            "--checkpoint", str(source), "--output", str(output / "physics_smoke.json")]))
    else:
        preflight = json.loads((root / "preflight/status.json").read_text())
        assert preflight["status"] == "completed", "Complete preflight before full training"
    stages.extend([
        ("train", [python, "scripts/train_v19_walk_stop.py", "train", "--checkpoint", str(source),
            "--output", str(output / "run"), "--num-envs", "128" if args.preflight else "2048",
            "--iterations", "3" if args.preflight else "4000", "--save-interval", "1" if args.preflight else "100"]),
        ("export", [python, "scripts/train_v19_walk_stop.py", "export", "--checkpoint", str(checkpoint),
            "--output", str(output / "export"), "--baseline-contract", str(root / "inputs/baseline_policy_contract.json")]),
    ])
    if not args.preflight:
        stages.append(("video_3x3", [python, "scripts/v19_walk_stop_video.py", "--checkpoint", str(checkpoint),
            "--output", str(output / "video/walk_stop_3x3.mp4"), "--gpu", "0"]))
    if args.preflight:
        stages.append(("video_preview", [python, "scripts/v19_walk_stop_video.py",
            "--checkpoint", str(checkpoint), "--output", str(output/"video/preview.mp4"),
            "--gpu", "0", "--robust", "--preview-frames", "2"]))
    else:
        stages.append(("video_3x3_robust", [python, "scripts/v19_walk_stop_video.py",
            "--checkpoint", str(checkpoint), "--output", str(output/"video/walk_stop_robust_3x3.mp4"),
            "--gpu", "0", "--robust"]))
    state = {"status": "running", "supervisor_pid": os.getpid(), "stages": [],
             "started_at": datetime.now(timezone.utc).isoformat()}
    write_json(output / "status.json", state)
    try:
        for stage, command in stages:
            row = {"stage": stage, "command": command, "started_at": datetime.now(timezone.utc).isoformat()}
            state["stages"].append(row)
            with (output / f"{stage}.log").open("w") as log:
                child = subprocess.Popen(command, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, start_new_session=True)
                row["pid"] = child.pid
                state["active_stage"] = stage
                state["active_pid"] = child.pid
                write_json(output / "status.json", state)
                row["exit_code"] = child.wait()
            row["finished_at"] = datetime.now(timezone.utc).isoformat()
            write_json(output / "status.json", state)
            if row["exit_code"] != 0:
                raise RuntimeError(f"{stage} failed: inspect {output / (stage+'.log')}")
        state["status"] = "completed"
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        state.update(finished_at=datetime.now(timezone.utc).isoformat(), active_pid=None)
        write_json(output / "status.json", state)


if __name__ == "__main__":
    main()
