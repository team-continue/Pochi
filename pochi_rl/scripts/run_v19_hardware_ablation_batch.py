"""Two independent GPU queues with a persistent, absolute-deadline watchdog."""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

LOCK = threading.Lock()
ACTIVE = {}


class AttachedProcess:
    """Observe a running training process without interrupting or relaunching it."""
    def __init__(self, pid, script, output):
        self.pid, self.script, self.output = pid, str(script), output

    def poll(self):
        try:
            command = Path(f"/proc/{self.pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
            if self.script in command:
                return None
        except FileNotFoundError:
            pass
        manifest = self.output / "run.json"
        status = json.loads(manifest.read_text()).get("status") if manifest.exists() else None
        return 0 if status in ("completed", "stopped_at_deadline") else 1

    def wait(self):
        while (code := self.poll()) is None:
            time.sleep(1)
        return code


def write_json(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--batch", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--deadline-guard", action="store_true")
    p.add_argument("--adopt-current", action="store_true")
    args = p.parse_args()
    root = args.batch.resolve()
    plan = json.loads((root / "batch_plan.json").read_text())
    if args.deadline_guard:
        # Separate process: enforce release even if the queue supervisor crashes.
        deadline = plan["release_gpus_at_epoch"]
        journal = {"pid": os.getpid(), "deadline": deadline, "actions": []}
        write_json(root / "deadline_guard.json", journal)
        stages = ((deadline - 30, signal.SIGUSR1), (deadline - 15, signal.SIGTERM), (deadline - 5, signal.SIGKILL))
        for scheduled, sig in stages:
            while time.time() < scheduled:
                time.sleep(max(0, min(5, scheduled - time.time())))
            state_path = root / "batch_status.json"
            if not state_path.exists():
                continue
            for job in json.loads(state_path.read_text()).get("jobs", {}).values():
                pid = job.get("pid")
                if not pid:
                    continue
                try:
                    command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
                    allowed = ("v19_hardware_ablation.py", "v19_hardware_ablation_video.py")
                    if not any(str(root / "source/scripts" / name) in command for name in allowed) or os.getpgid(pid) != pid:
                        continue
                    os.killpg(pid, sig)
                    journal["actions"].append({"pid": pid, "signal": int(sig), "at_epoch": time.time()})
                except (FileNotFoundError, ProcessLookupError):
                    continue
            write_json(root / "deadline_guard.json", journal)
        journal["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(root / "deadline_guard.json", journal)
        return
    source, checkpoint = root / "source", Path(plan["checkpoint"])
    script = source / "scripts/v19_hardware_ablation.py"
    video_script = source / "scripts/v19_hardware_ablation_video.py"
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == plan["checkpoint_sha256"]
    for relative, checksum in plan["source_sha256"].items():
        assert hashlib.sha256((source / relative).read_bytes()).hexdigest() == checksum, relative
    stop_training = plan["stop_training_at_epoch"]
    hard_deadline = plan["release_gpus_at_epoch"]
    state = {"pid": os.getpid(), "status": "running", "started_at": datetime.now(timezone.utc).isoformat(), "jobs": {}}
    state_path = root / ("smoke_batch_status.json" if args.smoke else "batch_status.json")
    if args.adopt_current:
        state = json.loads(state_path.read_text())
        state.update(pid=os.getpid(), status="running", postprocessing="3x3 video only",
            supervisor_restarted_at=datetime.now(timezone.utc).isoformat())

    def update(name, values):
        with LOCK:
            state["jobs"].setdefault(name, {}).update(values)
            write_json(state_path, state)

    def refresh_video_index():
        rows = []
        for variant in ("noise", "penalties", "noise_penalties", "noise_penalties_delay"):
            directory = root / variant
            row = {"variant": variant, "hardware_validated": False}
            if (directory / "run.json").exists():
                manifest = json.loads((directory / "run.json").read_text())
                row.update(training_status=manifest["status"], completed_updates=manifest.get("completed_updates"))
            policy = directory / "exports/latest/policy.torchscript.pt"
            if policy.exists():
                row["policy"] = str(policy)
            video = root / "videos" / f"{variant}_3x3.mp4"
            if video.exists() and video.with_suffix(".json").exists():
                row["video"] = str(video)
                row["poster"] = str(video.with_suffix(".jpg"))
            rows.append(row)
        with LOCK:
            write_json(root / "video_index.json", {"updated_at": datetime.now(timezone.utc).isoformat(),
                "variants": rows, "comparative_evaluation": False, "video_cases": 9, "video_duration_s": 20})

    def run_process(name, command, gpu, deadline, log_path, existing_pid=None):
        if time.time() >= deadline:
            return None
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=plan["gpu_uuids"][str(gpu)],
            MUJOCO_GL="egl", MUJOCO_EGL_DEVICE_ID="0", PYTHONUNBUFFERED="1",
            PYTHONPATH=str(source / "src") + os.pathsep + str(source / "scripts"),
            TMPDIR=str(root / "temporary"), WARP_CACHE_PATH=str(root / "warp_cache" / f"gpu{gpu}"),
            OMP_NUM_THREADS="4", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="4")
        Path(env["TMPDIR"]).mkdir(exist_ok=True)
        Path(env["WARP_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)
        with Path(log_path).open("a") as log:
            proc = AttachedProcess(existing_pid, script, root / name) if existing_pid else subprocess.Popen(
                command, env=env, cwd=source, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            with LOCK:
                ACTIVE[proc.pid] = {"proc": proc, "deadline": deadline, "stop_sent_at": None,
                    "term_sent_at": None, "kill_sent_at": None}
            update(name, {"pid": proc.pid, "physical_gpu": gpu, "status": "running", "log": str(log_path)})
            code = proc.wait()
            with LOCK:
                ACTIVE.pop(proc.pid, None)
            update(name, {"exit_code": code, "status": "completed" if code == 0 else "failed",
                "finished_at": datetime.now(timezone.utc).isoformat()})
            return code

    def watchdog():
        def send(pid, sig):
            try:
                os.killpg(pid, sig)
            except ProcessLookupError:
                pass
        while not done.is_set():
            now = time.time()
            with LOCK:
                for item in ACTIVE.values():
                    proc = item["proc"]
                    if proc.poll() is not None:
                        continue
                    if now >= item["deadline"] and item["stop_sent_at"] is None:
                        send(proc.pid, signal.SIGUSR1)
                        item["stop_sent_at"] = now
                    if (item["stop_sent_at"] is not None and now >= item["stop_sent_at"] + 30) or now >= hard_deadline - 15:
                        if item["term_sent_at"] is None:
                            send(proc.pid, signal.SIGTERM)
                            item["term_sent_at"] = now
                    if now >= hard_deadline - 5 or (item["term_sent_at"] is not None and now >= item["term_sent_at"] + 5):
                        if item["kill_sent_at"] is None:
                            send(proc.pid, signal.SIGKILL)
                            item["kill_sent_at"] = now
            done.wait(1)

    def export_files(checkpoint_path, output):
        if output.exists():
            return
        command = [sys.executable, str(script), "export", "--checkpoint", str(checkpoint_path),
            "--output", str(output), "--baseline-contract", plan["baseline_contract"]]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
            PYTHONPATH=str(source / "src") + os.pathsep + str(source / "scripts"))
        with (root / "export.log").open("a") as log:
            result = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, timeout=120)
        if result.returncode:
            raise RuntimeError(f"CPU export failed: {output}")

    def queue(gpu, variants):
        for variant in variants:
            prefix = "smoke_" if args.smoke else ""
            name = prefix + variant
            output = root / name
            if time.time() >= stop_training:
                update(name, {"status": "skipped_deadline"})
                continue
            adopted_pid = state["jobs"].get(name, {}).get("pid") if args.adopt_current and output.exists() else None
            code = run_process(name, [sys.executable, str(script), "train", "--checkpoint", str(checkpoint),
                "--output", str(output), "--variant", variant, "--gpu", str(gpu),
                "--num-envs", str(64 if args.smoke else plan["num_envs"]),
                "--iterations", str(3 if args.smoke else plan["iterations_per_variant"]),
                "--save-interval", str(1 if args.smoke else plan["save_interval"]),
                "--deadline", str(stop_training)], gpu, stop_training, root / f"{name}.log", adopted_pid)
            models = sorted(output.glob("model_*.pt"), key=lambda x: int(x.stem.split("_")[-1]))
            if not models:
                update(name, {"postprocessing": "no_trained_checkpoint"})
                continue
            latest = models[-1]
            export_files(latest, output / "exports/latest")
            for index in (1000, 2000, 3000):
                candidate = output / f"model_{index}.pt"
                if candidate.exists() and candidate != latest:
                    export_files(candidate, output / f"exports/update_{index + 1}")
            update(name, {"latest_checkpoint": str(latest), "policy": str(output / "exports/latest/policy.torchscript.pt")})
            if args.smoke:
                continue
            video = root / "videos" / f"{variant}_3x3.mp4"
            run_process(f"{name}/video_3x3", [sys.executable, str(video_script),
                "--checkpoint", str(latest), "--output", str(video), "--gpu", str(gpu)],
                gpu, min(hard_deadline - 60, time.time() + 1200), root / f"{name}_video.log")
            refresh_video_index()

    done = threading.Event()
    watcher = threading.Thread(target=watchdog, daemon=True)
    watcher.start()
    write_json(state_path, state)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(queue, int(gpu), variants) for gpu, variants in plan["queues"].items()]
            for future in futures:
                future.result()
        state["status"] = "finished"
        if not args.smoke:
            refresh_video_index()
    except BaseException as exc:
        state.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        done.set()
        state["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, state)


if __name__ == "__main__":
    main()
