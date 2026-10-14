"""Hardware-log-informed V19 refinement for translation, balance and quiet stops."""
from __future__ import annotations

import argparse
import json
import os
import signal
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

import v19_hardware_ablation as base
from pochi_rl.mjlab.walk_stop_robust import configuration, safety_contract

STATE_KEY = "pochi_v19_walk_stop_state"
SOURCE_SHA = "d6cf6e1e40abb706dd9a9f3deb0092b4d9900e744cf0d46180f9043b42d0f631"
TASK = "Pochi-V19-Walk-Stop-20261013"


def validate_source(path, saved, backend):
    if base.digest(path) != SOURCE_SHA:
        raise ValueError("Warm start must be the verified noise_penalties/model_3999 checkpoint")
    state = saved["infos"]["pochi_v19_hardware_ablation_state"]
    assert state["variant"] == "noise_penalties"
    assert state["completed_updates"] == 4000 and saved["iter"] == 3999
    assert saved["infos"]["lp_acrl_contact_backend"] == backend
    for key in ("actor_state_dict", "critic_state_dict"):
        assert all(torch.isfinite(value).all() for value in saved[key].values())


def train(args):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends
    from pochi_rl.mjlab.warp_contact_backend import install_contact_backend

    def request_stop(*_):
        base.STOP_REQUESTED = True
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    backend = install_contact_backend()
    configure_torch_backends()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    validate_source(args.checkpoint, saved, backend)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg, agent = configuration(args.num_envs, args.seed)
    agent.max_iterations, agent.save_interval = args.iterations, args.save_interval
    cfg.sim.nan_guard.enabled = True
    cfg.sim.nan_guard.buffer_size = 16
    cfg.sim.nan_guard.max_envs_to_dump = 2
    cfg.sim.nan_guard.output_dir = str(output / "nan_guard")
    dump_yaml(output / "params/env.yaml", asdict(cfg))
    dump_yaml(output / "params/agent.yaml", asdict(agent))
    manifest = {
        "task": TASK, "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": base.digest(args.checkpoint),
        "initialization": "Exact noise_penalties/model3999 actor, critic, normalizers; fresh Adam and low-speed LP statistics",
        "seed": args.seed, "num_envs": args.num_envs,
        "physical_gpu": 0, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_name": torch.cuda.get_device_name(0),
        "iterations_requested": args.iterations, "save_interval": args.save_interval,
        "contact_backend": backend, "safety_contract": safety_contract(),
        "reward_weights": {k: v.weight for k, v in cfg.rewards.items()},
        "actor_noise_enabled": cfg.observations["actor"].enable_corruption,
        "source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[1])): base.digest(path)
            for path in (Path(__file__).resolve(), Path(__file__).with_name("v19_hardware_ablation.py").resolve(),
                         Path(__file__).resolve().parents[1]/"src/pochi_rl/mjlab/walk_stop_robust.py")
        },
        "status": "initializing", "started_at": datetime.now(timezone.utc).isoformat(),
    }
    base.write_json(output / "run.json", manifest)
    env = None
    base.STATE_KEY = STATE_KEY
    parent = base.make_runner_class()

    class MarginRunner(parent):
        def save(self, path, infos=None):
            super().save(path, {**(infos or {}), "pochi_torque_margin_contract": safety_contract()})

    try:
        env = ManagerBasedRlEnv(cfg, device="cuda:0")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = MarginRunner(wrapped, asdict(agent), str(output), "cuda:0",
            variant="walk_stop_16nm", source_sha=SOURCE_SHA, deadline=float("inf"))
        for model, key in ((runner.alg._raw_actor, "actor_state_dict"), (runner.alg._raw_critic, "critic_state_dict")):
            model.load_state_dict(saved[key], strict=True)
            assert all(torch.equal(v.cpu(), saved[key][k]) for k, v in model.state_dict().items())
        assert not runner.alg.optimizer.state
        runner.save(str(output / "initial_state.pt"))
        manifest.update(status="running", exact_initial_tensors_verified=True)
        base.write_json(output / "run.json", manifest)
        try:
            runner.learn(args.iterations)
        except base.DeadlineStop:
            manifest["status"] = "stopped_by_signal"
            runner.logger.stop_logging_writer()
        else:
            manifest["status"] = "completed"
        manifest["completed_updates"] = runner.completed_updates
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        base.write_json(output / "run.json", manifest)
        if env is not None:
            env.close()


def export(args):
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert saved["infos"]["pochi_torque_margin_contract"] == safety_contract()
    base.STATE_KEY = STATE_KEY
    base.export(args)
    contract_path = args.output / "policy_contract.json"
    contract = json.loads(contract_path.read_text())
    contract.update(task=TASK, safety_contract=safety_contract())
    base.write_json(contract_path, contract)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("train", "export"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=2048)
    p.add_argument("--iterations", type=int, default=4000)
    p.add_argument("--save-interval", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--baseline-contract", type=Path)
    args = p.parse_args()
    if min(args.iterations, args.num_envs, args.save_interval) < 1:
        p.error("iterations, num-envs and save-interval must be positive")
    {"train": train, "export": export}[args.mode](args)


if __name__ == "__main__":
    main()
