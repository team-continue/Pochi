"""Isolated V19 hardware robustness ablations; never connects to the robot."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from types import MethodType

import torch
from tensordict import TensorDict

VARIANTS = ("noise", "penalties", "noise_penalties", "noise_penalties_delay")
STATE_KEY = "pochi_v19_hardware_ablation_state"
STOP_REQUESTED = False


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def configuration(variant, num_envs, seed, *, play=False):
    from pochi_rl.mjlab.lp_acrl_blind_v19_env_cfg import (
        pochi_lp_acrl_blind_v19_env_cfg, pochi_lp_acrl_blind_v19_ppo_runner_cfg,
    )
    cfg = pochi_lp_acrl_blind_v19_env_cfg(play=play)
    agent = pochi_lp_acrl_blind_v19_ppo_runner_cfg()
    cfg.seed = agent.seed = seed
    cfg.scene.num_envs = num_envs
    cfg.commands["base_velocity"].debug_vis = False
    cfg.observations["actor"].enable_corruption = "noise" in variant and not play
    cfg.observations["critic"].enable_corruption = False
    if "penalties" in variant:
        for key, weight in {
            "joint_vel_l2": -0.008, "ang_vel_xy_l2": -0.15,
            "joint_torques_l2": -0.0001, "target_position_rate": -0.64,
        }.items():
            cfg.rewards[key].weight = weight
    actuator = cfg.scene.entities["robot"].articulation.actuators[0]
    assert (actuator.stiffness, actuator.damping) == (140.0, 1.5)
    assert (actuator.effort_limit, actuator.saturation_effort) == (17.0, 17.0)
    if variant.endswith("_delay") and not play:
        actuator.delay_min_lag = 0
        actuator.delay_max_lag = 4  # 0..10ms, physics dt=2.5ms, NOT policy dt.
        actuator.delay_update_period = 400  # Resample about once per second.
        actuator.delay_hold_prob = 0.0
    agent.upload_model = False
    return cfg, agent


class DeadlineStop(RuntimeError):
    pass


def make_runner_class():
    from pochi_rl.mjlab.lp_acrl_runner import (
        LPACRLOnPolicyRunner, _capture_with_replay_inputs, _log_with_all_episode_keys,
    )
    from pochi_rl.mjlab.runner import PochiOnPolicyRunner
    from pochi_rl.mjlab.lp_acrl_phase_gait import PhaseQuietStopMLP
    from pochi_rl.mjlab.lp_acrl_symmetry import BLIND_OBSERVATION_TERMS
    from pochi_rl.mjlab.warp_contact_backend import active_contact_backend_identity
    from pochi_rl.robot import JOINT_NAMES

    class AblationRunner(LPACRLOnPolicyRunner):
        def __init__(self, *args, variant, source_sha, deadline, **kwargs):
            PochiOnPolicyRunner.__init__(self, *args, **kwargs)
            cfg = self.env.unwrapped.cfg
            assert int(os.environ.get("WORLD_SIZE", "1")) == 1
            for name in ("actor", "critic"):
                assert tuple(cfg.observations[name].terms) == (*BLIND_OBSERVATION_TERMS, "gait_phase")
            assert tuple(cfg.actions["joint_pos"].actuator_names) == tuple(JOINT_NAMES)
            assert isinstance(self.alg._raw_actor, PhaseQuietStopMLP)
            assert self.alg.symmetry.data_augmentation_func.__name__ == "augment_phase_symmetry"
            self.variant, self.source_sha, self.deadline = variant, source_sha, deadline
            self._backend = active_contact_backend_identity()
            self.completed_updates = 0
            self.logger.git_status_repos = []
            self.logger._lp_acrl_original_log = self.logger.log
            self.logger.log = MethodType(self._log_progress.__func__, self.logger)
            self.logger._ablation_runner = self
            guard = self.env.unwrapped.sim.nan_guard
            if guard.enabled:
                guard._lp_acrl_original_capture = guard.capture
                guard.capture = MethodType(_capture_with_replay_inputs, guard)

        def _log_progress(logger, *args, **kwargs):
            _log_with_all_episode_keys(logger, *args, **kwargs)
            runner = logger._ablation_runner
            runner.completed_updates = int(kwargs["it"]) + 1
            if runner.completed_updates == 1 or runner.completed_updates % 50 == 0:
                write_json(Path(logger.log_dir) / "progress.json", {
                    "completed_updates": runner.completed_updates,
                    "last_update_at": datetime.now(timezone.utc).isoformat(),
                    "collect_seconds": kwargs["collect_time"],
                    "learn_seconds": kwargs["learn_time"],
                    "learning_rate": runner.alg.learning_rate,
                })
            if STOP_REQUESTED or time.time() >= runner.deadline:
                runner.save(str(Path(logger.log_dir) / f"model_{runner.completed_updates - 1}.pt"))
                raise DeadlineStop("Saved at the next complete PPO update")

        def save(self, path, infos=None):
            assert active_contact_backend_identity() == self._backend
            count = self.completed_updates
            cursor = self.current_learning_iteration
            self.current_learning_iteration = count - 1
            temporary = str(path) + ".partial"
            try:
                super().save(temporary, {
                    **(infos or {}),
                    "lp_acrl_contact_backend": self._backend,
                    "lp_acrl_training_exploration": self.alg._raw_actor.exploration_identity(),
                    STATE_KEY: {
                        "version": 1, "variant": self.variant,
                        "source_checkpoint_sha256": self.source_sha,
                        "completed_updates": count,
                        "learning_rate": self.alg.learning_rate,
                        "actor_noise_enabled": self.env.unwrapped.cfg.observations["actor"].enable_corruption,
                    },
                })
                Path(temporary).replace(path)
            finally:
                self.current_learning_iteration = cursor
    return AblationRunner


def train(args):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.os import dump_yaml
    from mjlab.utils.torch import configure_torch_backends
    from pochi_rl.mjlab.lp_acrl_speed_runner import validate_speed_checkpoint
    from pochi_rl.mjlab.warp_contact_backend import install_contact_backend

    def request_stop(*_):
        global STOP_REQUESTED
        STOP_REQUESTED = True
    signal.signal(signal.SIGUSR1, request_stop)
    signal.signal(signal.SIGTERM, request_stop)
    backend = install_contact_backend()
    configure_torch_backends()
    torch.set_num_threads(4)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert validate_speed_checkpoint(saved) == 6000
    assert saved["infos"]["lp_acrl_contact_backend"] == backend
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cfg, agent = configuration(args.variant, args.num_envs, args.seed)
    agent.max_iterations, agent.save_interval = args.iterations, args.save_interval
    guard = output / "nan_guard"
    cfg.sim.nan_guard.enabled = True
    cfg.sim.nan_guard.buffer_size = 16
    cfg.sim.nan_guard.max_envs_to_dump = 2
    cfg.sim.nan_guard.output_dir = str(guard)
    # Save before ObservationManager removes disabled term noise in place.
    dump_yaml(output / "params/env.yaml", asdict(cfg))
    dump_yaml(output / "params/agent.yaml", asdict(agent))
    manifest = {
        "task": "Pochi-V19-Hardware-Ablation-20261010",
        "variant": args.variant, "source_checkpoint": str(args.checkpoint.resolve()),
        "source_checkpoint_sha256": digest(args.checkpoint),
        "initialization": "Exact V19/model5999 actor, critic and normalizers; fresh Adam and LP statistics",
        "seed": args.seed, "num_envs": args.num_envs,
        "physical_gpu": args.gpu, "gpu_uuid": os.environ["CUDA_VISIBLE_DEVICES"],
        "iterations_requested": args.iterations, "save_interval": args.save_interval,
        "stop_training_at_epoch": args.deadline, "contact_backend": backend,
        "status": "initializing", "started_at": datetime.now(timezone.utc).isoformat(),
        "actor_noise_enabled": cfg.observations["actor"].enable_corruption,
        "reward_weights": {k: v.weight for k, v in cfg.rewards.items()},
        "actuator": asdict(cfg.scene.entities["robot"].articulation.actuators[0]),
    }
    write_json(output / "run.json", manifest)
    env = None
    try:
        env = ManagerBasedRlEnv(cfg, device="cuda:0")
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        runner = make_runner_class()(wrapped, asdict(agent), str(output), "cuda:0",
            variant=args.variant, source_sha=manifest["source_checkpoint_sha256"], deadline=args.deadline)
        for model, key in ((runner.alg._raw_actor, "actor_state_dict"), (runner.alg._raw_critic, "critic_state_dict")):
            model.load_state_dict(saved[key], strict=True)
            assert all(torch.equal(v.cpu(), saved[key][k]) for k, v in model.state_dict().items())
        assert not runner.alg.optimizer.state
        runner.save(str(output / "initial_state.pt"))
        manifest.update(status="running", exact_initial_tensors_verified=True)
        write_json(output / "run.json", manifest)
        try:
            runner.learn(args.iterations)
        except DeadlineStop:
            manifest["status"] = "stopped_at_deadline"
            runner.logger.stop_logging_writer()
        else:
            manifest["status"] = "completed"
        manifest["completed_updates"] = runner.completed_updates
    except BaseException as exc:
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
        write_json(output / "run.json", manifest)
        if env is not None:
            env.close()


def actor_from_saved(saved):
    from pochi_rl.mjlab.lp_acrl_phase_gait import PhaseQuietStopMLP
    _, agent = configuration("noise", 1, 42, play=True)
    cfg = asdict(agent.actor)
    actor = PhaseQuietStopMLP(TensorDict({"actor": torch.zeros(1, 50)}, batch_size=[1]),
        {"actor": ["actor"]}, "actor", 12,
        **{k: cfg[k] for k in ("hidden_dims", "activation", "obs_normalization", "distribution_cfg")})
    actor.load_state_dict(saved["actor_state_dict"], strict=True)
    return actor.eval()


def export(args):
    torch.set_num_threads(1)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    state = saved["infos"].get(STATE_KEY)
    if state is None:
        assert "lp_acrl_speed_phase_state" in saved["infos"]
        state = {"variant": "baseline_v19", "completed_updates": 6000}
    assert saved["iter"] == state["completed_updates"] - 1
    assert all(torch.isfinite(v).all() for v in saved["actor_state_dict"].values())
    actor = actor_from_saved(saved)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    model = output / "policy.torchscript.pt"
    torch.jit.script(actor.as_jit().eval()).save(str(model))
    loaded = torch.jit.load(str(model), map_location="cpu").eval()
    inputs = torch.randn(4096, 50, generator=torch.Generator().manual_seed(21456))
    inputs[:512, 9:12] = 0.0
    inputs[:512, 48:50] = 0.0
    with torch.inference_mode():
        expected = actor(TensorDict({"actor": inputs}, batch_size=[len(inputs)]))
        actual = loaded(inputs)
    torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
    base = Path(args.baseline_contract)
    contract = json.loads(base.read_text())
    contract.update(task="Pochi-V19-Hardware-Ablation-20261010", variant=state["variant"],
        checkpoint=str(args.checkpoint.resolve()), checkpoint_sha256=digest(args.checkpoint),
        completed_updates=state["completed_updates"], model=model.name, model_sha256=digest(model),
        real_robot_validation="Not performed",
        validation={"observations": len(inputs), "standing_observations": 512,
            "torchscript_max_abs_error": float((actual - expected).abs().max())})
    write_json(output / "policy_contract.json", contract)
    print(json.dumps({"model": str(model), "validation": contract["validation"]}))


@torch.inference_mode()
def evaluate(args):
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.utils.torch import configure_torch_backends
    from pochi_rl.mjlab.warp_contact_backend import install_contact_backend
    import eval_lp_acrl_speed_search as evaluation
    from eval_lp_acrl_behavior_plane import FreshContactStepWrapper
    backend = install_contact_backend()
    configure_torch_backends()
    torch.set_num_threads(4)
    profiles = {
        "standing": ((0.0, 20.0, (0.0, 0.0, 0.0)),),
        **{f"forward_{v:+g}": ((0.0, 20.0, (v, 0.0, 0.0)),) for v in (-0.3, -0.1, 0.1, 0.3, 0.6)},
        **{f"lateral_{v:+g}": ((0.0, 20.0, (0.0, v, 0.0)),) for v in (-0.1, 0.1)},
        **{f"yaw_{v:+g}": ((0.0, 20.0, (0.0, 0.0, v)),) for v in (-0.3, 0.3)},
        "diagonal": ((0.0, 20.0, (0.2, 0.1, 0.0)),),
        "moving_low_speed": (
            (0.0, 2.0, (0.0, 0.0, 0.0)), (2.0, 6.0, (0.1, 0.0, 0.0)),
            (6.0, 10.0, (0.1, 0.05, 0.2)), (10.0, 14.0, (0.1, -0.05, -0.2)),
            (14.0, 18.0, (0.1, 0.0, 0.0)), (18.0, 20.0, (0.0, 0.0, 0.0)),
        ),
    }
    evaluation.PROFILE_SEGMENTS, evaluation.PROFILES = profiles, tuple(profiles)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor = actor_from_saved(saved).to("cuda:0")
    cfg, agent = configuration("noise", len(profiles) * 4, 21456, play=True)
    cfg.auto_reset = False
    if args.condition == "noise_delay":
        actuator = cfg.scene.entities["robot"].articulation.actuators[0]
        actuator.delay_min_lag = actuator.delay_max_lag = 4
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        generator = torch.Generator(device="cuda:0").manual_seed(21456)
        noise_range = torch.tensor([.1] * 3 + [.2] * 3 + [.05] * 3 + [0.] * 3 +
            [.01] * 12 + [1.5] * 12 + [0.] * 14, device="cuda:0")
        def policy(obs):
            if args.condition != "clean":
                obs = obs.clone()
                obs["actor"] = obs["actor"] + (2 * torch.rand(obs["actor"].shape,
                    device="cuda:0", generator=generator) - 1) * noise_range
            return actor(obs)
        samples, rows = evaluation.evaluate_profiles(env, FreshContactStepWrapper(env, wrapped), policy)
        mask = samples["valid"] & (samples["time_s"][:, None] > 2.)
        standing = [i for i in range(env.num_envs) if tuple(profiles)[i % len(profiles)] == "standing"]
        stand_mask = mask[:, standing]
        velocities = samples["joint_velocity"][:, standing][stand_mask]
        actions = samples["action_raw"][:, standing]
        delta = .25 * (actions[1:].clip(-4, 4) - actions[:-1].clip(-4, 4))
        delta_mask = stand_mask[1:] & stand_mask[:-1]
        report = {
            "protocol": "v19-hardware-ablation-low-speed-screen-v1",
            "condition": args.condition, "checkpoint": str(args.checkpoint.resolve()),
            "checkpoint_sha256": digest(args.checkpoint), "seed": 21456,
            "num_envs": env.num_envs, "profile_segments": profiles,
            "contact_backend": backend, "episodes": rows,
            "passed_episodes": sum(r["passed"] for r in rows),
            "falls": sum(r["fall"] for r in rows), "invalid": sum(r["invalid_state"] for r in rows),
            "standing_joint_speed_rms_rad_s": float((velocities ** 2).mean() ** .5) if velocities.size else None,
            "standing_target_step_rms_deg": float((delta[delta_mask] ** 2).mean() ** .5 * 180 / 3.141592653589793) if delta_mask.any() else None,
            "scope": "Held-out 20s simulation profiles, 4 repeats each; clean, legacy-range noise, or noise plus 10ms command delay. No hardware validation.",
        }
        write_json(args.output, report)
        print(json.dumps({k: v for k, v in report.items() if k not in ("episodes", "profile_segments", "contact_backend")}))
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("train", "export", "evaluate"))
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variant", choices=VARIANTS, default="noise")
    p.add_argument("--gpu", type=int, choices=(2, 3), default=2)
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=4000)
    p.add_argument("--save-interval", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--deadline", type=float, default=float("inf"))
    p.add_argument("--condition", choices=("clean", "noise", "noise_delay"), default="clean")
    p.add_argument("--baseline-contract", type=Path)
    args = p.parse_args()
    {"train": train, "export": export, "evaluate": evaluate}[args.mode](args)


if __name__ == "__main__":
    main()
