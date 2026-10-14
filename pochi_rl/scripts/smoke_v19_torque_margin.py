"""Finite-state GPU integration check; does not claim locomotion competence."""
import argparse
import json
from pathlib import Path

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.torch import configure_torch_backends

import v19_hardware_ablation as base
from pochi_rl.mjlab.torque_margin import configuration
from pochi_rl.mjlab.warp_contact_backend import install_contact_backend


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    backend = install_contact_backend()
    configure_torch_backends()
    torch.set_num_threads(4)
    cfg, agent = configuration(64, 42, play=True)
    cfg.auto_reset = False
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    actor = base.actor_from_saved(saved).to("cuda:0")
    env = ManagerBasedRlEnv(cfg, device="cuda:0")
    try:
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        obs = wrapped.get_observations()
        action_term = env.action_manager.get_term("joint_pos")
        assert action_term.motor in env.scene["robot"]._custom_actuators
        maxima = {"applied_nm": 0., "raw_policy_pd_nm": 0., "substep_pd_nm": 0.}
        with torch.inference_mode():
            for step in range(120):
                actions = actor(obs).clip(-4, 4)
                obs, reward, dones, extras = wrapped.step(actions)
                assert torch.isfinite(reward).all()
                assert all(torch.isfinite(value).all() for value in obs.values())
                assert torch.isfinite(env.scene["robot"].data.joint_vel).all()
                torch.testing.assert_close(action_term.raw_action, actions)
                assert torch.equal(env.action_manager.action, actions)
                for key, tensor in (
                    ("applied_nm", env.scene["robot"].data.actuator_force),
                    ("raw_policy_pd_nm", action_term.raw_request),
                    ("substep_pd_nm", action_term.motor.peak_request),
                ):
                    maxima[key] = max(maxima[key], tensor.abs().max().item())
        assert maxima["applied_nm"] <= 8.0001
        assert maxima["substep_pd_nm"] > 0.
        base.write_json(args.output, {
            "status": "passed", "steps": 120, "num_envs": env.num_envs,
            "contact_backend": backend, "maxima": maxima,
            "raw_last_action_preserved": True,
            "scope": "2.4-second deterministic smoke; no comparative or hardware evaluation",
        })
        print(json.dumps(maxima))
    finally:
        env.close()


if __name__ == "__main__":
    main()
