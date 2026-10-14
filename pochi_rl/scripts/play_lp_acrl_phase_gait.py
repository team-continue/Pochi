"""Show the no-touchdown-bonus policy with the standard flat Viser controls."""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path

from play_lp_acrl import install_flat_command_gui

TASK = "Pochi-LP-ACRL-Blind-v17"


def run(
  checkpoint: Path,
  *,
  task_id: int = 2,
  device: str = "cuda:0",
  host: str = "0.0.0.0",
  port: int = 8080,
) -> None:
  """Load the validated V17 actor and start its unchanged native simulator."""
  if type(task_id) is not int or not 0 <= task_id < 5:
    raise ValueError("V17 requires a flat task ID between0 and4")
  if not 1 <= port <= 65535:
    raise ValueError("Port must be between1 and65535")
  checkpoint = checkpoint.expanduser().resolve(strict=True)
  os.environ.setdefault("MUJOCO_GL", "egl")
  import viser
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends
  from mjlab.viewer import ViserPlayViewer

  from pochi_rl.mjlab.lp_acrl_blind_v17 import register_blind_v17
  from pochi_rl.mjlab.warp_contact_backend import install_contact_backend

  install_contact_backend()
  register_blind_v17()
  configure_torch_backends()
  env_cfg = load_env_cfg(TASK, play=True)
  agent_cfg = load_rl_cfg(TASK)
  env_cfg.scene.num_envs = 1
  env_cfg.commands["base_velocity"].manual_control_on_play = True
  runner_cls = load_runner_cls(TASK)
  if runner_cls is None:
    raise RuntimeError("V17 requires its checkpoint-validating runner")
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
  server = None
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = runner_cls(wrapped, asdict(agent_cfg), device=device)
    runner.load(
      str(checkpoint), load_cfg={"actor": True}, strict=True, map_location=device
    )
    policy = runner.get_inference_policy(device=device)
    command = env.command_manager.get_term("base_velocity")
    install_flat_command_gui(command)
    command.set_evaluation(task_ids=task_id, command=(0.0, 0.0, 0.0))
    env.reset()
    print(f"[LP-ACRL V17] Checkpoint: {checkpoint}", flush=True)
    print(
      f"[LP-ACRL V17] Fixed flat task{task_id}; initial command=(0,0,0), "
      "manual Enable=on; lin_vel_x=-0.5..+0.5m/s; Zero stops",
      flush=True,
    )
    server = viser.ViserServer(host=host, port=port, label="Pochi LP-ACRL V17")
    if server.get_port() != port:
      raise RuntimeError(f"Requested port{port} is occupied; choose another --port")
    ViserPlayViewer(wrapped, policy, viser_server=server).run()
  finally:
    if server is not None:
      server.stop()
    env.close()


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--task-id", type=int, default=2)
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--host", default="0.0.0.0")
  parser.add_argument("--port", type=int, default=8080)
  run(**vars(parser.parse_args()))


if __name__ == "__main__":
  main()
