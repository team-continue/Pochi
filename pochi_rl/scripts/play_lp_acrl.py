"""Open the blind LP-ACRL policy in Viser, initially stopped on a fixed task.

Example:
  uv run python scripts/play_lp_acrl.py --checkpoint <model.pt> --device cuda:2

The default task360 starts on the flat central platform of the 2cm ascending
stairs tile. Commands start at zero with manual controls enabled. The ordinary
policy, simulator, action wrapper and Viser camera controls are preserved.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import asdict
from pathlib import Path
from types import MethodType, SimpleNamespace

TASKS = (
  "Pochi-LP-ACRL-Blind-v0",
  "Pochi-LP-ACRL-Blind-v1",
  "Pochi-LP-ACRL-Blind-v2",
  "Pochi-LP-ACRL-Blind-v3",
  "Pochi-LP-ACRL-Blind-v4",
  "Pochi-LP-ACRL-Blind-v5",
  "Pochi-LP-ACRL-Blind-v6",
  "Pochi-LP-ACRL-Blind-v7",
  "Pochi-LP-ACRL-Blind-v8",
  "Pochi-LP-ACRL-Blind-v9",
  "Pochi-LP-ACRL-Blind-v10",
  "Pochi-LP-ACRL-Blind-v11",
  "Pochi-LP-ACRL-Blind-v12",
  "Pochi-LP-ACRL-Blind-v13",
)
TASK = "Pochi-LP-ACRL-Blind-v1"


def resolve_task_id(task: str, task_id: int | None) -> int:
  """Keep rough task360; use the .2–.3m/s flat bin for V13."""
  flat = task == "Pochi-LP-ACRL-Blind-v13"
  selected = (2 if flat else 360) if task_id is None else task_id
  count = 5 if flat else 600
  if type(selected) is not int or not 0 <= selected < count:
    raise ValueError(f"--task-id must be between 0 and {count - 1} for {task}")
  return selected


def install_flat_command_gui(command) -> None:
  """Expose the trained forward/backward range without zero-width sliders."""
  def create_gui(term, name, server, get_env_idx, **_kwargs):
    import viser

    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Enable", initial_value=True)
      forward = server.gui.add_slider(
        "lin_vel_x", min=-0.5, max=0.5, step=0.05, initial_value=0.0
      )
      zero = server.gui.add_button("Zero", icon=viser.Icon.SQUARE_X)

      @zero.on_click
      def _(_event):
        forward.value = 0.0

    term._joystick_enabled = enabled
    term._joystick_sliders = [
      forward, SimpleNamespace(value=0.0), SimpleNamespace(value=0.0)
    ]
    term._joystick_get_env_idx = get_env_idx
    term.compute(0.0)

  command.create_gui = MethodType(create_gui, command)


def run(
  checkpoint: Path,
  *,
  task: str = TASK,
  task_id: int | None = None,
  device: str = "cuda:2",
  host: str = "0.0.0.0",
  port: int = 8080,
) -> None:
  """Load strictly, reset to the selected task, then run the standard viewer."""
  os.environ.setdefault("MUJOCO_GL", "egl")
  import viser
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends
  from mjlab.viewer import ViserPlayViewer

  import pochi_rl  # noqa: F401
  checkpoint = checkpoint.expanduser().resolve(strict=True)
  if task not in TASKS:
    raise ValueError(f"This height-scan-free viewer requires one of {TASKS}")
  task_id = resolve_task_id(task, task_id)
  if not 1 <= port <= 65535:
    raise ValueError("--port must be between 1 and 65535")
  if task == "Pochi-LP-ACRL-Blind-v2":
    from pochi_rl.mjlab.lp_acrl_blind_v2 import register_blind_v2

    register_blind_v2()
  elif task == "Pochi-LP-ACRL-Blind-v3":
    from pochi_rl.mjlab.lp_acrl_blind_v3 import register_blind_v3

    register_blind_v3()
  elif task == "Pochi-LP-ACRL-Blind-v4":
    from pochi_rl.mjlab.lp_acrl_blind_v4 import register_blind_v4
    from pochi_rl.mjlab.warp_contact_backend import install_contact_backend

    install_contact_backend()
    register_blind_v4()
  elif task == "Pochi-LP-ACRL-Blind-v5":
    from pochi_rl.mjlab.lp_acrl_blind_v5 import register_blind_v5
    from pochi_rl.mjlab.warp_contact_backend import install_contact_backend

    install_contact_backend()
    register_blind_v5()
  elif task == "Pochi-LP-ACRL-Blind-v6":
    from pochi_rl.mjlab.lp_acrl_blind_v6 import register_blind_v6
    from pochi_rl.mjlab.warp_contact_backend import install_contact_backend

    install_contact_backend()
    register_blind_v6()
  elif task == "Pochi-LP-ACRL-Blind-v7":
    from pochi_rl.mjlab.lp_acrl_blind_v7 import register_blind_v7
    from pochi_rl.mjlab.warp_contact_backend import install_contact_backend

    install_contact_backend()
    register_blind_v7()
  elif task == "Pochi-LP-ACRL-Blind-v8":
    from pochi_rl.mjlab.lp_acrl_blind_v8 import register_blind_v8
    from pochi_rl.mjlab.warp_contact_backend import install_contact_backend

    install_contact_backend()
    register_blind_v8()
  elif task == "Pochi-LP-ACRL-Blind-v9":
    from pochi_rl.mjlab.lp_acrl_blind_v9 import register_blind_v9
    from pochi_rl.mjlab.warp_contact_backend import install_contact_backend

    install_contact_backend()
    register_blind_v9()
  elif task == "Pochi-LP-ACRL-Blind-v10":
    from pochi_rl.mjlab.lp_acrl_blind_v10 import register_blind_v10
    from pochi_rl.mjlab.warp_contact_backend import install_contact_backend

    install_contact_backend()
    register_blind_v10()
  elif task == "Pochi-LP-ACRL-Blind-v11":
    from pochi_rl.mjlab.lp_acrl_blind_v11 import register_blind_v11
    from pochi_rl.mjlab.warp_contact_backend import install_contact_backend

    install_contact_backend()
    register_blind_v11()
  elif task == "Pochi-LP-ACRL-Blind-v12":
    from pochi_rl.mjlab.lp_acrl_blind_v12 import register_blind_v12
    from pochi_rl.mjlab.warp_contact_backend import install_contact_backend

    install_contact_backend()
    register_blind_v12()
  elif task == "Pochi-LP-ACRL-Blind-v13":
    from pochi_rl.mjlab.lp_acrl_blind_v13 import register_blind_v13
    from pochi_rl.mjlab.warp_contact_backend import install_contact_backend

    install_contact_backend()
    register_blind_v13()
  configure_torch_backends()
  env_cfg = load_env_cfg(task, play=True)
  agent_cfg = load_rl_cfg(task)
  env_cfg.scene.num_envs = 1
  env_cfg.commands["base_velocity"].manual_control_on_play = True
  runner_cls = load_runner_cls(task)
  if runner_cls is None:
    raise RuntimeError("LP-ACRL requires its registered checkpoint-validating runner")

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
    if task == "Pochi-LP-ACRL-Blind-v13":
      install_flat_command_gui(command)
    command.set_evaluation(task_ids=task_id, command=(0.0, 0.0, 0.0))
    env.reset()

    coordinate = command.catalog.task_indices()[task_id].tolist()
    terrain = command.catalog.terrains[coordinate[2]]
    print(f"[LP-ACRL] Checkpoint: {checkpoint}", flush=True)
    print(
      f"[LP-ACRL] Fixed task {task_id}: {terrain}, level {coordinate[3]}; "
      "initial command=(0,0,0), manual Enable=on",
      flush=True,
    )
    controls = "lin_vel_x (−0.5 to +0.5 m/s)" if task.endswith("-v13") else (
      "lin_vel_x/y, ang_vel_z"
    )
    print(
      f"[LP-ACRL] Controls > Commands > Base_velocity: {controls}; "
      "Zero stops. Reset Environment returns to the same terrain task.",
      flush=True,
    )
    server = viser.ViserServer(host=host, port=port, label="Pochi blind LP-ACRL")
    # Viser may silently pick another port if the requested one is occupied.
    # Never replace or stop another viewer, and report a conflicting port.
    if server.get_port() != port:
      raise RuntimeError(f"Requested port {port} is occupied; choose another --port")
    ViserPlayViewer(wrapped, policy, viser_server=server).run()
  finally:
    if server is not None:
      server.stop()
    env.close()


def parse_args(argv=None) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--task", choices=TASKS, default=TASK)
  parser.add_argument("--task-id", type=int, default=None)
  parser.add_argument("--device", default="cuda:2")
  parser.add_argument("--host", default="0.0.0.0")
  parser.add_argument("--port", type=int, default=8080)
  return parser.parse_args(argv)


def main() -> None:
  run(**vars(parse_args()))


if __name__ == "__main__":
  main()
