"""Run the latest V19 policy in Viser with held WASDQE controls on GPU3."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from dataclasses import asdict
from pathlib import Path
from types import MethodType

from pochi_rl.mjlab.viser_keyboard import HeldKeyController, ViserKeyboardGateway

ROOT = Path(__file__).resolve().parents[1]


def install_gui(command, controller):
  original_compute = command.compute

  def create_gui(term, name, server, get_env_idx, **_kwargs):
    with server.gui.add_folder("WASDQE 操縦"):
      server.gui.add_markdown(
        "W/S：前後、A/D：左右、Q/E：左/右旋回。\n\n"
        "押している間だけ動きます。同時押し可。Space/Escで停止。"
      )
      keyboard = server.gui.add_checkbox("キーボード操縦", initial_value=True)
      speeds = [
        server.gui.add_slider(
          label, min=step, max=maximum, step=step, initial_value=initial
        )
        for label, maximum, step, initial in (
          ("前後速度 m/s", 4.0, 0.05, 0.5),
          ("左右速度 m/s", 1.5, 0.05, 0.3),
          ("旋回速度 rad/s", 4.0, 0.05, 0.6),
        )
      ]
      sliders = [
        server.gui.add_slider(
          label, min=lo, max=hi, step=0.01, initial_value=0.0, disabled=True
        )
        for label, (lo, hi) in (
          ("lin_vel_x", term.cfg.ranges.lin_vel_x),
          ("lin_vel_y", term.cfg.ranges.lin_vel_y),
          ("ang_vel_z", term.cfg.ranges.ang_vel_z),
        )
      ]
      zero = server.gui.add_button("停止 / Zero")
      enabled = server.gui.add_checkbox("指令を有効にする", initial_value=True)

      @zero.on_click
      def _(_event):
        controller.stop()
        for slider in sliders:
          slider.value = 0.0

      @keyboard.on_update
      def _(_event):
        controller.stop()
        controller.enabled = keyboard.value
        for slider in sliders:
          slider.value = 0.0
          slider.disabled = keyboard.value

      @enabled.on_update
      def _(_event):
        controller.stop()
        for slider in sliders:
          slider.value = 0.0

    term._keyboard_checkbox, term._keyboard_speeds = keyboard, speeds
    term._joystick_enabled, term._joystick_sliders = enabled, sliders
    term._joystick_get_env_idx = get_env_idx
    with controller.lock:
      controller.telemetry["ready"] = True
    term.compute(0.0)

  def compute(term, dt):
    if term._joystick_sliders:
      controller.enabled = term._keyboard_checkbox.value
      controller.speeds = tuple(slider.value for slider in term._keyboard_speeds)
      if controller.enabled:
        target = controller.command() if term._joystick_enabled.value else (0.0,) * 3
        for slider, value in zip(term._joystick_sliders, target, strict=True):
          if slider.value != value:
            slider.value = value
    original_compute(dt)
    if term._joystick_enabled is not None and not term._joystick_enabled.value:
      term.vel_command_b[:] = 0.0
      term.vel_command_w[:] = 0.0
      term.is_standing_env[:] = True

  command.create_gui = MethodType(create_gui, command)
  command.compute = MethodType(compute, command)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--host", default="0.0.0.0")
  parser.add_argument("--port", type=int, default=8084)
  parser.add_argument("--backend-port", type=int, default=8085)
  args = parser.parse_args()
  if not (1 <= args.port <= 65535 and 1 <= args.backend_port <= 65535):
    parser.error("Ports must be between 1 and 65535")
  if args.port == args.backend_port:
    parser.error("Public and loopback ports must differ")
  os.environ.update(
    CUDA_VISIBLE_DEVICES="3",
    MUJOCO_GL="egl",
    OMP_NUM_THREADS="4",
    OPENBLAS_NUM_THREADS="1",
    TMPDIR="/dev/shm/pochi_lp_acrl_compile_tmp",
    WARP_CACHE_PATH="/dev/shm/pochi_lp_acrl_warp_cache",
  )
  gpu_uuid = subprocess.check_output(
    ["nvidia-smi", "--id=3", "--query-gpu=uuid", "--format=csv,noheader"], text=True
  ).strip()
  if gpu_uuid != "GPU-ca672c85-e00a-6d95-ae2b-d37b117b9cd2":
    raise ValueError("Physical GPU3 UUID changed")

  import torch
  import viser
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
  from mjlab.utils.torch import configure_torch_backends
  from mjlab.viewer import ViserPlayViewer

  from pochi_rl.mjlab.lp_acrl_blind_v19 import TASK_ID, register_blind_v19
  from pochi_rl.mjlab.lp_acrl_speed_runner import (
    SpeedPhaseGaitRunner,
    validate_speed_checkpoint,
  )
  from pochi_rl.mjlab.warp_contact_backend import install_contact_backend

  checkpoint = args.checkpoint.resolve(strict=True)
  pins = json.loads((checkpoint.parent / "run.json").read_text())["source_sha256"]
  for name, digest in pins.items():
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
      raise ValueError(f"Frozen training source changed: {name}")
  count = validate_speed_checkpoint(
    torch.load(checkpoint, map_location="cpu", weights_only=False)
  )
  install_contact_backend()
  register_blind_v19()
  configure_torch_backends()
  cfg, agent = load_env_cfg(TASK_ID, play=True), load_rl_cfg(TASK_ID)
  cfg.scene.num_envs = 1
  cfg.commands["base_velocity"].manual_control_on_play = True
  cfg.commands["base_velocity"].debug_vis = False
  cfg.viewer.distance, cfg.viewer.azimuth, cfg.viewer.elevation = 1.8, 130, 24
  env = ManagerBasedRlEnv(cfg, device="cuda:0")
  server, gateway = None, None
  try:
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = SpeedPhaseGaitRunner(wrapped, asdict(agent), device="cuda:0")
    runner.load(
      checkpoint, load_cfg={"actor": True}, strict=True, map_location="cuda:0"
    )
    actor = runner.get_inference_policy(device="cuda:0")
    command = env.command_manager.get_term("base_velocity")
    controller = HeldKeyController()
    install_gui(command, controller)
    command.set_evaluation(task_ids=0, command=(0.0, 0.0, 0.0))
    env.reset()
    server = viser.ViserServer(
      host="127.0.0.1",
      port=args.backend_port,
      label=f"Pochi V19 / {count}更新 / WASDQE",
    )
    if server.get_port() != args.backend_port:
      raise RuntimeError("Loopback port occupied")
    gateway = ViserKeyboardGateway(
      controller,
      host=args.host,
      port=args.port,
      backend_port=args.backend_port,
      javascript=ROOT / "scripts/pochi_keyboard.js",
    )
    gateway.start()

    @torch.inference_mode()
    def policy(_observations):
      from tensordict import TensorDict

      command.compute(0.0)
      observations = TensorDict(
        {
          name: env.observation_manager.compute_group(name, update_history=False)
          for name in env.cfg.observations
        },
        batch_size=[env.num_envs],
      )
      robot = env.scene["robot"].data
      with controller.lock:
        controller.telemetry.update(
          command=command.vel_command_b[0].tolist(),
          actual=[
            *robot.root_link_lin_vel_b[0, :2].tolist(),
            robot.root_link_ang_vel_b[0, 2].item(),
          ],
          position=robot.root_link_pos_w[0].tolist(),
          control_step=env.common_step_counter,
        )
      return actor(observations)

    print(
      json.dumps(
        {
          "task": TASK_ID,
          "checkpoint": str(checkpoint),
          "completed_updates": count,
          "physical_gpu": 3,
          "port": args.port,
          "initial_command": [0, 0, 0],
        }
      ),
      flush=True,
    )
    ViserPlayViewer(wrapped, policy, frame_rate=30, viser_server=server).run()
  finally:
    if gateway is not None:
      gateway.stop()
    if server is not None:
      server.stop()
    env.close()


if __name__ == "__main__":
  main()
