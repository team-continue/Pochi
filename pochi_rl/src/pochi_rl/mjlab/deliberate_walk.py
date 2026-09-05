"""Slow, sequential footfalls, in FL, FR, RL, RR order.

The clock is an observation as well as a reward target: velocity alone does
not tell a stationary policy which foot should swing next.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg

from pochi_rl.robot.pochi_constants import FOOT_SITES


@dataclass(frozen=True)
class DeliberateWalkSpec:
  period_s: float = 2.8
  swing_fraction: float = 0.20
  clearance_m: float = 0.045
  command_threshold: float = 0.025
  # RL -> FL -> RR -> FR, with four-foot support between steps.
  offsets: tuple[float, ...] = (0.25, 0.75, 0.0, 0.5)


SPEC = DeliberateWalkSpec()
FEET = SceneEntityCfg("robot", site_names=FOOT_SITES, preserve_order=True)


def gait_reference(time: torch.Tensor, moving: torch.Tensor):
  """Per-foot swing mask and smooth ground-relative lift [m]."""
  offsets = time.new_tensor(SPEC.offsets)
  phase = (time[:, None] / SPEC.period_s - offsets).remainder(1.0)
  swing = (phase < SPEC.swing_fraction) & moving[:, None]
  u = (phase / SPEC.swing_fraction).clamp(0.0, 1.0)
  lift = SPEC.clearance_m * torch.sin(torch.pi * u).square()
  return swing, lift * swing


def _clock(env):
  command = env.command_manager.get_command("base_velocity")
  moving = command[:, :2].norm(dim=1) + command[:, 2].abs()
  return (
    env.episode_length_buf.float() * env.step_dt,
    moving > SPEC.command_threshold,
  )


def phase_observation(env):
  time, moving = _clock(env)
  angle = 2.0 * torch.pi * time / SPEC.period_s
  return torch.stack((angle.sin(), angle.cos()), dim=-1) * moving[:, None]


def contact_schedule(env, sensor_name: str):
  """Penalize missed support and missing swing equally, per foot."""
  swing, _ = gait_reference(*_clock(env))
  contact = env.scene[sensor_name].data.found > 0
  return (contact != ~swing).float().sum(dim=-1)


def insufficient_support(env, sensor_name: str):
  contact = env.scene[sensor_name].data.found > 0
  return (3 - contact.sum(dim=-1)).clamp(min=0).float().square()


def foot_height_error(env, asset_cfg: SceneEntityCfg = FEET):
  """Flat terrain only; foot sites are at the nominal sole height."""
  _, lift = gait_reference(*_clock(env))
  height = env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids, 2]
  height = height - env.scene.env_origins[:, None, 2]
  return (height - lift).abs().sum(dim=-1)


class DeliberateVelocityCommand(UniformVelocityCommand):
  """The normal velocity command with controls suited to centimetres/second.

  Upstream sliders assume ranges of at least 0.1 and reject fixed zero axes.
  Keep the trained ranges and the fixed forward demo, with finer GUI controls.
  """

  def create_gui(self, name, server, get_env_idx, on_change=None, request_action=None):
    del on_change, request_action
    with server.gui.add_folder(name.capitalize()):
      enabled = server.gui.add_checkbox("Manual control", initial_value=False)
      axes = (
        ("Forward [m/s]", 0.0, 0.14, 0.01, 0.1),
        ("Sideways [m/s]", -0.025, 0.025, 0.005, 0.0),
        ("Turn [rad/s]", -0.15, 0.15, 0.01, 0.0),
      )
      sliders = [
        server.gui.add_slider(label, min=lo, max=hi, step=step, initial_value=value)
        for label, lo, hi, step, value in axes
      ]
      stop = server.gui.add_button("Stop")
      walk = server.gui.add_button("Walk slowly")

      @stop.on_click
      def _(_event):
        for slider in sliders:
          slider.value = 0.0
        enabled.value = True

      @walk.on_click
      def _(_event):
        for slider, value in zip(sliders, (0.1, 0.0, 0.0), strict=True):
          slider.value = value
        enabled.value = True

    self._joystick_enabled = enabled
    self._joystick_sliders = sliders
    self._joystick_get_env_idx = get_env_idx


@dataclass(kw_only=True)
class DeliberateVelocityCommandCfg(UniformVelocityCommandCfg):
  def build(self, env):
    return DeliberateVelocityCommand(self, env)
