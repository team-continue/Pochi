"""Flat playback uses its actual catalog and only trained velocity controls."""

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
from test_lp_acrl_blind_demo import viewer


def test_flat_and_rough_default_task_ids_are_different_catalogs():
  assert viewer.resolve_task_id("Pochi-LP-ACRL-Blind-v13", None) == 2
  assert viewer.resolve_task_id("Pochi-LP-ACRL-Blind-v12", None) == 360
  assert viewer.resolve_task_id("Pochi-LP-ACRL-Blind-v13", 4) == 4
  for invalid in (360, -1, 5, True):
    with pytest.raises(ValueError):
      viewer.resolve_task_id("Pochi-LP-ACRL-Blind-v13", invalid)


def test_flat_gui_starts_stopped_and_exposes_only_forward_backward_velocity():
  widgets = {}
  callbacks = {}

  def widget(label, **kwargs):
    result = SimpleNamespace(value=kwargs.get("initial_value"), **kwargs)
    result.on_click = lambda callback: callbacks.update({label: callback})
    widgets[label] = result
    return result

  server = SimpleNamespace(gui=SimpleNamespace(
    add_folder=lambda name: nullcontext(), add_checkbox=widget,
    add_slider=widget, add_button=widget,
  ))
  observed = []
  command = SimpleNamespace()
  command.compute = lambda dt: observed.append(
    (dt, [slider.value for slider in command._joystick_sliders])
  )
  viewer.install_flat_command_gui(command)
  command.create_gui("base_velocity", server, lambda: 0)
  assert observed == [(0.0, [0.0, 0.0, 0.0])]
  assert widgets["Enable"].value is True
  assert set(widgets) == {"Enable", "lin_vel_x", "Zero"}
  assert (widgets["lin_vel_x"].min, widgets["lin_vel_x"].max) == (-0.5, 0.5)
  widgets["lin_vel_x"].value = 0.25
  callbacks["Zero"](None)
  assert widgets["lin_vel_x"].value == 0.0


def test_flat_cli_registration_precedes_environment_creation(monkeypatch, tmp_path):
  from mjlab.tasks import registry
  from mjlab.utils import torch as torch_utils

  from pochi_rl.mjlab import lp_acrl_blind_v13, warp_contact_backend

  calls = []
  monkeypatch.setenv("MUJOCO_GL", "egl")
  monkeypatch.setattr(
    warp_contact_backend, "install_contact_backend", lambda: calls.append("backend")
  )
  monkeypatch.setattr(
    lp_acrl_blind_v13, "register_blind_v13", lambda: calls.append("v13")
  )
  monkeypatch.setattr(torch_utils, "configure_torch_backends", lambda: None)

  class ReachedEnvironment(Exception):
    pass

  def load_env(task, *, play):
    assert task == "Pochi-LP-ACRL-Blind-v13" and play
    assert calls == ["backend", "v13"]
    raise ReachedEnvironment

  monkeypatch.setattr(registry, "load_env_cfg", load_env)
  path = tmp_path / "model.pt"
  path.touch()
  args = viewer.parse_args(
    ["--checkpoint", str(path), "--task", "Pochi-LP-ACRL-Blind-v13"]
  )
  assert args.task_id is None
  with pytest.raises(ReachedEnvironment):
    viewer.run(path, task=args.task, device="cpu")
