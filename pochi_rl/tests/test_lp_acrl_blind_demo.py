"""Command timing and terminal-state integrity of the single-take blind demo."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from pochi_rl.robot import FOOT_GEOMS, FOOT_SITES

_SPEC = importlib.util.spec_from_file_location(
  "blind_demo",
  Path(__file__).resolve().parents[1] / "scripts/record_lp_acrl_blind_demo.py",
)
demo = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(demo)
_VIEWER_SPEC = importlib.util.spec_from_file_location(
  "blind_viewer", Path(__file__).resolve().parents[1] / "scripts/play_lp_acrl.py"
)
viewer = importlib.util.module_from_spec(_VIEWER_SPEC)
_VIEWER_SPEC.loader.exec_module(viewer)


class _Scene(dict):
  env_origins = torch.tensor([[10.0, 20.0, 0.0]])


def _environment(*, fall_at=None, invalid_at=None, force_nan_at=None):
  data = SimpleNamespace(
    joint_pos=torch.zeros(1, 12),
    default_joint_pos=torch.zeros(1, 12),
    joint_vel=torch.zeros(1, 12),
    joint_acc=torch.zeros(1, 12),
    root_link_pos_w=torch.tensor([[10.0, 20.0, 0.3173]]),
    root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]),
    root_link_lin_vel_b=torch.zeros(1, 3),
    root_link_ang_vel_b=torch.zeros(1, 3),
    site_pos_w=torch.zeros(1, 4, 3),
    site_lin_vel_w=torch.zeros(1, 4, 3),
  )
  sensor = SimpleNamespace(
    primary_names=[f"robot/{name}" for name in FOOT_GEOMS],
    data=SimpleNamespace(found=torch.ones(1, 4), force=torch.zeros(1, 4, 3)),
    _invalidate_cache=lambda: None,
  )
  sensor.data.force[:, :, 2] = 25
  scene = _Scene(
    robot=SimpleNamespace(
      data=data, find_sites=lambda names, preserve_order: (list(range(4)), FOOT_SITES)
    ),
    feet_ground_contact=sensor,
    limb_ground_contact=SimpleNamespace(
      data=SimpleNamespace(found=torch.zeros(1, 1)), _invalidate_cache=lambda: None
    ),
  )
  term = SimpleNamespace(
    vel_command_b=torch.zeros(1, 3),
    vel_command_w=torch.zeros(1, 3),
    is_standing_env=torch.ones(1, dtype=torch.bool),
  )

  def set_evaluation(task_ids, command):
    assert task_ids == 360
    term.vel_command_b[:] = torch.tensor(command)

  term.set_evaluation = set_evaluation
  env = SimpleNamespace(
    cfg=SimpleNamespace(auto_reset=False, observations={"actor": None, "critic": None}),
    num_envs=1,
    device="cpu",
    step_dt=0.02,
    max_episode_length=1000,
    scene=scene,
    command_manager=SimpleNamespace(get_term=lambda _: term),
    reset_terminated=torch.zeros(1, dtype=torch.bool),
    reset_time_outs=torch.zeros(1, dtype=torch.bool),
    step_count=0,
    reset_count=0,
  )

  def reset():
    env.reset_count += 1

  env.reset = reset

  def observations(name, update_history):
    assert name in ("actor", "critic") and not update_history
    obs = torch.zeros(1, 48)
    obs[:, 9:12] = term.vel_command_b
    return obs

  env.observation_manager = SimpleNamespace(compute_group=observations)

  def step(actions):
    # The recorder must pass through the actual policy output unchanged.
    torch.testing.assert_close(actions, torch.full((1, 12), 0.125))
    env.step_count += 1
    data.root_link_lin_vel_b[:, :2] = term.vel_command_b[:, :2]
    data.root_link_pos_w[:, :2] += term.vel_command_b[:, :2] * env.step_dt
    env.reset_terminated[:] = env.step_count == fall_at
    env.reset_time_outs[:] = env.step_count == 1000
    if env.step_count == invalid_at:
      data.joint_vel[0, 0] = 201
    if env.step_count == force_nan_at:
      sensor.data.force[0, 0, 2] = float("nan")
    done = env.reset_terminated | env.reset_time_outs
    return None, torch.ones(1), done, {}

  def policy(obs):
    for name in ("actor", "critic"):
      torch.testing.assert_close(obs[name][:, 9:12], term.vel_command_b)
    return torch.full((1, 12), 0.125)

  frame = np.zeros((540, 960, 3), dtype=np.uint8)
  env.render = lambda: frame
  return env, SimpleNamespace(step=step), policy


def test_exact_profile_boundaries_and_fresh_commands_without_reset_mixing():
  env, wrapped, policy = _environment()
  result = demo.record_demo(env, wrapped, policy)
  assert result["complete"] and result["recorded_duration_s"] == 20
  assert env.reset_count == 1 and env.step_count == 1000
  records = result["records"]
  assert records[249]["command_vx_vy_wz"] == [0, 0, 0]
  assert records[250]["command_vx_vy_wz"] == [0.25, 0, 0]
  assert records[749]["command_vx_vy_wz"] == [0.25, 0, 0]
  assert records[750]["command_vx_vy_wz"] == [0, 0, 0]
  assert [row["valid_samples"] for row in result["segments"]] == [200, 450, 200]
  assert result["records"][-1]["truncated"]
  assert result["records"][-1]["valid"]
  json.dumps(result, allow_nan=False)


def test_fall_records_terminal_pose_and_frame_then_stops_before_any_reset():
  env, wrapped, policy = _environment(fall_at=3)
  frames = []
  result = demo.record_demo(
    env,
    wrapped,
    policy,
    writer=SimpleNamespace(append_data=lambda frame: frames.append(frame.shape)),
  )
  assert not result["complete"] and result["fall"] and not result["invalid"]
  assert result["stop_reason"] == "fall"
  assert env.step_count == 3 and env.reset_count == 1
  assert len(result["records"]) == 3 and not result["records"][-1]["valid"]
  assert result["video_frame_times_s"] == [0.04, 0.06]
  assert frames == [(540, 960, 3), (540, 960, 3)]


def test_huge_finite_state_stops_without_rendering_the_bad_terminal_state():
  env, wrapped, policy = _environment(invalid_at=3)
  frames = []
  result = demo.record_demo(
    env, wrapped, policy, writer=SimpleNamespace(append_data=lambda _: frames.append(1))
  )
  assert result["invalid"] and result["stop_reason"] == "invalid_state"
  assert result["failure_reasons"] == ["joint_speed"]
  assert env.step_count == 3 and env.reset_count == 1
  assert len(frames) == 1 and result["video_frame_times_s"] == [0.04]


def test_nonfinite_force_is_preserved_as_null_and_report_is_valid_json():
  env, wrapped, policy = _environment(force_nan_at=2)
  result = demo.record_demo(env, wrapped, policy)
  assert result["invalid"] and env.step_count == 2
  assert "nonfinite/foot_force" in result["failure_reasons"]
  assert result["records"][-1]["foot_vertical_load_abs_N"][0] is None
  json.dumps(result, allow_nan=False)


def test_bounded_smoke_is_explicitly_incomplete_and_does_not_reset():
  env, wrapped, policy = _environment()
  result = demo.record_demo(env, wrapped, policy, max_steps=4)
  assert result["stop_reason"] == "bounded_check" and not result["complete"]
  assert result["recorded_duration_s"] == 0.08 and env.reset_count == 1


def test_auto_reset_and_wrong_horizon_are_rejected():
  env, wrapped, policy = _environment()
  env.cfg.auto_reset = True
  with pytest.raises(ValueError, match="auto_reset=False"):
    demo.record_demo(env, wrapped, policy)
  env.cfg.auto_reset = False
  env.max_episode_length = 500
  with pytest.raises(ValueError, match="20s episode"):
    demo.record_demo(env, wrapped, policy)


@pytest.mark.parametrize(
  "task",
  [
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
  ],
)
def test_viewer_and_demo_cli_accept_all_blind_versions(task, tmp_path):
  checkpoint = tmp_path / "model.pt"
  checkpoint.touch()
  common = ["--checkpoint", str(checkpoint), "--task", task]
  assert viewer.parse_args(common).task == task
  assert demo.parse_args(common + ["--output", str(tmp_path / "take.mp4")]).task == task


def test_viewer_and_demo_default_to_v1_and_reject_height_scan_teacher(tmp_path):
  checkpoint = tmp_path / "model.pt"
  checkpoint.touch()
  common = ["--checkpoint", str(checkpoint)]
  record = common + ["--output", str(tmp_path / "take.mp4")]
  assert viewer.parse_args(common).task == "Pochi-LP-ACRL-Blind-v1"
  assert demo.parse_args(record).task == "Pochi-LP-ACRL-Blind-v1"
  for parse, args in ((viewer.parse_args, common), (demo.parse_args, record)):
    with pytest.raises(SystemExit) as exc:
      parse(args + ["--task", "Pochi-LP-ACRL-Rough-v0"])
    assert exc.value.code == 2


@pytest.mark.parametrize("invalid_force", [False, True])
def test_contact_caches_refresh_before_initial_faults_and_terminal_snapshots(
  invalid_force,
):
  env, wrapped, policy = _environment(fall_at=3)

  class CachedSensor:
    def __init__(self, raw, names=()):
      self.raw = raw
      self.primary_names = names
      self.cache = None
      self.invalidations = 0

    @property
    def data(self):
      if self.cache is None:
        self.cache = SimpleNamespace(
          **{key: value.clone() for key, value in vars(self.raw).items()}
        )
      return self.cache

    def _invalidate_cache(self):
      self.invalidations += 1
      self.cache = None

  old_feet = env.scene["feet_ground_contact"]
  feet = CachedSensor(old_feet.data, old_feet.primary_names)
  limb = CachedSensor(env.scene["limb_ground_contact"].data)
  env.scene["feet_ground_contact"] = feet
  env.scene["limb_ground_contact"] = limb
  # Prime deliberately stale pre-reset reads. Reset changes the simulator data.
  feet.raw.found.zero_()
  feet.raw.force.zero_()
  limb.raw.found.fill_(1)
  assert not feet.data.found.any() and limb.data.found.any()
  original_reset = env.reset

  def reset():
    original_reset()
    feet.raw.found.fill_(1)
    feet.raw.force[:, :, 2] = 25
    limb.raw.found.zero_()

  env.reset = reset
  original_step = wrapped.step

  def step(actions):
    result = original_step(actions)
    # Emulate a read during the last substep, then final-forward/sense updating
    # raw buffers. No fresh contact read is available until cache invalidation.
    assert feet.data is feet.cache and limb.data is limb.cache
    feet.raw.found.fill_(1)
    feet.raw.found[0, env.step_count - 1] = 0
    feet.raw.force[:, :, 2] = feet.raw.found * (10 + env.step_count)
    limb.raw.found.fill_(env.step_count == 2)
    if invalid_force and env.step_count == 2:
      feet.raw.force[0, 0, 2] = float("nan")
    return result

  wrapped.step = step
  result = demo.record_demo(env, wrapped, policy)
  expected_steps = 2 if invalid_force else 3
  assert env.step_count == expected_steps and env.reset_count == 1
  assert feet.invalidations == limb.invalidations == expected_steps + 1
  assert result["initial_state"]["contact"] == [True] * 4
  assert result["initial_state"]["foot_vertical_load_abs_N"] == [25] * 4
  assert not result["initial_state"]["nonfoot_contact"]
  for index, row in enumerate(result["records"]):
    assert row["contact"] == [foot != index for foot in range(4)]
    assert row["nonfoot_contact"] == (index == 1)
  if invalid_force:
    assert result["stop_reason"] == "invalid_state"
    assert result["failure_reasons"] == ["nonfinite/foot_force"]
    assert result["records"][-1]["foot_vertical_load_abs_N"][0] is None
  else:
    assert result["stop_reason"] == "fall"
    assert result["records"][-1]["foot_vertical_load_abs_N"] == [13, 13, 0, 13]
  assert result["protocol"] == "pochi-lp-acrl-blind-native-demo-v2"
  assert "No extra physics" in result["metric_conventions"]["contact_alignment"]
  json.dumps(result, allow_nan=False)


@pytest.mark.parametrize("entrypoint", ["viewer", "recorder"])
@pytest.mark.parametrize("version", [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
def test_optional_task_registration_occurs_only_for_selected_task_before_loading(
  entrypoint, version, monkeypatch, tmp_path,
):
  from mjlab.tasks import registry
  from mjlab.utils import torch as torch_utils

  from pochi_rl.mjlab import (
    lp_acrl_blind_v2,
    lp_acrl_blind_v3,
    lp_acrl_blind_v4,
    lp_acrl_blind_v5,
    lp_acrl_blind_v6,
    lp_acrl_blind_v7,
    lp_acrl_blind_v8,
    lp_acrl_blind_v9,
    lp_acrl_blind_v10,
    lp_acrl_blind_v11,
    lp_acrl_blind_v12,
    warp_contact_backend,
  )

  task = f"Pochi-LP-ACRL-Blind-v{version}"
  calls = []
  monkeypatch.setattr(lp_acrl_blind_v2, "register_blind_v2", lambda: calls.append(2))
  monkeypatch.setattr(lp_acrl_blind_v3, "register_blind_v3", lambda: calls.append(3))
  monkeypatch.setattr(lp_acrl_blind_v4, "register_blind_v4", lambda: calls.append(4))
  monkeypatch.setattr(lp_acrl_blind_v5, "register_blind_v5", lambda: calls.append(5))
  monkeypatch.setattr(lp_acrl_blind_v6, "register_blind_v6", lambda: calls.append(6))
  monkeypatch.setattr(lp_acrl_blind_v7, "register_blind_v7", lambda: calls.append(7))
  monkeypatch.setattr(lp_acrl_blind_v8, "register_blind_v8", lambda: calls.append(8))
  monkeypatch.setattr(lp_acrl_blind_v9, "register_blind_v9", lambda: calls.append(9))
  monkeypatch.setattr(lp_acrl_blind_v10, "register_blind_v10", lambda: calls.append(10))
  monkeypatch.setattr(lp_acrl_blind_v11, "register_blind_v11", lambda: calls.append(11))
  monkeypatch.setattr(lp_acrl_blind_v12, "register_blind_v12", lambda: calls.append(12))
  monkeypatch.setattr(
    warp_contact_backend, "install_contact_backend", lambda: calls.append("backend")
  )
  monkeypatch.setattr(torch_utils, "configure_torch_backends", lambda: None)
  monkeypatch.setenv("MUJOCO_GL", "egl")

  class ReachedEnvironmentLoader(Exception):
    pass

  def load_env_cfg(selected, *, play):
    assert selected == task and play
    expected = (
      ["backend", version] if version >= 4 else ([version] if version >= 2 else [])
    )
    assert calls == expected
    raise ReachedEnvironmentLoader

  monkeypatch.setattr(registry, "load_env_cfg", load_env_cfg)
  checkpoint = tmp_path / "model.pt"
  checkpoint.touch()
  with pytest.raises(ReachedEnvironmentLoader):
    if entrypoint == "viewer":
      viewer.run(checkpoint, task=task, device="cpu")
    else:
      args = SimpleNamespace(
        checkpoint=checkpoint, task=task, output=tmp_path / "take.mp4",
        device="cpu", seed=123,
      )
      monkeypatch.setattr(demo, "parse_args", lambda: args)
      demo.main()
