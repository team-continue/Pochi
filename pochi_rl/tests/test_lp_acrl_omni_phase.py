from types import SimpleNamespace

import torch

from pochi_rl.mjlab.lp_acrl_omni_commands import (
  COVERAGE_FRACTION,
  MAX_COMMAND,
  OMNI_CATALOG,
  CoverageLPACRLCurriculum,
  OmniPhaseVelocityCommand,
  planar_tracking_huber_error,
)
from pochi_rl.mjlab.lp_acrl_omni_runner import initialize_new_axis_inputs
from pochi_rl.mjlab.lp_acrl_phase_gait import moving


def test_catalog_covers_pure_axes_both_signs_and_full_ranges():
  assert OMNI_CATALOG.num_tasks == 88
  bounds = OMNI_CATALOG.task_bounds()
  assert torch.allclose(bounds.abs().amax((0, 2)), torch.tensor(MAX_COMMAND))
  for name, signs in OMNI_CATALOG.directions():
    if name in ("forward", "lateral", "yaw"):
      assert sum(x != 0 for x in signs) == 1
  reference = OMNI_CATALOG.coverage_probabilities()
  assert torch.allclose(reference.sum(), torch.tensor(1.0, dtype=reference.dtype))
  for name in ("forward", "lateral", "yaw", "diagonal", "arc", "omni"):
    ids = [
      i * 4 + band
      for i, (mode, _) in enumerate(OMNI_CATALOG.directions())
      if mode == name
      for band in range(4)
    ]
    assert torch.allclose(reference[ids].sum(), reference.new_tensor(1 / 6))


def test_lp_coverage_survives_softmax_collapse_and_state_restore():
  lp = CoverageLPACRLCurriculum(88)
  lp.learning_progress[0] = 10000
  lp.probabilities.copy_(torch.softmax(lp.learning_progress / lp.beta, 0))
  effective = lp.effective_probabilities()
  assert torch.all(
    effective >= COVERAGE_FRACTION * OMNI_CATALOG.coverage_probabilities()
  )
  restored = CoverageLPACRLCurriculum(88)
  restored.load_state_dict(lp.state_dict())
  assert torch.equal(restored.effective_probabilities(), effective)
  torch.manual_seed(18)
  sampled = lp.sample(10000)
  assert all(
    bool(((sampled >= i * 4) & (sampled < (i + 1) * 4)).any()) for i in range(22)
  )


def test_resampling_keeps_pure_axes_and_zero_stop_distinct():
  cmd = object.__new__(OmniPhaseVelocityCommand)
  cmd._env = SimpleNamespace(device="cpu", num_envs=88)
  cmd._task_bounds = OMNI_CATALOG.task_bounds()
  cmd.task_ids = torch.arange(88)
  cmd.episode_mode = torch.zeros(88, dtype=torch.long)
  cmd.episode_mode[0] = 1
  cmd.cfg = SimpleNamespace(adaptive=True, transition_interval_s=(4.0, 6.0))
  cmd.vel_command_b = torch.zeros(88, 3)
  cmd.vel_command_w = torch.zeros_like(cmd.vel_command_b)
  cmd._movement_command = torch.zeros_like(cmd.vel_command_b)
  cmd.is_standing_env = torch.zeros(88, dtype=torch.bool)
  cmd._transition_time_left = torch.zeros(88)
  cmd._forced_command = None
  cmd.phase_cycles = torch.zeros(88)
  cmd._resample_command(torch.arange(88))
  assert not moving(cmd.vel_command_b[:1]).any()
  assert cmd.phase_cycles[0] == 0
  assert (cmd.vel_command_b[8:16, 0] == 0).all()
  assert (cmd.vel_command_b[8:16, 2] == 0).all()
  assert moving(cmd.vel_command_b[8:24]).all()
  assert (cmd.vel_command_b[16:24, :2] == 0).all()
  assert (cmd.vel_command_b.abs() <= torch.tensor(MAX_COMMAND)).all()


def test_new_axis_initialization_preserves_zero_axis_hidden_activation():
  torch.manual_seed(18)
  source = {
    "mlp.0.weight": torch.randn(256, 50),
    "mlp.0.bias": torch.randn(256),
    "obs_normalizer._mean": torch.randn(1, 50) * 0.03,
    "obs_normalizer._std": torch.rand(1, 50) + 0.2,
    "obs_normalizer._var": torch.rand(1, 50),
    "obs_normalizer.count": torch.tensor(400000000),
  }
  x = torch.randn(16, 50)
  x[:, 10:12] = 0
  target = initialize_new_axis_inputs(source)

  def latent(state):
    normalized = (x - state["obs_normalizer._mean"]) / (
      state["obs_normalizer._std"] + 0.01
    )
    return normalized @ state["mlp.0.weight"].T + state["mlp.0.bias"]

  assert torch.allclose(latent(source), latent(target), atol=2e-5, rtol=1e-5)
  assert not target["mlp.0.weight"][:, 10:12].count_nonzero()
  assert torch.equal(target["mlp.0.weight"][:, 48:], source["mlp.0.weight"][:, 48:])
  assert torch.equal(target["obs_normalizer.count"], source["obs_normalizer.count"])
  assert torch.equal(target["mlp.0.weight"][:, :10], source["mlp.0.weight"][:, :10])
  assert source["mlp.0.weight"][:, 10:12].count_nonzero()  # parent remains intact


def test_planar_cost_tracks_nonzero_lateral_instead_of_penalizing_it():
  commands = torch.tensor([[0.0, 0.5, 0.0], [0.0, 0.5, 0.0]])
  env = SimpleNamespace(
    command_manager=SimpleNamespace(get_command=lambda _: commands),
    scene={
      "robot": SimpleNamespace(
        data=SimpleNamespace(
          root_link_lin_vel_b=torch.tensor([[0.0, 0.5, 0.0], [0.0, 0.0, 0.0]])
        )
      )
    },
  )
  costs = planar_tracking_huber_error(env)
  assert costs[0] == 0 and costs[1] > 0


def test_omni_evaluator_grades_pure_yaw_as_moving_and_detects_nontracking(monkeypatch):
  import importlib.util
  from pathlib import Path

  import numpy as np

  from pochi_rl.mjlab.lp_acrl_phase_gait import foot_targets

  folder = Path(__file__).resolve().parents[1] / "scripts"
  monkeypatch.syspath_prepend(str(folder))
  spec = importlib.util.spec_from_file_location(
    "eval_omni_under_test", folder / "eval_lp_acrl_omni_phase.py"
  )
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  t = np.arange(1, 1001) * 0.02
  height, contact = foot_targets(torch.tensor(t * 1.5, dtype=torch.float32) % 1)
  data = {
    "time_s": t,
    "valid": np.ones((1000, 1), bool),
    "velocity": np.tile([0.0, 0.0, 0.6], (1000, 1, 1)),
    "position": np.tile([0.0, 0.0, 0.317], (1000, 1, 1)),
    "contact": contact.numpy()[:, None],
    "foot_z": height.numpy()[:, None],
    "foot_vertical_force": np.full((1000, 1, 4), 27.0),
    "nonfoot_contact": np.zeros((1000, 1), bool),
    "slip": np.zeros((1000, 1)),
    "tilt_deg": np.zeros((1000, 1)),
    "joint_error": np.zeros((1000, 1, 12)),
    "joint_velocity": np.zeros((1000, 1, 12)),
  }
  result = module.summarize_omni_segment(data, 0, 0, 20, (0.0, 0.0, 0.6), 0.02, 2.0)
  assert result["passed"] and "all_four_contact" not in result["checks"]
  data["velocity"][:] = 0
  failed = module.summarize_omni_segment(data, 0, 0, 20, (0.0, 0.0, 0.6), 0.02, 2.0)
  assert not failed["passed"] and not failed["checks"]["each_axis_command_error"]
  assert len(module.PROFILES) == 36
  assert module.profile_command("stop_start_yaw", 5.0) == (0.0, 0.0, 0.6)
  assert module.profile_command("stop_start_yaw", 10.0) == (0.0, 0.0, 0.0)
