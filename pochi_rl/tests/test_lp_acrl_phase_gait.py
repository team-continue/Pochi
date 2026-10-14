from types import SimpleNamespace

import torch

from pochi_rl.mjlab.lp_acrl_phase_gait import (
  FREQUENCY_HZ,
  foot_targets,
  gait_phase,
  mirror_phase_observations,
  phase_foot_contact_error,
  phase_foot_lift_error,
)
from pochi_rl.mjlab.lp_acrl_phase_runner import expand_body_model
from pochi_rl.robot import FOOT_GEOMS


def test_diagonal_targets_frequency_and_support():
  # Sample cell centres, avoiding exact discontinuous contact boundaries.
  cycles = (torch.arange(1000) + 0.5) / 1000
  height, contact = foot_targets(cycles)
  assert FREQUENCY_HZ == 1.5
  assert torch.equal(height[:, 0], height[:, 3])
  assert torch.equal(height[:, 1], height[:, 2])
  assert (contact.sum(-1) >= 2).all()
  assert torch.allclose(height.max(0).values, torch.full((4,), 0.05))
  _, reflected = foot_targets((cycles + 0.5) % 1)
  assert torch.equal(reflected, contact[:, [1, 0, 3, 2]])


def test_phase_reflection_is_involution_and_negates_both_components():
  x = torch.randn(8, 50)
  y = mirror_phase_observations(x)
  assert torch.equal(y[:, 48:], -x[:, 48:])
  assert torch.equal(mirror_phase_observations(y), x)


def test_stop_phase_is_zero_and_pure_yaw_is_moving():
  term = SimpleNamespace(
    phase_cycles=torch.tensor([0.25, 0.25]),
    vel_command_b=torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.6]]),
  )
  env = SimpleNamespace(command_manager=SimpleNamespace(get_term=lambda _: term))
  phase = gait_phase(env)
  assert torch.equal(phase[0], torch.zeros(2))
  assert torch.allclose(phase[1], torch.tensor([1.0, 0.0]), atol=1e-6)


def test_lift_reference_is_last_contact_and_continuous_cost_is_nonpositive():
  term = SimpleNamespace(
    phase_cycles=torch.tensor([0.17]), vel_command_b=torch.tensor([[0.25, 0.0, 0.0]])
  )
  sensor = SimpleNamespace(
    primary_names=FOOT_GEOMS, data=SimpleNamespace(found=torch.ones(1, 4))
  )
  z = torch.zeros(1, 4, 3)
  z[:, :, 2] = 0.2
  env = SimpleNamespace(
    num_envs=1,
    device="cpu",
    step_dt=0.02,
    command_manager=SimpleNamespace(get_term=lambda _: term),
    scene={
      "feet_ground_contact": sensor,
      "robot": SimpleNamespace(data=SimpleNamespace(site_pos_w=z)),
    },
  )
  asset = SimpleNamespace(name="robot", site_ids=slice(None))
  cost = phase_foot_lift_error(None, env)
  assert cost(env, asset) > 0
  sensor.data.found[:] = torch.tensor([[0.0, 1.0, 1.0, 0.0]])
  z[:, [0, 3], 2] = 0.25
  assert torch.allclose(cost(env, asset), torch.zeros(1), atol=1e-10)
  assert torch.equal(phase_foot_contact_error(env), torch.zeros(1))
  term.vel_command_b.zero_()
  assert torch.equal(cost(env, asset), torch.zeros(1))


def test_input_expansion_preserves_all_body_tensors_and_adds_zero_weights():
  source = {
    "mlp.0.weight": torch.randn(256, 48),
    "obs_normalizer._mean": torch.randn(1, 48),
    "obs_normalizer._var": torch.rand(1, 48),
    "obs_normalizer._std": torch.rand(1, 48),
    "obs_normalizer.count": torch.tensor(100),
    "mlp.0.bias": torch.randn(256),
  }
  result = expand_body_model(source)
  assert result["mlp.0.weight"].shape == (256, 50)
  assert torch.equal(result["mlp.0.weight"][:, :48], source["mlp.0.weight"])
  assert not result["mlp.0.weight"][:, 48:].count_nonzero()
  assert torch.equal(result["obs_normalizer.count"], source["obs_normalizer.count"])
  assert torch.equal(
    result["obs_normalizer._mean"][:, :48], source["obs_normalizer._mean"]
  )
  assert torch.equal(result["obs_normalizer._std"][:, 48:], torch.ones(1, 2))
