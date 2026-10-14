"""Classical Huber values, commanded body-frame yaw and manager scaling."""

from types import SimpleNamespace

import pytest
import torch
from mjlab.managers import RewardManager, RewardTermCfg, SceneEntityCfg

from pochi_rl.mjlab.lp_acrl_yaw_stability_rewards import yaw_tracking_huber_error


def _setup(actual, command=None, dtype=torch.float64):
  actual = torch.as_tensor(actual, dtype=dtype)
  count = len(actual)
  velocity = torch.zeros(count, 3, dtype=dtype)
  velocity[:, 2] = actual
  commands = torch.zeros_like(velocity)
  if command is not None:
    commands[:, 2] = torch.as_tensor(command, dtype=dtype)
  entity = SimpleNamespace(data=SimpleNamespace(root_link_ang_vel_b=velocity))
  env = SimpleNamespace(
    num_envs=count,
    device="cpu",
    step_dt=0.02,
    max_episode_length_s=20.0,
    scene={"robot": entity},
    command=commands,
    command_manager=SimpleNamespace(get_command=lambda _: commands),
  )
  return env, entity.data


def test_classical_values_zero_boundary_tail_and_both_signs():
  env, _ = _setup([0, 0.1, -0.1, 0.3, -0.3, 0.6, -0.6, 1.0])
  expected = torch.tensor(
    [0, 0.005, 0.005, 0.045, 0.045, 0.135, 0.135, 0.255],
    dtype=torch.float64,
  )
  torch.testing.assert_close(yaw_tracking_huber_error(env), expected)


@pytest.mark.parametrize("delta", [0.15, 0.3, 0.6])
def test_matches_independent_torch_classical_huber_and_not_smooth_l1(delta):
  env, data = _setup([-0.7, -0.2, 0, 0.1, 0.4, 1.2])
  expected = torch.nn.functional.huber_loss(
    data.root_link_ang_vel_b[:, 2],
    env.command[:, 2],
    reduction="none",
    delta=delta,
  )
  result = yaw_tracking_huber_error(env, delta=delta)
  torch.testing.assert_close(result, expected)
  smooth = torch.nn.functional.smooth_l1_loss(
    data.root_link_ang_vel_b[:, 2],
    env.command[:, 2],
    reduction="none",
    beta=delta,
  )
  torch.testing.assert_close(result, delta * smooth)
  assert not torch.equal(result, smooth)


def test_tracks_nonzero_command_instead_of_penalizing_all_turning():
  env, _ = _setup([0.8, -0.8, 0.6, -0.6], command=[0.8, -0.8, 0.8, -0.8])
  torch.testing.assert_close(
    yaw_tracking_huber_error(env),
    torch.tensor([0, 0, 0.02, 0.02], dtype=torch.float64),
  )


def test_body_frame_yaw_and_selected_asset_command_only():
  env, data = _setup([0.4, -0.4], command=[0.1, -0.1])
  data.root_link_ang_vel_b[:, :2] = 999
  data.root_link_ang_vel_w = torch.full((2, 3), float("nan"))
  data.root_com_ang_vel_b = torch.full((2, 3), float("nan"))
  env.command[:, :2] = float("nan")
  env.scene["selected"] = env.scene.pop("robot")
  env.scene["robot"] = SimpleNamespace(data=SimpleNamespace())
  calls = []

  def get_command(name):
    calls.append(name)
    return env.command

  env.command_manager.get_command = get_command
  result = yaw_tracking_huber_error(
    env, command_name="chosen_command", asset_cfg=SceneEntityCfg("selected")
  )
  torch.testing.assert_close(result, torch.full((2,), 0.045, dtype=torch.float64))
  assert calls == ["chosen_command"]


def test_gradients_continuous_at_delta_with_constant_nonzero_tail():
  actual = torch.tensor([-1.0, -0.3, -0.1, 0, 0.1, 0.3, 1.0], dtype=torch.float64)
  env, data = _setup(actual)
  data.root_link_ang_vel_b.requires_grad_()
  yaw_tracking_huber_error(env).sum().backward()
  gradient = data.root_link_ang_vel_b.grad
  torch.testing.assert_close(gradient[:, 2], actual.clamp(-0.3, 0.3))
  assert gradient[:, :2].eq(0).all()


def test_large_finite_error_no_exponential_underflow_or_tail_square_overflow():
  env, data = _setup([100.0, -100.0, 1e30, -1e30], dtype=torch.float32)
  data.root_link_ang_vel_b.requires_grad_()
  result = yaw_tracking_huber_error(env)
  assert torch.isfinite(result).all() and (result > 0).all()
  torch.testing.assert_close(result[:2], torch.full((2,), 29.955))
  torch.testing.assert_close(result[2:], torch.full((2,), 3e29))
  result.sum().backward()
  torch.testing.assert_close(
    data.root_link_ang_vel_b.grad[:, 2], torch.tensor([0.3, -0.3, 0.3, -0.3])
  )


def test_stateless_zero_command_still_penalizes_spin_and_never_mutates_inputs():
  env, data = _setup([0.4, -0.4])
  before_velocity = data.root_link_ang_vel_b.clone()
  before_command = env.command.clone()
  with torch.inference_mode():
    first = yaw_tracking_huber_error(env)
  torch.testing.assert_close(first, torch.full((2,), 0.075, dtype=torch.float64))
  env.command[0, 2] = 0.4
  second = yaw_tracking_huber_error(env)
  assert second[0] == 0 and second[1] == first[1]
  env.command.copy_(before_command)
  torch.testing.assert_close(yaw_tracking_huber_error(env), first)
  torch.testing.assert_close(data.root_link_ang_vel_b, before_velocity)


@pytest.mark.parametrize(
  "delta", [0, -0.3, float("nan"), float("inf"), -float("inf"), True, "0.3", None]
)
def test_invalid_delta_rejected(delta):
  env, _ = _setup([0.1])
  with pytest.raises(ValueError, match="finite and positive"):
    yaw_tracking_huber_error(env, delta=delta)


@pytest.mark.parametrize("dt", [0.01, 0.02, 0.04])
def test_reward_manager_applies_weight_and_dt_exactly_once(dt):
  env, _ = _setup([0.1, 0.3, 0.6], dtype=torch.float32)
  # This helper is stateless; only the manager's supplied dt must scale cost.
  env.step_dt = 123
  manager = RewardManager(
    {
      "yaw_tracking_huber": RewardTermCfg(
        func=yaw_tracking_huber_error, weight=-8, params={"delta": 0.3}
      )
    },
    env,
  )
  expected_rate = -8 * torch.tensor([0.005, 0.045, 0.135])
  torch.testing.assert_close(manager.compute(dt), expected_rate * dt)
  torch.testing.assert_close(manager._step_reward[:, 0], expected_rate)


def test_nonfinite_state_is_not_silently_hidden_from_physics_guard():
  env, _ = _setup([float("nan"), float("inf")])
  result = yaw_tracking_huber_error(env)
  assert not torch.isfinite(result).any()
