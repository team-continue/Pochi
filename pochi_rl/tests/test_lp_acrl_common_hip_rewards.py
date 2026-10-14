"""Common-mode algebra, named indexing, standing gate and manager integration."""

from types import SimpleNamespace

import pytest
import torch
from mjlab.managers import RewardManager, RewardTermCfg, SceneEntityCfg

from pochi_rl.mjlab.lp_acrl_common_hip_rewards import (
  HIP_PITCH_NAMES,
  moving_common_hip_pitch_deviation,
)
from pochi_rl.robot import DEFAULT_JOINT_POS, JOINT_NAMES


def _setup(count=3, names=JOINT_NAMES):
  command = torch.zeros(count, 3)
  command[:, 0] = 0.25
  names = list(names)
  ids = [names.index(name) for name in HIP_PITCH_NAMES]
  default = torch.tensor([DEFAULT_JOINT_POS[name] for name in names])
  data = SimpleNamespace(
    joint_pos=default.repeat(count, 1), default_joint_pos=default.repeat(count, 1)
  )
  entity = SimpleNamespace(
    data=data,
    joint_names=names,
    num_joints=len(names),
    find_joints=lambda requested, preserve_order: (
      [names.index(name) for name in requested],
      list(requested),
    ),
  )
  env = SimpleNamespace(
    num_envs=count,
    device="cpu",
    step_dt=0.02,
    max_episode_length_s=20,
    scene={"robot": entity},
    command=command,
    command_manager=SimpleNamespace(get_command=lambda _: command),
  )
  asset = SceneEntityCfg(
    "robot", joint_names=HIP_PITCH_NAMES, joint_ids=ids, preserve_order=True
  )
  return env, data, asset


def test_default_and_common_offset_are_square_mean_not_mean_square():
  env, data, asset = _setup(4)
  offsets = torch.tensor(
    [
      [0.0, 0.0, 0.0, 0.0],
      [0.37, 0.37, 0.37, 0.37],
      [0.4, 0.0, 0.0, 0.0],
      [0.2, -0.2, 0.3, -0.3],
    ]
  )
  data.joint_pos[:, asset.joint_ids] += offsets
  torch.testing.assert_close(
    moving_common_hip_pitch_deviation(env, asset),
    torch.tensor([0.0, 0.37**2, 0.1**2, 0.0]),
  )


def test_named_selection_works_when_robot_joint_storage_is_permuted():
  names = tuple(reversed(JOINT_NAMES))
  env, data, asset = _setup(2, names)
  data.joint_pos[:, asset.joint_ids] -= 0.25
  # A large unselected knee or roll offset must never contaminate the mean.
  other = [i for i in range(12) if i not in asset.joint_ids]
  data.joint_pos[:, other] += 1.0
  torch.testing.assert_close(
    moving_common_hip_pitch_deviation(env, asset), torch.full((2,), 0.25**2)
  )


def test_uses_per_world_defaults_before_averaging_and_allows_broadcast():
  env, data, asset = _setup(2)
  data.default_joint_pos[1, asset.joint_ids] += 0.3
  data.joint_pos.copy_(data.default_joint_pos)
  assert moving_common_hip_pitch_deviation(env, asset).eq(0).all()
  data.default_joint_pos = data.default_joint_pos[:1]
  torch.testing.assert_close(
    moving_common_hip_pitch_deviation(env, asset), torch.tensor([0.0, 0.09])
  )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_shared_exact_stop_gate_and_all_command_axes(axis):
  env, data, asset = _setup(4)
  data.joint_pos[:, asset.joint_ids] += 0.2
  env.command.zero_()
  env.command[1, axis] = 1e-6
  env.command[2, axis] = 2e-6
  env.command[3, axis] = -2e-6
  torch.testing.assert_close(
    moving_common_hip_pitch_deviation(env, asset),
    torch.tensor([0.0, 0.0, 0.04, 0.04]),
  )


def test_left_right_permutation_and_offset_sign_have_equal_cost():
  env, data, asset = _setup(3)
  residual = torch.tensor([0.2, 0.3, -0.1, 0.4])
  data.joint_pos[:, asset.joint_ids] += torch.stack(
    (
      residual,
      residual[[1, 0, 3, 2]],
      -residual,
    )
  )
  cost = moving_common_hip_pitch_deviation(env, asset)
  torch.testing.assert_close(cost, torch.full((3,), 0.2**2))


def test_current_positions_only_and_no_state_leaks_across_stops_or_subset_reset():
  env, data, asset = _setup(3)
  data.joint_pos[:, asset.joint_ids] += 0.2
  before = data.joint_pos.clone()
  with torch.inference_mode():
    initial = moving_common_hip_pitch_deviation(env, asset)
  torch.testing.assert_close(data.joint_pos, before)
  env.command[0].zero_()
  data.joint_pos[1].copy_(data.default_joint_pos[1])
  env.action_manager = SimpleNamespace(action=torch.full((3, 12), 4.0))
  data.joint_vel = torch.full((3, 12), 100.0)
  torch.testing.assert_close(
    moving_common_hip_pitch_deviation(env, asset), torch.tensor([0.0, 0.0, 0.04])
  )
  env.command[0, 0] = -0.25
  assert moving_common_hip_pitch_deviation(env, asset)[0] == initial[0]


def test_analytic_gradient_has_four_equal_hip_components_and_no_other_joints():
  env, data, asset = _setup(1)
  data.joint_pos[:, asset.joint_ids] -= 0.3
  data.joint_pos.requires_grad_()
  cost = moving_common_hip_pitch_deviation(env, asset)
  cost.sum().backward()
  expected = torch.zeros_like(data.joint_pos)
  expected[:, asset.joint_ids] = -0.15  # d(mean4 residual)^2 / dq_i = mean/2.
  torch.testing.assert_close(data.joint_pos.grad, expected)


def test_reward_manager_resolves_names_and_applies_weight_dt_once():
  env, data, asset = _setup(1)
  data.joint_pos[:, asset.joint_ids] -= 0.37
  asset.joint_ids = slice(None)
  manager = RewardManager(
    {
      "common_hip": RewardTermCfg(
        func=moving_common_hip_pitch_deviation,
        weight=-4,
        params={"asset_cfg": asset},
      )
    },
    env,
  )
  torch.testing.assert_close(manager.compute(0.02), torch.tensor([-4 * 0.37**2 * 0.02]))


@pytest.mark.parametrize(
  "fault",
  [
    "names",
    "duplicate",
    "unordered",
    "unresolved",
    "resolved_order",
    "outside",
    "position_shape",
    "default_shape",
  ],
)
def test_incorrect_joint_selection_or_state_shape_rejected(fault):
  env, data, asset = _setup()
  if fault == "names":
    asset.joint_names = JOINT_NAMES[:4]
  elif fault == "duplicate":
    asset.joint_ids = [1, 4, 7, 7]
  elif fault == "unordered":
    asset.preserve_order = False
  elif fault == "unresolved":
    asset.joint_ids = slice(None)
  elif fault == "resolved_order":
    asset.joint_ids = list(reversed(asset.joint_ids))
  elif fault == "outside":
    asset.joint_ids = [1, 4, 7, 99]
  elif fault == "position_shape":
    data.joint_pos = torch.zeros(3, 11)
  else:
    data.default_joint_pos = torch.zeros(2, 12)
  with pytest.raises(ValueError):
    moving_common_hip_pitch_deviation(env, asset)
