"""The optional posture cost is moving-only, mean12, stateless and phase-free."""

from types import SimpleNamespace

import pytest
import torch
from mjlab.managers import RewardManager, RewardTermCfg, SceneEntityCfg

from pochi_rl.mjlab.lp_acrl_moving_posture_rewards import moving_default_joint_deviation
from pochi_rl.robot import JOINT_NAMES


def _setup(count=3):
  command = torch.zeros(count, 3)
  command[:, 0] = 0.25
  default = torch.tensor([0, 0.8, -1.5, 0, 0.8, -1.5, 0, -0.8, 1.5, 0, -0.8, 1.5])
  data = SimpleNamespace(
    joint_pos=default.repeat(count, 1), default_joint_pos=default.repeat(count, 1)
  )
  entity = SimpleNamespace(
    data=data,
    joint_names=list(JOINT_NAMES),
    num_joints=12,
    find_joints=lambda names, preserve_order: (list(range(12)), list(names)),
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
  asset = SceneEntityCfg("robot", joint_names=JOINT_NAMES, preserve_order=True)
  return env, data, asset


def test_default_pose_zero_and_single_joint_error_uses_fixed_twelve_denominator():
  env, data, asset = _setup()
  assert moving_default_joint_deviation(env, asset).eq(0).all()
  data.joint_pos[0, 0] += 0.3
  data.joint_pos[1] += 0.3
  result = moving_default_joint_deviation(env, asset)
  torch.testing.assert_close(result, torch.tensor([0.3**2 / 12, 0.3**2, 0.0]))


def test_all_four_common_hip_pitch_offsets_have_analytic_cost():
  env, data, asset = _setup(1)
  data.joint_pos[:, [1, 4, 7, 10]] -= 0.377
  assert moving_default_joint_deviation(env, asset).item() == pytest.approx(
    0.377**2 / 3
  )


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_exact_stop_threshold_and_each_moving_command_axis(axis):
  env, data, asset = _setup(3)
  data.joint_pos += 0.2
  env.command.zero_()
  env.command[1, axis] = 1e-6
  env.command[2, axis] = -2e-6
  torch.testing.assert_close(
    moving_default_joint_deviation(env, asset), torch.tensor([0.0, 0.0, 0.04])
  )


def test_actual_positions_not_action_targets_or_velocity_determine_cost():
  env, data, asset = _setup(1)
  data.joint_pos += 0.1
  data.joint_vel = torch.full_like(data.joint_pos, 100.0)
  env.action_manager = SimpleNamespace(action=torch.zeros(1, 12))
  first = moving_default_joint_deviation(env, asset)
  env.action_manager.action.fill_(10)
  data.joint_vel.zero_()
  torch.testing.assert_close(moving_default_joint_deviation(env, asset), first)


def test_per_environment_defaults_and_broadcast_default_are_supported():
  env, data, asset = _setup(2)
  data.default_joint_pos[1] += 0.2
  data.joint_pos.copy_(data.default_joint_pos)
  assert moving_default_joint_deviation(env, asset).eq(0).all()
  data.default_joint_pos = data.default_joint_pos[:1]
  torch.testing.assert_close(
    moving_default_joint_deviation(env, asset), torch.tensor([0.0, 0.04])
  )


def test_sign_joint_permutation_and_sample_order_do_not_define_a_gait_phase():
  env, data, asset = _setup(3)
  error = torch.linspace(-0.3, 0.3, 12)
  data.joint_pos += torch.stack((error, -error, error.flip(0)))
  result = moving_default_joint_deviation(env, asset)
  torch.testing.assert_close(result, result[0].expand(3))
  data.joint_pos[:] = data.default_joint_pos
  assert moving_default_joint_deviation(env, asset).eq(0).all()


def test_inference_and_subset_environment_resets_need_no_reward_state():
  env, data, asset = _setup(3)
  data.joint_pos += 0.2
  with torch.inference_mode():
    moving_default_joint_deviation(env, asset)
  data.joint_pos[1].copy_(data.default_joint_pos[1])
  torch.testing.assert_close(
    moving_default_joint_deviation(env, asset), torch.tensor([0.04, 0.0, 0.04])
  )


def test_reward_manager_applies_negative_weight_and_dt_once():
  env, data, asset = _setup(1)
  data.joint_pos += 0.2
  manager = RewardManager(
    {
      "moving_posture": RewardTermCfg(
        func=moving_default_joint_deviation, weight=-4, params={"asset_cfg": asset}
      )
    },
    env,
  )
  torch.testing.assert_close(manager.compute(0.02), torch.tensor([-4 * 0.04 * 0.02]))


@pytest.mark.parametrize(
  "fault", ["names", "unordered", "resolved", "count", "default"]
)
def test_missing_or_incorrect_joint_selection_rejected(fault):
  env, data, asset = _setup()
  if fault == "names":
    asset.joint_names = JOINT_NAMES[:4]
  elif fault == "unordered":
    asset.preserve_order = False
  elif fault == "resolved":
    asset.joint_ids = list(reversed(range(12)))
  elif fault == "count":
    data.joint_pos = torch.zeros(3, 13)
  else:
    data.default_joint_pos = torch.zeros(2, 12)
  with pytest.raises(ValueError):
    moving_default_joint_deviation(env, asset)
