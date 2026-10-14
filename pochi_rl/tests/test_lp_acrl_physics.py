"""Reject finite simulator explosions before rewards or curriculum credit."""

import json
from collections import deque
from types import SimpleNamespace

import pytest
import torch
from mjlab.managers import RewardTermCfg
from mjlab.managers.reward_manager import RewardManager
from mjlab.utils.nan_guard import NanGuard
from rsl_rl.utils.logger import Logger

from pochi_rl.lp_acrl import LPACRLCurriculum
from pochi_rl.lp_acrl.physics import (
  PHYSICS_BODY_FIELDS,
  PHYSICS_FIELDS,
  PHYSICS_LIMITS,
  invalid_physics_mask,
)
from pochi_rl.mjlab.lp_acrl_env_cfg import (
  pochi_lp_acrl_flat_env_cfg,
  pochi_lp_acrl_rough_env_cfg,
)
from pochi_rl.mjlab.lp_acrl_rewards import assert_valid_physics
from pochi_rl.mjlab.lp_acrl_runner import (
  LPACRLOnPolicyRunner,
  _capture_with_replay_inputs,
  _log_with_all_episode_keys,
)


def _env(num_envs=3, *, evaluation=False):
  data = SimpleNamespace(
    **{
      name: torch.zeros(num_envs, 12 if name.startswith("joint_") else 3)
      for name in PHYSICS_FIELDS
    }
  )
  data.root_link_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(num_envs, 1)
  term = SimpleNamespace(
    _evaluation=evaluation,
    cfg=SimpleNamespace(rough=False),
    task_ids=torch.full((num_envs,), 267),
  )
  return SimpleNamespace(
    num_envs=num_envs,
    device="cpu",
    scene={"robot": SimpleNamespace(data=data)},
    command_manager=SimpleNamespace(get_term=lambda _: term),
  )


@pytest.mark.parametrize(
  ("field", "limit"),
  [
    ("joint_vel", PHYSICS_LIMITS["joint_speed_rad_s"]),
    ("joint_acc", PHYSICS_LIMITS["joint_acceleration_rad_s2"]),
    ("root_link_lin_vel_b", PHYSICS_LIMITS["base_linear_speed_m_s"]),
    ("root_link_ang_vel_b", PHYSICS_LIMITS["base_angular_speed_rad_s"]),
  ],
)
def test_threshold_boundaries_reject_only_impossible_environments(field, limit):
  data = _env().scene["robot"].data
  state = getattr(data, field)
  state[0, 0] = limit
  state[1, 0] = -(limit + 1)
  original = state.clone()
  assert invalid_physics_mask(data).tolist() == [False, True, False]
  torch.testing.assert_close(state, original)


def test_base_speed_limits_use_vector_norms():
  data = _env().scene["robot"].data
  data.root_link_lin_vel_b[0] = torch.tensor([15.0, 15.0, 0.0])
  data.root_link_ang_vel_b[1] = torch.tensor([40.0, 40.0, 0.0])
  assert invalid_physics_mask(data).tolist() == [True, True, False]


@pytest.mark.parametrize("field", PHYSICS_FIELDS)
@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
def test_nonfinite_pose_velocity_and_joint_state_are_rejected(field, invalid):
  data = _env().scene["robot"].data
  getattr(data, field)[1, 0] = invalid
  assert invalid_physics_mask(data).tolist() == [False, True, False]


@pytest.mark.parametrize("field", PHYSICS_BODY_FIELDS)
def test_full_body_state_is_checked_when_available(field):
  data = _env().scene["robot"].data
  state = torch.zeros(3, 13, 7 if field == "body_link_pose_w" else 6)
  setattr(data, field, state)
  assert not invalid_physics_mask(data).any()
  state[1, 12, 0] = float("nan")
  assert invalid_physics_mask(data).tolist() == [False, True, False]


def test_training_guard_preserves_normal_values_and_evaluation_classifies_itself():
  env = _env()
  env.scene["robot"].data.joint_vel[:] = 35.0
  torch.testing.assert_close(assert_valid_physics(env), torch.zeros(3))
  assert not hasattr(env, "_lp_acrl_physics_fault_ids")
  env.scene["robot"].data.joint_vel[1, 0] = 57000.0
  with pytest.raises(ValueError, match="before reward/PPO/LP credit"):
    assert_valid_physics(env)
  assert env._lp_acrl_physics_fault_ids.tolist() == [1]
  env.command_manager.get_term("base_velocity")._evaluation = True
  torch.testing.assert_close(assert_valid_physics(env), torch.zeros(3))
  assert invalid_physics_mask(env.scene["robot"].data).tolist() == [False, True, False]


def test_rough_guard_is_first_and_nonzero_weight_flat_task_is_unchanged():
  rough = pochi_lp_acrl_rough_env_cfg()
  assert next(iter(rough.rewards)) == "physics_validity"
  assert rough.rewards["physics_validity"].func is assert_valid_physics
  assert rough.rewards["physics_validity"].weight == 1.0
  assert "physics_validity" not in pochi_lp_acrl_flat_env_cfg().rewards


def test_native_reward_manager_aborts_before_reward_ppo_or_lp_credit():
  env = _env()
  calls = []

  def reward_after_guard(env):
    calls.append("reward")
    return torch.full((env.num_envs,), 2.0)

  manager = RewardManager(
    {
      "physics_validity": RewardTermCfg(func=assert_valid_physics, weight=1.0),
      "tracking": RewardTermCfg(func=reward_after_guard, weight=0.5),
    },
    env,
  )
  # Ordinary steps keep the exact existing reward and episode sum.
  torch.testing.assert_close(manager.compute(0.02), torch.full((3,), 0.02))
  calls.clear()
  original_sums = {key: value.clone() for key, value in manager._episode_sums.items()}
  curriculum = LPACRLCurriculum(3)
  env.scene["robot"].data.joint_vel[1, 0] = 57000.0
  with pytest.raises(ValueError, match="before reward/PPO/LP credit"):
    rewards = manager.compute(0.02)
    calls.append("PPO")
    curriculum.record_episodes(torch.arange(3), rewards)
  assert calls == []
  assert curriculum.episode_counts.tolist() == [0, 0, 0]
  for name, previous in original_sums.items():
    torch.testing.assert_close(manager._episode_sums[name], previous)
  assert env.scene["robot"].data.joint_vel[1, 0] == 57000.0


def test_finite_failure_artifact_preserves_raw_state_and_reason(tmp_path):
  env = _env()
  env.scene["robot"].data.joint_vel[1, 0] = 57000.0
  env._lp_acrl_physics_fault_ids = torch.tensor([1])
  env.obs_buf = {"actor": torch.ones(3, 48)}
  env.action_manager = SimpleNamespace(action=torch.zeros(3, 12))
  env.reward_buf = torch.zeros(3)
  env.common_step_counter = 11184
  runner = LPACRLOnPolicyRunner.__new__(LPACRLOnPolicyRunner)
  runner.env = SimpleNamespace(unwrapped=env)
  runner.logger = SimpleNamespace(log_dir=str(tmp_path))
  runner.current_learning_iteration = 466
  runner._record_failure(reason="finite joint speed fault")
  report = json.loads((tmp_path / "failure.json").read_text())
  assert report["env_ids"] == [1]
  assert report["task_ids"] == [267]
  assert report["reason"] == "finite joint speed fault"
  assert report["physics_fault_reasons"]["1"] == ["joint_speed"]
  assert report["physics_fault_counts"]["joint_speed"] == 1
  assert sum(report["nonfinite_counts"].values()) == 0
  states = torch.load(tmp_path / "failure.pt", weights_only=True)
  assert states["joint_vel"].shape == (1, 12)
  assert states["joint_vel"][0, 0] == 57000.0


def test_capture_preserves_pre_step_controls_and_warmstart_independently():
  guard = NanGuard.__new__(NanGuard)
  guard.enabled = True
  guard.buffer = deque(maxlen=2)
  guard.step_counter = 0
  guard.mj_model = SimpleNamespace(na=1, nmocap=1)
  guard._lp_acrl_original_capture = guard.capture
  data = SimpleNamespace(
    qpos=torch.zeros(3, 19),
    qvel=torch.zeros(3, 18),
    act=torch.zeros(3, 1),
    mocap_pos=torch.zeros(3, 1, 3),
    mocap_quat=torch.zeros(3, 1, 4),
    ctrl=torch.zeros(3, 12),
    qacc_warmstart=torch.zeros(3, 18),
  )
  for step in range(3):
    data.ctrl.fill_(step)
    data.qacc_warmstart.fill_(step * 10)
    _capture_with_replay_inputs(guard, data)
  data.ctrl.fill_(999)
  data.qacc_warmstart.fill_(999)
  assert [frame["step"] for frame in guard.buffer] == [1, 2]
  for step, frame in zip([1, 2], guard.buffer, strict=True):
    assert (frame["ctrl"] == step).all()
    assert (frame["qacc_warmstart"] == step * 10).all()
    assert set(frame) == {
      "step", "qpos", "qvel", "act", "mocap_pos", "mocap_quat",
      "ctrl", "qacc_warmstart",
    }
  guard.enabled = False
  _capture_with_replay_inputs(guard, data)
  assert guard.step_counter == 3
  assert (guard.buffer[-1]["ctrl"] == 2).all()


def test_failure_replay_is_bounded_and_preserves_world_randomization(tmp_path):
  env = _env(num_envs=40)
  env.scene["robot"].data.joint_vel[6:, 0] = 57000.0
  env.obs_buf = {"actor": torch.ones(40, 48)}
  world = torch.arange(40).float()[:, None]
  env.action_manager = SimpleNamespace(
    action=torch.zeros(40, 12), prev_action=world.repeat(1, 12)
  )
  env.reward_buf = torch.zeros(40)
  env.common_step_counter = 15
  env.cfg = SimpleNamespace(decimation=8)
  history = deque(maxlen=20)
  for step in range(100, 120):
    history.append({
      "step": step,
      "qpos": world.repeat(1, 19) + step,
      "qvel": world.repeat(1, 18),
      "ctrl": world.repeat(1, 12),
      "qacc_warmstart": world.repeat(1, 18) + step * 10,
      "act": world.clone(),
      "mocap_pos": world[:, None].repeat(1, 1, 3),
      "mocap_quat": world[:, None].repeat(1, 1, 4),
    })
  env.sim = SimpleNamespace(
    nan_guard=SimpleNamespace(
      buffer=history, enabled=True, state_spec=7, state_size=38, step_counter=120
    ),
    model=SimpleNamespace(
      body_mass=world.repeat(1, 13),
      geom_friction=torch.full((1, 17, 3), 1.25),
    ),
  )
  manifest = {
    "source_sha256": {"assets/pochi/pochi.xml": "modelhash"},
    "physics_solver": "cg",
    "seed": 42,
  }
  (tmp_path / "run.json").write_text(json.dumps(manifest))
  runner = LPACRLOnPolicyRunner.__new__(LPACRLOnPolicyRunner)
  runner.env = SimpleNamespace(unwrapped=env)
  runner.logger = SimpleNamespace(log_dir=str(tmp_path))
  runner.current_learning_iteration = 4
  runner._record_failure(reason="finite fault")

  saved = torch.load(tmp_path / "failure.pt", weights_only=True)
  replay = saved["replay"]
  assert replay["env_ids"].tolist() == list(range(6, 38))
  assert [frame["step"] for frame in replay["history"]] == list(range(104, 120))
  assert replay["guard_state_spec"] == 7
  assert replay["guard_step_counter"] == 120
  assert replay["control_decimation"] == 8
  assert replay["source_manifest"] == manifest
  assert replay["history"][0]["qpos"].shape == (32, 19)
  assert replay["history"][0]["qpos"][0, 0] == 110
  assert replay["history"][-1]["qacc_warmstart"][-1, 0] == 1227
  assert replay["history"][-1]["ctrl"][-1, 0] == 37
  assert replay["history"][0]["mocap_pos"].shape == (32, 1, 3)
  assert replay["history"][0]["act"].shape == (32, 1)
  torch.testing.assert_close(saved["previous_action"], world[6:38].repeat(1, 12))
  torch.testing.assert_close(
    replay["model_parameters"]["body_mass"], world[6:38].repeat(1, 13)
  )
  torch.testing.assert_close(
    replay["model_parameters"]["geom_friction"], torch.full((32, 17, 3), 1.25)
  )
  report = json.loads((tmp_path / "failure.json").read_text())
  assert report["total_fault_envs"] == 34
  assert report["replay"]["history_frames"] == 16
  assert report["replay"]["previous_action_available"]
  assert (tmp_path / "failure.pt").stat().st_size < 1024**2
  assert not list(tmp_path.glob("*.mjb"))


def test_logger_includes_later_keys_without_biasing_episode_means():
  first_packet = {}
  logger = Logger(
    log_dir=None,
    cfg={"algorithm": {"rnd_cfg": None}, "num_steps_per_env": 24},
    env_cfg={},
    num_envs=3,
    is_distributed=False,
    gpu_world_size=1,
    gpu_global_rank=0,
    device="cpu",
  )
  logger.ep_extras = [
    first_packet,
    {"Episode_Reward/joint_acc_l2": torch.tensor(-2.0)},
    {"Episode_Reward/joint_acc_l2": torch.tensor(-4.0), "LP_ACRL/stage": 5},
  ]
  scalars = {}
  logger.writer = SimpleNamespace(
    add_scalar=lambda key, value, step: scalars.update({key: (float(value), step)})
  )
  logger.logger_type = "tensorboard"
  logger._lp_acrl_original_log = logger.log
  _log_with_all_episode_keys(
    logger,
    it=7,
    start_it=0,
    total_it=10,
    collect_time=0.1,
    learn_time=0.1,
    loss_dict={},
    learning_rate=1e-3,
    action_std=torch.ones(12),
    rnd_weight=None,
    print_minimal=True,
  )
  assert scalars["Episode_Reward/joint_acc_l2"] == (-3.0, 7)
  assert scalars["LP_ACRL/stage"] == (5.0, 7)
  assert first_packet == {}
  assert logger.ep_extras == []
