"""Named whole-COM geometry, EMA lifecycle, top-two costs and stop recovery gate."""

import math
from types import SimpleNamespace

import mujoco
import pytest
import torch
from mjlab.managers import RewardManager, RewardTermCfg, SceneEntityCfg

from pochi_rl.mjlab.entity_cfg import POCHI_XML
from pochi_rl.mjlab.lp_acrl_blind_v1_rewards import standing_joint_velocity_bounded
from pochi_rl.mjlab.lp_acrl_stride_center_rewards import (
  NOMINAL_FOOT_COM_X,
  moving_top_two_stride_center,
  nominal_stride_center_offsets,
  standing_joint_velocity_bounded_all_contact,
)
from pochi_rl.robot import (
  BASE_BODY,
  DEFAULT_JOINT_POS,
  FOOT_GEOMS,
  FOOT_SITES,
  JOINT_NAMES,
)


def _setup(count=1, dt=0.02, names=FOOT_SITES, **params):
  names = list(names)
  command = torch.zeros(count, 3)
  command[:, 0] = 0.25
  sites = torch.zeros(count, len(names), 3)
  ids = [names.index(n) for n in FOOT_SITES]
  sites[:, ids, 0] = torch.tensor(NOMINAL_FOOT_COM_X)
  sites[:, ids, 1] = torch.tensor([0.16, -0.16, 0.16, -0.16])
  sites[:, ids, 2] = -0.30
  data = SimpleNamespace(
    site_pos_w=sites,
    root_link_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(count, 1),
    root_com_pos_w=torch.full((count, 3), 123.0),  # Must NEVER be read.
    joint_vel=torch.ones(count, 12) * 0.1,
  )
  robot = SimpleNamespace(
    data=data,
    site_names=names,
    num_sites=len(names),
    indexing=SimpleNamespace(root_body_id=2),
    find_sites=lambda requested, preserve_order: (
      [names.index(name) for name in requested],
      list(requested),
    ),
  )
  subtree = torch.full((count, 4, 3), -987.0)
  subtree[:, 2] = 0
  sensor = SimpleNamespace(
    primary_names=[f"robot/{n}" for n in reversed(FOOT_GEOMS)],
    data=SimpleNamespace(found=torch.ones(count, 4)),
  )
  env = SimpleNamespace(
    num_envs=count,
    device="cpu",
    step_dt=dt,
    max_episode_length_s=20,
    command=command,
    command_manager=SimpleNamespace(get_command=lambda _: command),
    scene={"robot": robot, "feet_ground_contact": sensor},
    sim=SimpleNamespace(data=SimpleNamespace(subtree_com=subtree)),
  )
  p = {
    "asset_cfg": SceneEntityCfg(
      "robot", site_names=FOOT_SITES, site_ids=ids, preserve_order=True
    ),
    "tau_s": 0.25,
    "nominal_offsets_m": NOMINAL_FOOT_COM_X,
    **params,
  }
  cfg = RewardTermCfg(func=moving_top_two_stride_center, weight=-36, params=p)
  term = moving_top_two_stride_center(cfg, env)
  return env, term, cfg


def _step(env, term, cfg):
  return term(env, **cfg.params)


def _error(env, cfg, values):
  ids = cfg.params["asset_cfg"].site_ids
  env.scene["robot"].data.site_pos_w[:, ids, 0] = torch.tensor(
    NOMINAL_FOOT_COM_X
  ) + torch.as_tensor(values)


def test_nominal_constants_match_actual_asset_whole_com_and_named_default_pose():
  model = mujoco.MjModel.from_xml_path(str(POCHI_XML))
  data = mujoco.MjData(model)
  data.qpos[3:7] = [1, 0, 0, 0]
  for name in JOINT_NAMES:
    data.qpos[model.joint(name).qposadr] = DEFAULT_JOINT_POS[name]
  mujoco.mj_kinematics(model, data)
  mujoco.mj_comPos(model, data)
  root = model.body(BASE_BODY).id
  expected = tuple(
    data.site_xpos[model.site(n).id, 0] - data.subtree_com[root, 0] for n in FOOT_SITES
  )
  assert nominal_stride_center_offsets() == pytest.approx(expected, abs=1e-12)
  assert NOMINAL_FOOT_COM_X == pytest.approx(expected, abs=1e-12)
  body_com_target = (
    data.site_xpos[model.site(FOOT_SITES[0]).id, 0] - data.xipos[root, 0]
  )
  assert abs(body_com_target - expected[0]) > 1e-5
  assert model.body_mass.sum() == pytest.approx(10.922878, abs=1e-6)


def test_default_zero_and_exact_top_two_not_mean_or_square_of_mean():
  env, term, cfg = _setup(3)
  assert _step(env, term, cfg).eq(0).all()
  residual = torch.tensor(
    [[0.1, -0.2, 0.3, -0.4], [0.4, 0.3, 0.2, 0.1], [0.1, -0.1, 0.1, -0.1]]
  )
  _error(env, cfg, residual)
  alpha = -math.expm1(-0.02 / 0.25)
  expected = alpha**2 * residual.square().topk(2, dim=1).values.mean(1)
  torch.testing.assert_close(_step(env, term, cfg), expected)
  torch.testing.assert_close(term.ema_error_m, residual * alpha)


def test_named_site_order_with_extra_and_permuted_storage():
  env, term, cfg = _setup(names=("extra", *reversed(FOOT_SITES)))
  env.scene["robot"].data.site_pos_w[:, 0] = 100
  _error(env, cfg, [0.1, 0.2, 0.3, 0.4])
  expected = ((-math.expm1(-0.02 / 0.25)) ** 2) * (0.3**2 + 0.4**2) / 2
  assert _step(env, term, cfg).item() == pytest.approx(expected, abs=1e-8)


def test_world_translation_and_yaw_rotation_leave_metric_unchanged():
  env, term, cfg = _setup(3)
  _error(env, cfg, [[0.1, 0.2, 0.3, 0.4]] * 3)
  robot = env.scene["robot"]
  feet = robot.data.site_pos_w
  angle = 1.23
  rotation = torch.tensor(
    [
      [math.cos(angle), -math.sin(angle), 0.0],
      [math.sin(angle), math.cos(angle), 0.0],
      [0.0, 0.0, 1.0],
    ]
  )
  origin = torch.tensor([8.0, -13.0, 5.0])
  feet[1] = feet[1] @ rotation.T + origin
  env.sim.data.subtree_com[1, 2] = origin
  robot.data.root_link_quat_w[1] = torch.tensor(
    [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]
  )
  feet[2] += torch.tensor([-19.0, 6.0, 2.0])
  env.sim.data.subtree_com[2, 2] = torch.tensor([-19.0, 6.0, 2.0])
  out = _step(env, term, cfg)
  torch.testing.assert_close(out, out[0].expand_as(out), atol=5e-8, rtol=1e-5)


def test_world_horizontal_not_body_pitched_projection_and_actual_subtree_com():
  env, term, cfg = _setup(2)
  env.scene["robot"].data.root_link_quat_w[:] = torch.tensor(
    [math.cos(0.5), 0.0, math.sin(0.5), 0.0]
  )
  _error(env, cfg, [[0.1] * 4, [0.1] * 4])
  env.sim.data.subtree_com[1, 2, 0] = 0.05
  out = _step(env, term, cfg)
  alpha = -math.expm1(-0.02 / 0.25)
  torch.testing.assert_close(out, torch.tensor([0.1**2, 0.05**2]) * alpha**2)


def test_dt_invariance_for_constant_geometry_equal_elapsed_time():
  outputs = []
  for dt, steps in ((0.01, 100), (0.02, 50), (0.04, 25)):
    env, term, cfg = _setup(dt=dt)
    _error(env, cfg, [0.1, 0.2, 0.3, 0.4])
    for _ in range(steps):
      out = _step(env, term, cfg)
    outputs.append(out.item())
  expected = (1 - math.exp(-1 / 0.25)) ** 2 * (0.3**2 + 0.4**2) / 2
  assert outputs == pytest.approx([expected] * 3, abs=2e-7)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_exact_zero_command_clears_history_and_restart_has_no_leak(axis):
  env, term, cfg = _setup(3)
  _error(env, cfg, [[0.2] * 4] * 3)
  for _ in range(20):
    _step(env, term, cfg)
  env.command.zero_()
  env.command[1, axis] = 1e-6
  env.command[2, axis] = 2e-6
  out = _step(env, term, cfg)
  assert out[:2].eq(0).all() and out[2] > 0
  assert term.ema_error_m[:2].eq(0).all()
  env.command[:2, axis] = -0.1
  out = _step(env, term, cfg)
  expected = (0.2 * (-math.expm1(-0.02 / 0.25))) ** 2
  torch.testing.assert_close(out[:2], torch.full((2,), expected))


def test_subset_reset_and_inference_created_buffers_can_reset_outside_inference():
  with torch.inference_mode():
    env, term, cfg = _setup(3)
    _error(env, cfg, [[0.2] * 4] * 3)
    for _ in range(5):
      _step(env, term, cfg)
  before = term.ema_error_m.clone()
  term.reset(torch.tensor([1]))
  assert term.ema_error_m[1].eq(0).all()
  torch.testing.assert_close(term.ema_error_m[[0, 2]], before[[0, 2]])
  assert not term.ema_error_m.is_inference()
  term.reset(slice(0, 1))
  assert term.ema_error_m[0].eq(0).all()
  term.reset()
  assert term.ema_error_m.eq(0).all()


def test_foot_permutation_and_error_sign_do_not_change_top_two_fairness():
  env, term, cfg = _setup(3)
  errors = torch.tensor(
    [[0.1, 0.3, -0.4, 0.2], [0.3, 0.1, 0.2, -0.4], [-0.1, -0.3, 0.4, -0.2]]
  )
  _error(env, cfg, errors)
  out = _step(env, term, cfg)
  torch.testing.assert_close(out, out[0].expand_as(out))


def test_top_two_does_not_guarantee_lift_or_contact():
  env, term, cfg = _setup()
  env.scene["feet_ground_contact"].data.found.zero_()
  assert _step(env, term, cfg).item() == 0
  env.scene["feet_ground_contact"].data.found.fill_(1)
  assert _step(env, term, cfg).item() == 0


@pytest.mark.parametrize(
  "fault",
  [
    "quat_nan",
    "quat_inf",
    "quat_zero",
    "vertical_heading",
    "foot_nan",
    "com_inf",
    "finite_overflow",
  ],
)
def test_invalid_geometry_clears_only_affected_world_without_nan(fault):
  env, term, cfg = _setup(2)
  _error(env, cfg, [[0.2] * 4] * 2)
  for _ in range(5):
    _step(env, term, cfg)
  data = env.scene["robot"].data
  if fault == "quat_nan":
    data.root_link_quat_w[0, 0] = float("nan")
  elif fault == "quat_inf":
    data.root_link_quat_w[0, 1] = float("inf")
  elif fault == "quat_zero":
    data.root_link_quat_w[0].zero_()
  elif fault == "vertical_heading":
    data.root_link_quat_w[0] = torch.tensor([2**-0.5, 0.0, 2**-0.5, 0.0])
  elif fault == "foot_nan":
    data.site_pos_w[0, 0, 0] = float("nan")
  elif fault == "finite_overflow":
    data.site_pos_w[0, 0, 0] = 1e30
  else:
    env.sim.data.subtree_com[0, 2, 0] = float("inf")
  out = _step(env, term, cfg)
  assert torch.isfinite(out).all() and out[0] == 0 and out[1] > 0
  assert not term.geometry_valid[0] and term.geometry_valid[1]
  assert term.ema_error_m[0].eq(0).all()


def test_upside_down_and_scaled_quaternion_are_finite_no_heading_explosion():
  env, term, cfg = _setup(2)
  env.scene["robot"].data.root_link_quat_w[0] = torch.tensor([0.0, 1.0, 0.0, 0.0])
  env.scene["robot"].data.root_link_quat_w[1] *= 3
  assert torch.isfinite(_step(env, term, cfg)).all()
  assert term.geometry_valid.all()


@pytest.mark.parametrize("tau", [0.0, -0.2, float("inf"), float("nan")])
def test_invalid_tau_rejected(tau):
  with pytest.raises(ValueError, match="tau"):
    _setup(tau_s=tau)


def test_arbitrary_target_not_matching_actual_assets_rejected():
  with pytest.raises(ValueError, match="assets/default"):
    _setup(nominal_offsets_m=(0.25, 0.25, -0.25, -0.25))


def test_bad_resolved_order_and_runtime_parameter_change_rejected():
  env, term, cfg = _setup()
  cfg.params["asset_cfg"].site_ids = [3, 2, 1, 0]
  with pytest.raises(ValueError, match="named order"):
    _step(env, term, cfg)
  cfg.params["asset_cfg"].site_ids = [0, 1, 2, 3]
  cfg.params["tau_s"] = 0.3
  with pytest.raises(ValueError, match="cannot change"):
    _step(env, term, cfg)


def test_reward_manager_resolves_names_and_applies_weight_dt_once():
  env, _, cfg = _setup()
  cfg.params["asset_cfg"].site_ids = slice(None)
  manager = RewardManager({"stride_center": cfg}, env)
  asset = manager.get_term_cfg("stride_center").params["asset_cfg"]
  env.scene["robot"].data.site_pos_w[:, asset.site_ids, 0] += 0.1
  out = manager.compute(0.02)
  expected = -36 * 0.02 * (0.1 * (-math.expm1(-0.02 / 0.25))) ** 2
  torch.testing.assert_close(out, torch.tensor([expected]))
  manager.reset(torch.tensor([0]))
  assert manager.get_term_cfg("stride_center").func.ema_error_m.eq(0).all()


def test_stopped_gate_equals_original_exactly_in_all_contact_worlds():
  env, _, _ = _setup(5)
  env.command.zero_()
  env.scene["robot"].data.joint_vel[:] = torch.tensor([0.0, 0.01, 0.1, 1.0, 100.0])[
    :, None
  ]
  old = standing_joint_velocity_bounded(env, std=0.2)
  actual = standing_joint_velocity_bounded_all_contact(env, std=0.2)
  assert torch.equal(old, actual)
  assert actual[0] == 0 and actual[-1] < 1


def test_stopped_gate_releases_entire_cost_for_each_missing_named_foot():
  env, _, _ = _setup(6)
  env.command.zero_()
  found = env.scene["feet_ground_contact"].data.found
  for i in range(4):
    found[i, i] = 0
  found[4].zero_()
  out = standing_joint_velocity_bounded_all_contact(env)
  assert out[:5].eq(0).all() and out[5] > 0.4
  env.command[5, 2] = 0.1
  assert standing_joint_velocity_bounded_all_contact(env).eq(0).all()


def test_stopped_gate_preserves_original_std_validation_and_sensor_requirements():
  env, _, _ = _setup()
  with pytest.raises(ValueError, match="std"):
    standing_joint_velocity_bounded_all_contact(env, std=0)
  env.scene["feet_ground_contact"].primary_names = ["not_a_foot"] * 4
  with pytest.raises(ValueError, match="Pochi foot"):
    standing_joint_velocity_bounded_all_contact(env)
