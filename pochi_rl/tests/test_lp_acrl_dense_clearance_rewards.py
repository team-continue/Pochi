"""Physical clearance shaping and reference lifecycle, including its loopholes."""

from types import SimpleNamespace

import pytest
import torch
from mjlab.managers import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pochi_rl.mjlab.lp_acrl_dense_clearance_rewards import moving_dense_clearance
from pochi_rl.robot import FOOT_GEOMS, FOOT_SITES


def _setup(count=1, tracked=True, dt=0.02, **params):
  command = torch.zeros(count, 3)
  command[:, 0] = 0.25
  robot = SimpleNamespace(
    site_names=list(FOOT_SITES),
    data=SimpleNamespace(
      site_pos_w=torch.zeros(count, 4, 3),
      site_lin_vel_w=torch.zeros(count, 4, 3),
    ),
  )
  sensor = SimpleNamespace(
    primary_names=[f"robot/{name}" for name in FOOT_GEOMS],
    data=SimpleNamespace(
      found=torch.ones(count, 4),
      current_air_time=torch.zeros(count, 4) if tracked else None,
    ),
  )
  env = SimpleNamespace(
    num_envs=count,
    device="cpu",
    step_dt=dt,
    command=command,
    command_manager=SimpleNamespace(get_command=lambda _: command),
    scene={"robot": robot, "feet_ground_contact": sensor},
  )
  cfg = RewardTermCfg(
    func=moving_dense_clearance,
    weight=-4,
    params={
      "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES, preserve_order=True),
      **params,
    },
  )
  term = moving_dense_clearance(cfg, env)
  return env, sensor, term, cfg


def _step(env, sensor, term, cfg, contact=1, height=0.0, vx=0.0, vy=0.0, vz=0.0):
  sensor.data.found[:] = torch.as_tensor(contact)
  robot = env.scene["robot"].data
  robot.site_pos_w[..., 2] = torch.as_tensor(height)
  for axis, value in enumerate((vx, vy, vz)):
    robot.site_lin_vel_w[..., axis] = torch.as_tensor(value)
  if sensor.data.current_air_time is not None:
    sensor.data.current_air_time.copy_(
      torch.where(sensor.data.found > 0, 0, sensor.data.current_air_time + env.step_dt)
    )
  return term(env, **cfg.params)


@pytest.mark.parametrize("tracked", [True, False])
def test_first_air_sample_has_analytic_dense_cost_without_airtime_gate(tracked):
  env, sensor, term, cfg = _setup(tracked=tracked)
  _step(env, sensor, term, cfg, height=-0.5)
  cost = _step(env, sensor, term, cfg, contact=0, height=-0.49, vx=0.3, vy=0.4)
  assert cost.item() == pytest.approx(0.5 * (0.025 / 0.035) ** 2, abs=2e-6)
  assert term.air_s.max() <= 0.02  # Still below any qualified-swing air threshold.


def test_fixed_four_foot_average_does_not_renormalize_by_airborne_count():
  env, sensor, term, cfg = _setup()
  _step(env, sensor, term, cfg)
  cost = _step(env, sensor, term, cfg, contact=[0, 1, 1, 1], vx=0.8)
  assert cost.item() == pytest.approx(0.2)


def test_current_lift_not_peak_reintroduces_cost_during_descent():
  env, sensor, term, cfg = _setup()
  _step(env, sensor, term, cfg)
  assert _step(env, sensor, term, cfg, contact=0, height=0.05, vx=0.2) == 0
  cost = _step(env, sensor, term, cfg, contact=0, height=0.0175, vx=0.2)
  assert cost.item() == pytest.approx(0.05)


@pytest.mark.parametrize("height", [0.035, 0.05])
def test_target_equality_or_greater_clearance_has_zero_cost(height):
  env, sensor, term, cfg = _setup()
  _step(env, sensor, term, cfg)
  assert _step(env, sensor, term, cfg, contact=0, height=height, vx=1).item() == 0


def test_negative_lift_is_bounded_and_vertical_velocity_does_not_count():
  env, sensor, term, cfg = _setup()
  _step(env, sensor, term, cfg)
  assert _step(env, sensor, term, cfg, contact=0, height=-1, vx=0.2) == pytest.approx(
    0.2
  )
  assert _step(env, sensor, term, cfg, contact=0, height=-1, vz=10) == 0


def test_more_lift_or_less_horizontal_speed_monotonically_reduces_cost():
  env, sensor, term, cfg = _setup(4)
  _step(env, sensor, term, cfg)
  cost = _step(
    env,
    sensor,
    term,
    cfg,
    contact=0,
    height=[[0.0] * 4, [0.01] * 4, [0.02] * 4, [0.03] * 4],
    vx=0.4,
  )
  assert torch.all(cost[:-1] > cost[1:])
  slower = _step(
    env,
    sensor,
    term,
    cfg,
    contact=0,
    height=[[0.0] * 4, [0.01] * 4, [0.02] * 4, [0.03] * 4],
    vx=0.2,
  )
  torch.testing.assert_close(slower, cost / 2)


def test_planting_and_stationary_hover_are_explicit_zero_cost_loopholes():
  env, sensor, term, cfg = _setup()
  assert _step(env, sensor, term, cfg, contact=1, vx=2) == 0
  assert _step(env, sensor, term, cfg, contact=0, height=0.001, vx=0) == 0


def test_no_observed_contact_means_no_invented_reference_or_cost():
  env, sensor, term, cfg = _setup()
  for _ in range(10):
    assert _step(env, sensor, term, cfg, contact=0, height=3, vx=0.5) == 0
  assert not term.has_reference.any()
  _step(env, sensor, term, cfg, contact=1, height=3)
  assert _step(env, sensor, term, cfg, contact=0, height=3.01, vx=0.5) > 0


def test_each_observed_touchdown_updates_reference_and_clears_air_clock():
  env, sensor, term, cfg = _setup()
  _step(env, sensor, term, cfg, height=0)
  _step(env, sensor, term, cfg, contact=0, height=0.02, vx=1)
  assert _step(env, sensor, term, cfg, contact=1, height=-0.5, vx=1) == 0
  assert term.air_s.eq(0).all()
  torch.testing.assert_close(term.contact_height, torch.full((1, 4), -0.5))
  assert _step(
    env, sensor, term, cfg, contact=0, height=-0.4825, vx=0.2
  ).item() == pytest.approx(0.05, abs=2e-7)


def test_unseen_substep_touchdown_invalidates_old_reference_until_observed_contact():
  env, sensor, term, cfg = _setup()
  _step(env, sensor, term, cfg)
  assert _step(env, sensor, term, cfg, contact=0, height=0.01, vx=1) > 0
  sensor.data.current_air_time.fill_(0.0025)
  assert term(env, **cfg.params) == 0
  assert not term.has_reference.any()
  for _ in range(3):
    assert _step(env, sensor, term, cfg, contact=0, height=0.01, vx=1) == 0
  _step(env, sensor, term, cfg, contact=1, height=0.01)
  assert _step(env, sensor, term, cfg, contact=0, height=0.02, vx=1) > 0


def test_stopping_clears_reference_even_hovering_and_restart_requires_new_contact():
  env, sensor, term, cfg = _setup(2)
  _step(env, sensor, term, cfg)
  _step(env, sensor, term, cfg, contact=0, height=0.01, vx=1)
  env.command[0].zero_()
  cost = _step(env, sensor, term, cfg, contact=0, height=0.01, vx=1)
  assert cost[0] == 0 and cost[1] > 0
  assert not term.has_reference[0].any()
  assert term.air_s[0].eq(0).all() and term.contact_height[0].eq(0).all()
  env.command[0, 0] = 0.25
  assert _step(env, sensor, term, cfg, contact=0, height=0.01, vx=1)[0] == 0


def test_subset_reset_and_inference_mode_preserve_normal_buffer_storage():
  env, sensor, _, cfg = _setup(3)
  with torch.inference_mode():
    term = moving_dense_clearance(cfg, env)
    _step(env, sensor, term, cfg)
    _step(env, sensor, term, cfg, contact=0, height=0.01, vx=1)
  names = ("contact_height", "air_s", "has_reference")
  pointers = {name: getattr(term, name).data_ptr() for name in names}
  term.reset(torch.tensor([1]))
  assert term.has_reference[0].all() and not term.has_reference[1].any()
  term.reset(slice(2, 3))
  assert not term.has_reference[2].any()
  term.reset(None)
  for name in names:
    value = getattr(term, name)
    assert not value.is_inference() and value.data_ptr() == pointers[name]
    assert value.eq(0).all()


def test_foot_permutation_and_world_height_translation_leave_cost_unchanged():
  env, sensor, term, cfg = _setup(2)
  permutation = torch.tensor([2, 0, 3, 1])
  contact = torch.tensor([0, 1, 0, 0])
  height = torch.tensor([0.01, 0.0, 0.03, 0.02])
  speed = torch.tensor([0.1, 0.7, 0.3, 0.5])
  _step(env, sensor, term, cfg, height=[[0] * 4, [2] * 4])
  cost = _step(
    env,
    sensor,
    term,
    cfg,
    contact=torch.stack((contact, contact[permutation])),
    height=torch.stack((height, height[permutation] + 2)),
    vx=torch.stack((speed, speed[permutation])),
  )
  torch.testing.assert_close(cost[0], cost[1], atol=2e-7, rtol=0)


def test_sensor_slot_order_is_mapped_to_named_site_order():
  env, sensor, term, cfg = _setup()
  _step(env, sensor, term, cfg)
  sensor.primary_names = [f"robot/{FOOT_GEOMS[i]}" for i in (2, 0, 3, 1)]
  cost = _step(env, sensor, term, cfg, contact=[1, 0, 1, 1], vx=[0.8, 0, 0, 0])
  assert cost.item() == pytest.approx(0.2)


@pytest.mark.parametrize("target", [0, -0.01, float("nan"), float("inf")])
def test_invalid_clearance_target_rejected(target):
  with pytest.raises(ValueError, match="target_lift"):
    _setup(target_lift=target)


@pytest.mark.parametrize("dt", [0, -0.02, float("nan")])
def test_invalid_control_dt_rejected(dt):
  with pytest.raises(ValueError, match="control dt"):
    _setup(dt=dt)


@pytest.mark.parametrize(
  "fault", ["order", "unordered", "duplicate_contact", "slots", "airtime"]
)
def test_invalid_foot_mapping_and_sensor_shapes_rejected(fault):
  env, sensor, term, cfg = _setup()
  if fault == "order":
    cfg.params["asset_cfg"].site_names = tuple(reversed(FOOT_SITES))
  elif fault == "unordered":
    cfg.params["asset_cfg"].preserve_order = False
  elif fault == "duplicate_contact":
    sensor.primary_names[0] = sensor.primary_names[1]
  elif fault == "slots":
    sensor.data.found = torch.zeros(1, 3)
  else:
    sensor.data.current_air_time = torch.zeros(1, 3)
  with pytest.raises(ValueError):
    term(env, **cfg.params)
