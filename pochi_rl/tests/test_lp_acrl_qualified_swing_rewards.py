"""Qualified occupancy measures real swings, without claiming a phase or gait."""

import math
from types import SimpleNamespace

import pytest
import torch
from mjlab.managers import RewardManager, RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from pochi_rl.mjlab.lp_acrl_qualified_swing_rewards import (
  moving_qualified_swing_occupancy,
)
from pochi_rl.robot import FOOT_GEOMS, FOOT_SITES


def _setup(count=1, dt=0.02, tracked=True, **parameters):
  command = torch.zeros(count, 3)
  command[:, 0] = 0.25
  data = SimpleNamespace(site_pos_w=torch.zeros(count, 4, 3))
  robot = SimpleNamespace(
    data=data,
    site_names=list(FOOT_SITES),
    num_sites=4,
    find_sites=lambda names, preserve_order: (list(range(4)), list(names)),
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
    max_episode_length_s=20,
    command=command,
    command_manager=SimpleNamespace(get_command=lambda _: command),
    scene={"robot": robot, "feet_ground_contact": sensor},
  )
  params = {
    "asset_cfg": SceneEntityCfg("robot", site_names=FOOT_SITES, preserve_order=True),
    **parameters,
  }
  cfg = RewardTermCfg(func=moving_qualified_swing_occupancy, weight=-8, params=params)
  term = moving_qualified_swing_occupancy(cfg, env)
  return env, sensor, term, cfg


def _step(env, sensor, term, cfg, contact=1, height=0.0):
  sensor.data.found[:] = torch.as_tensor(contact)
  env.scene["robot"].data.site_pos_w[:, :, 2] = torch.as_tensor(height)
  if sensor.data.current_air_time is not None:
    sensor.data.current_air_time.copy_(
      torch.where(sensor.data.found > 0, 0, sensor.data.current_air_time + env.step_dt)
    )
  return term(env, **cfg.params)


def test_planted_feet_have_analytic_cost_approaching_point_zero_four():
  env, sensor, term, cfg = _setup()
  for _ in range(100):
    cost = _step(env, sensor, term, cfg)
  expected = 0.2**2 * (1 - math.exp(-2)) ** 2
  torch.testing.assert_close(cost, torch.tensor([expected]), atol=1e-7, rtol=0)
  for _ in range(900):
    cost = _step(env, sensor, term, cfg)
  assert cost.item() == pytest.approx(0.04, abs=1e-7)
  assert not term.qualified.any()


@pytest.mark.parametrize("air_frames", [1, 2])
def test_brief_contact_gaps_never_qualify_even_with_large_lift(air_frames):
  env, sensor, term, cfg = _setup()
  for _ in range(100):
    _step(env, sensor, term, cfg)
    for _ in range(air_frames):
      _step(env, sensor, term, cfg, contact=0, height=0.08)
      assert not term.qualified.any()
  expected = 0.2 * math.exp(-100 * (1 + air_frames) * env.step_dt)
  torch.testing.assert_close(
    term.occupancy, torch.full((1, 4), expected), atol=1e-7, rtol=0
  )


@pytest.mark.parametrize("tracked", [True, False])
def test_threshold_equality_qualifies_then_latches_until_actual_touchdown(tracked):
  env, sensor, term, cfg = _setup(tracked=tracked)
  _step(env, sensor, term, cfg, height=-0.4)
  for _ in range(2):
    _step(env, sensor, term, cfg, contact=0, height=-0.375)
    assert not term.qualified.any()
  _step(env, sensor, term, cfg, contact=0, height=-0.375)
  assert term.qualified.all()
  _step(env, sensor, term, cfg, contact=0, height=-0.399)
  assert term.qualified.all()  # Descending below 25 mm does not erase a true swing.
  _step(env, sensor, term, cfg, contact=1, height=-0.399)
  assert not term.qualified.any() and term.peak_lift.eq(0).all()
  assert term.air_s.eq(0).all()
  torch.testing.assert_close(term.contact_height, torch.full((1, 4), -0.399))


def test_long_low_hover_cannot_qualify_and_sustained_true_hover_costs_point_six_four():
  env, sensor, term, cfg = _setup(2)
  _step(env, sensor, term, cfg)
  for _ in range(1000):
    cost = _step(env, sensor, term, cfg, contact=0, height=[[0.0249] * 4, [0.025] * 4])
  assert not term.qualified[0].any() and term.qualified[1].all()
  torch.testing.assert_close(cost, torch.tensor([0.04, 0.64]), atol=3e-6, rtol=0)


def test_initial_air_without_observed_contact_never_invents_reference():
  env, sensor, term, cfg = _setup()
  for _ in range(20):
    _step(env, sensor, term, cfg, contact=0, height=3.0)
  assert not term.has_reference.any() and not term.qualified.any()
  _step(env, sensor, term, cfg, contact=1, height=3.0)
  for _ in range(3):
    _step(env, sensor, term, cfg, contact=0, height=3.025)
  assert term.qualified.all()


def test_sensor_substep_air_time_prevents_false_continuous_air_and_clears_latch():
  env, sensor, term, cfg = _setup()
  _step(env, sensor, term, cfg)
  sensor.data.found.zero_()
  env.scene["robot"].data.site_pos_w[:, :, 2] = 0.03
  for _ in range(10):
    sensor.data.current_air_time.fill_(0.0025)
    term(env, **cfg.params)
    assert not term.qualified.any()
  assert not term.has_reference.any()
  _step(env, sensor, term, cfg)
  sensor.data.found.zero_()
  env.scene["robot"].data.site_pos_w[:, :, 2] = 0.03
  for duration in (0.02, 0.04, 0.06, 0.08):
    sensor.data.current_air_time.fill_(duration)
    term(env, **cfg.params)
  assert term.qualified.all()
  # Touchdown and takeoff within one control step.
  sensor.data.current_air_time.fill_(0.0025)
  term(env, **cfg.params)
  assert not term.qualified.any() and not term.has_reference.any()
  assert term.peak_lift.eq(0).all()
  for _ in range(10):
    _step(env, sensor, term, cfg, contact=0, height=0.08)
  # An actually observed contact must restore the reference.
  assert not term.qualified.any()


def test_substep_touchdown_cannot_transfer_old_high_peak_to_later_low_swing():
  env, sensor, term, cfg = _setup()
  _step(env, sensor, term, cfg)
  _step(env, sensor, term, cfg, contact=0, height=0.04)
  assert term.peak_lift.eq(0.04).all() and not term.qualified.any()
  env.scene["robot"].data.site_pos_w[:, :, 2] = 0.005
  # A touchdown inside the next 20 ms leaves the SAME sampled 20 ms air clock.
  sensor.data.current_air_time.fill_(0.02)
  term(env, **cfg.params)
  assert term.peak_lift.eq(0).all() and not term.has_reference.any()
  for _ in range(10):
    _step(env, sensor, term, cfg, contact=0, height=0.005)
  assert not term.qualified.any()


def test_stop_clears_all_swing_memory_even_when_feet_remain_hovering():
  env, sensor, term, cfg = _setup(2)
  _step(env, sensor, term, cfg)
  for _ in range(5):
    _step(env, sensor, term, cfg, contact=0, height=0.04)
  env.command[0].zero_()
  cost = _step(env, sensor, term, cfg, contact=0, height=0.04)
  assert cost[0] == 0 and term.occupancy[0].eq(0.2).all()
  for name in ("air_s", "contact_height", "peak_lift", "has_reference", "qualified"):
    assert getattr(term, name)[0].eq(0).all()
  assert term.qualified[1].all()
  env.command[0, 0] = 0.25
  for _ in range(10):
    _step(env, sensor, term, cfg, contact=0, height=0.10)
  assert not term.qualified[0].any()
  _step(env, sensor, term, cfg, contact=1, height=0.10)
  for _ in range(3):
    _step(env, sensor, term, cfg, contact=0, height=0.125)
  assert term.qualified.all()


def test_stopped_contacts_do_not_leak_a_reference_into_restart():
  env, sensor, term, cfg = _setup()
  env.command.zero_()
  _step(env, sensor, term, cfg)
  assert not term.has_reference.any()
  env.command[0, 2] = 0.3
  for _ in range(10):
    _step(env, sensor, term, cfg, contact=0, height=0.1)
  assert not term.qualified.any()


def test_subset_resets_and_inference_lifecycle_preserve_normal_buffer_storage():
  env, sensor, _, cfg = _setup(3)
  with torch.inference_mode():
    term = moving_qualified_swing_occupancy(cfg, env)
    _step(env, sensor, term, cfg)
    for _ in range(5):
      _step(env, sensor, term, cfg, contact=0, height=0.04)
  fields = (
    "occupancy",
    "air_s",
    "contact_height",
    "peak_lift",
    "has_reference",
    "qualified",
  )
  pointers = {name: getattr(term, name).data_ptr() for name in fields}
  assert not any(getattr(term, name).is_inference() for name in fields)
  before = term.occupancy.clone()
  term.reset(torch.tensor([1]))
  torch.testing.assert_close(term.occupancy[[0, 2]], before[[0, 2]])
  assert term.occupancy[1].eq(0.2).all() and not term.qualified[1].any()
  term.reset(slice(0, 1))
  assert not term.has_reference[:2].any() and term.has_reference[2].all()
  term.reset()
  for name in fields:
    assert getattr(term, name).data_ptr() == pointers[name]
    assert getattr(term, name).eq(0.2 if name == "occupancy" else 0).all()
  # Outside inference after both creation and stepping within it.
  _step(env, sensor, term, cfg)


def test_sensor_name_mapping_and_arbitrary_foot_permutation_preserve_cost():
  env, sensor, term, cfg = _setup(2)
  sensor.primary_names.reverse()
  permutation = torch.tensor([2, 0, 3, 1])
  generator = torch.Generator().manual_seed(17)
  _step(env, sensor, term, cfg)
  for _ in range(100):
    canonical = torch.randint(0, 2, (4,), generator=generator)
    height = torch.rand(4, generator=generator) * 0.05
    contacts = torch.stack((canonical, canonical[permutation]))
    heights = torch.stack((height, height[permutation]))
    cost = _step(env, sensor, term, cfg, contact=contacts.flip(1), height=heights)
    torch.testing.assert_close(cost[0], cost[1])
    torch.testing.assert_close(term.occupancy[0, permutation], term.occupancy[1])
    torch.testing.assert_close(term.qualified[0, permutation], term.qualified[1])


def test_synchronous_and_staggered_swings_can_have_equal_cost_without_gait_guarantee():
  env, sensor, term, cfg = _setup(2)
  _step(env, sensor, term, cfg)
  phases = torch.tensor([[0, 0, 0, 0], [0, 5, 10, 15]])
  costs, occupancy = [], []
  for step in range(2000):
    airborne = (step + phases) % 20 < 6  # 4 qualified frames / .4 s period = .20.
    cost = _step(env, sensor, term, cfg, contact=~airborne, height=airborne * 0.03)
    if step >= 1000:
      costs.append(cost.clone())
      occupancy.append(term.qualified.clone())
  means = torch.stack(occupancy).float().mean(dim=0)
  torch.testing.assert_close(means, torch.full((2, 4), 0.2))
  mean_cost = torch.stack(costs).mean(dim=0)
  torch.testing.assert_close(mean_cost[0], mean_cost[1], atol=1e-7, rtol=0)
  assert mean_cost[0] > 0  # EMA ripple remains; phase agreement is not evaluated.


def test_reward_manager_applies_external_negative_weight_and_control_dt_once():
  env, _, _, cfg = _setup(2)
  manager = RewardManager({"qualified_swing": cfg}, env)
  reward = manager.compute(env.step_dt)
  expected = -8 * env.step_dt * (0.2 * (1 - math.exp(-env.step_dt))) ** 2
  torch.testing.assert_close(reward, torch.full((2,), expected), atol=1e-9, rtol=0)
  manager.reset(torch.tensor([1]))
  next_reward = manager.compute(env.step_dt)
  assert next_reward[0] < next_reward[1]
  torch.testing.assert_close(next_reward[1], torch.tensor(expected), atol=1e-9, rtol=0)


@pytest.mark.parametrize("name", ["tau_s", "min_air", "min_lift"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_positive_parameters(name, value):
  with pytest.raises(ValueError, match=f"{name} must be finite and positive"):
    _setup(**{name: value})


@pytest.mark.parametrize("target", [0, 1, -0.1, 1.1, float("nan"), float("inf")])
def test_rejects_invalid_target(target):
  with pytest.raises(ValueError, match="target must be finite"):
    _setup(target=target)


@pytest.mark.parametrize("dt", [0, -0.01, float("nan"), float("inf")])
def test_rejects_invalid_control_dt(dt):
  with pytest.raises(ValueError, match="control dt must be finite"):
    _setup(dt=dt)


@pytest.mark.parametrize(
  "error", ["unnamed", "unordered", "reversed_names", "reversed_ids"]
)
def test_rejects_ambiguous_or_mismatched_site_order(error):
  env, _, _, cfg = _setup()
  sites = cfg.params["asset_cfg"]
  if error == "unnamed":
    sites.site_names = None
  elif error == "unordered":
    sites.preserve_order = False
  elif error == "reversed_names":
    sites.site_names = tuple(reversed(FOOT_SITES))
  else:
    sites.site_ids = [3, 2, 1, 0]
  with pytest.raises(ValueError, match="FOOT_SITES"):
    moving_qualified_swing_occupancy(cfg, env)


@pytest.mark.parametrize(
  "error", ["duplicate", "unknown", "contact_slots", "air_slots"]
)
def test_rejects_ambiguous_contact_sensor_layout(error):
  env, sensor, term, cfg = _setup()
  if error == "duplicate":
    sensor.primary_names[0] = sensor.primary_names[1]
  elif error == "unknown":
    sensor.primary_names[0] = "robot/not_a_foot"
  elif error == "contact_slots":
    sensor.data.found = torch.zeros(1, 8)
  else:
    sensor.data.current_air_time = torch.zeros(1, 8)
  with pytest.raises(ValueError, match="Require"):
    term(env, **cfg.params)
