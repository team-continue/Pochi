"""Behavioral criteria, narrow identity overrides, and terminal sample integrity."""

import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_lp_acrl_behavior.py"
_SPEC = importlib.util.spec_from_file_location("eval_lp_acrl_behavior", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_EVAL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EVAL)


def identity():
  return {
    "forward_edges": (0, 0.1, 0.2),
    "environment": {
      "physics": {"dt": 0.0025},
      "actions": {"scale": 0.25},
      "observations": {"policy": ("velocity", "command")},
      "robot_articulation": {"kp": 140},
      "events": {"mass": (-0.5, 0.5)},
      "terrain_type": "generator",
      "terrain_generator": {
        "num_rows": 4,
        "num_cols": 5,
        "sub_terrains": {
          "rough": {"type": "OriginalRough", "size": (24, 24), "proportion": 1.0}
        },
      },
      "rewards": {"tracking": 1.0},
    },
  }


def flatten(source):
  current = deepcopy(source)
  current["environment"]["terrain_generator"]["sub_terrains"]["rough"]["type"] = (
    "mjlab.terrains.primitive_terrains.BoxFlatTerrainCfg"
  )
  current["environment"]["rewards"] = {"new_diagnostic_rewards": 3.0}
  return current


def test_identity_allows_only_documented_overrides_without_mutation():
  saved = identity()
  current = flatten(saved)
  before = deepcopy((saved, current))
  result = _EVAL.validate_behavior_identity(saved, current)
  assert result["terrain_subfamilies"]["saved"]["rough"]["type"] == "OriginalRough"
  assert result["rewards"]["saved"] != result["rewards"]["diagnostic"]
  assert (saved, current) == before


@pytest.mark.parametrize(
  "field", ["physics", "observations", "actions", "events", "robot_articulation"]
)
def test_identity_rejects_changed_nonterrain_contract(field):
  saved = identity()
  current = flatten(saved)
  current["environment"][field] = {"changed": True}
  with pytest.raises(ValueError, match="identity mismatch"):
    _EVAL.validate_behavior_identity(saved, current)


@pytest.mark.parametrize(
  "field,value", [("type", "OtherTerrain"), ("size", (2, 2)), ("proportion", 2.0)]
)
def test_identity_rejects_unapproved_flat_geometry_parameters(field, value):
  saved = identity()
  current = flatten(saved)
  current["environment"]["terrain_generator"]["sub_terrains"]["rough"][field] = value
  with pytest.raises(ValueError):
    _EVAL.validate_behavior_identity(saved, current)


def test_stop_start_boundaries_are_exact():
  assert [
    _EVAL.profile_command("stop_start", t)[0]
    for t in (0, 4.98, 5, 9.98, 10, 14.98, 15, 19.98)
  ] == [0, 0, 0.25, 0.25, 0, 0, 0.25, 0.25]


def test_short_swing_count_and_bout_censoring():
  result = _EVAL.contact_bouts(np.array([1, 1, 0, 0, 1, 0, 1, 0, 0, 0]), 0.02)
  assert result["footfalls"] == 1  # .04s swing counted; .02s and final air not.
  assert result["contact"][0]["left_censored"]
  assert result["air"][-1]["right_censored"]
  assert result["air"][0]["duration_s"] == pytest.approx(0.04)


def samples(count=1000, worlds=4):
  return {
    "time_s": np.arange(1, count + 1) * 0.02,
    "valid": np.ones((count, worlds), bool),
    "velocity": np.zeros((count, worlds, 3)),
    "position": np.tile([0.0, 0.0, 0.317], (count, worlds, 1)),
    "contact": np.ones((count, worlds, 4), bool),
    "foot_vertical_force": np.full((count, worlds, 4), 26.8),
    "nonfoot_contact": np.zeros((count, worlds), bool),
    "slip": np.zeros((count, worlds)),
    "tilt_deg": np.zeros((count, worlds)),
    "joint_error": np.zeros((count, worlds, 12)),
    "joint_velocity": np.zeros((count, worlds, 12)),
  }


def test_still_four_leg_stand_passes_but_diagonal_support_fails():
  data = samples()
  result = _EVAL.summarize_segment(data, 0, 0, 20, (0, 0, 0), 0.02, settle_s=2)
  assert result["passed"] and result["valid_samples"] == 900
  data["contact"][:, 0, 1:3] = False
  result = _EVAL.summarize_segment(data, 0, 0, 20, (0, 0, 0), 0.02, settle_s=2)
  assert not result["passed"]
  assert not result["checks"]["all_four_contact"]


def test_zero_command_rejects_persistent_rotation_even_without_xy_drift():
  data = samples()
  data["velocity"][:, 0, 2] = 0.06
  result = _EVAL.summarize_segment(data, 0, 0, 20, (0, 0, 0), 0.02, settle_s=2)
  assert not result["passed"]
  assert not result["checks"]["yaw_rate_absolute_mean_rad_s"]


def test_tracking_with_diagonal_lameness_fails_balanced_walking():
  data = samples()
  data["velocity"][:, 0, 0] = 0.25
  data["position"][:, 0, 0] = data["time_s"] * 0.25
  phase = np.arange(1000) % 10 < 2
  data["contact"][:, 0, 0] = ~phase
  data["contact"][:, 0, 3] = ~phase
  data["contact"][:, 0, 1] = phase
  data["contact"][:, 0, 2] = phase
  result = _EVAL.summarize_segment(data, 0, 0, 20, (0.25, 0, 0), 0.02, settle_s=2)
  assert result["checks"]["forward_error_mean_m_s"]
  assert result["checks"]["all_legs_footfalls"]
  assert not result["checks"]["contact_duty_balance"]
  assert not result["passed"]


def test_short_balanced_swings_pass_and_terminal_loss_cannot_be_hidden():
  data = samples()
  data["velocity"][:, 0, 0] = 0.25
  pattern = np.arange(1000) % 10 < 5
  data["contact"][:, 0] = np.stack([pattern, ~pattern, ~pattern, pattern], axis=-1)
  result = _EVAL.summarize_segment(data, 0, 0, 20, (0.25, 0, 0), 0.02, settle_s=2)
  assert result["passed"]
  data["valid"][950:, 0] = False
  data["velocity"][950:, 0] = np.nan
  result = _EVAL.summarize_segment(data, 0, 0, 20, (0.25, 0, 0), 0.02, settle_s=2)
  assert not result["complete"] and not result["passed"]
  assert np.isfinite(result["metrics"]["actual_mean_vx_vy_wz"]).all()


def test_command_refresh_uses_uncached_group_computation():
  class Manager:
    def compute_group(self, name, update_history):
      assert name == "policy" and not update_history
      return torch.full((4, 3), 0.25)

    def compute(self):
      raise AssertionError("cached path would return stale commands")

  env = SimpleNamespace(
    num_envs=4,
    cfg=SimpleNamespace(observations={"policy": None}),
    observation_manager=Manager(),
  )
  obs = _EVAL.fresh_command_observations(env)
  assert torch.equal(obs["policy"], torch.full((4, 3), 0.25))


class FakeBehaviorEnv:
  def __init__(self):
    from pochi_rl.robot import FOOT_GEOMS

    self.num_envs = 4
    self.step_dt = 0.02
    self.max_episode_length = 1000
    self.device = "cpu"
    self.cfg = SimpleNamespace(auto_reset=False, observations={"policy": None})
    self.steps = 0
    self.reset_ids = []
    command = SimpleNamespace(
      vel_command_b=torch.zeros(4, 3),
      vel_command_w=torch.zeros(4, 3),
    )
    command.set_evaluation = lambda **kwargs: None
    self.command_manager = SimpleNamespace(get_term=lambda _: command)
    self.command = command
    data = SimpleNamespace(
      root_link_pos_w=torch.tensor([[0.0, 0.0, 0.317]]).repeat(4, 1),
      root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(4, 1),
      root_link_lin_vel_b=torch.zeros(4, 3),
      root_link_ang_vel_b=torch.zeros(4, 3),
      joint_pos=torch.zeros(4, 12),
      default_joint_pos=torch.zeros(4, 12),
      joint_vel=torch.zeros(4, 12),
      joint_acc=torch.zeros(4, 12),
      site_lin_vel_w=torch.zeros(4, 4, 3),
    )
    self.scene = {
      "robot": SimpleNamespace(
        data=data,
        find_sites=lambda names, preserve_order: ([0, 1, 2, 3], names),
      ),
      "feet_ground_contact": SimpleNamespace(
        primary_names=FOOT_GEOMS,
        data=SimpleNamespace(found=torch.ones(4, 4), force=torch.zeros(4, 4, 3)),
      ),
      "limb_ground_contact": SimpleNamespace(
        data=SimpleNamespace(found=torch.zeros(4, 9))
      ),
    }
    self.scene["feet_ground_contact"].data.force[:, :, 2] = 26.8
    self.observation_manager = SimpleNamespace(
      compute_group=lambda name, update_history: command.vel_command_b.clone()
    )
    self.reset_terminated = torch.zeros(4, dtype=torch.bool)

  def reset(self, env_ids=None):
    if env_ids is not None:
      self.reset_ids.extend(env_ids.tolist())
      # A later episode deliberately corrupts world0's location. Its samples
      # must never contribute to the first episode's post-transient drift.
      self.scene["robot"].data.root_link_pos_w[env_ids, 0] = 1000
      self.scene["robot"].data.joint_vel[env_ids] = 0

  def step(self, actions):
    self.steps += 1
    robot = self.scene["robot"].data
    robot.root_link_lin_vel_b[:] = actions
    robot.root_link_pos_w += robot.root_link_lin_vel_b * self.step_dt
    contact = self.scene["feet_ground_contact"].data.found
    for i in range(4):
      if abs(float(actions[i, 0])) > 0.01:
        phase = self.steps % 10 < 5
        contact[i] = torch.tensor([phase, not phase, not phase, phase])
      else:
        contact[i] = 1
    self.reset_terminated[:] = False
    if self.steps == 250:
      self.reset_terminated[0] = True
    if self.steps == 950:
      robot.joint_vel[1, 0] = 201
    done = self.reset_terminated.clone()
    if self.steps == 1000:
      done[:] = True
    return None, torch.zeros(4), done, {}


def test_actual_loop_uses_first_episode_and_current_stop_start_commands():
  env = FakeBehaviorEnv()
  commands = []

  def policy(observations):
    command = observations["policy"].clone()
    commands.append(command.clone())
    return command

  rows = _EVAL.evaluate_profiles(env, env, policy)
  assert rows[0]["fall"] and not rows[0]["passed"]
  assert rows[0]["segments"][0]["valid_samples"] == 149
  assert rows[0]["segments"][0]["metrics"]["horizontal_drift_max_m"] == 0
  assert rows[1]["invalid_state"] and not rows[1]["passed"]
  assert rows[1]["segments"][0]["valid_samples"] == 849
  assert rows[2]["passed"] and rows[3]["passed"]
  assert [
    float(commands[step][3, 0]) for step in (0, 249, 250, 499, 500, 749, 750)
  ] == [0, 0, 0.25, 0.25, 0, 0, 0.25]
  assert env.reset_ids[:2] == [0, 1]
