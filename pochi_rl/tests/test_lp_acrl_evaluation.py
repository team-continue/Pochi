"""First-episode evaluation, timeout accounting, and false locomotion guards."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "eval_lp_acrl.py"
_SPEC = importlib.util.spec_from_file_location("eval_lp_acrl", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_EVAL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EVAL)


class FakeCommand:
  def __init__(self, count):
    from pochi_rl.lp_acrl import FLAT_TASK_CATALOG

    self.catalog = FLAT_TASK_CATALOG
    self.command = torch.zeros(count, 3)
    self.command[:, 0] = 0.1
    self.command[-1, 0] = 0.01

  def set_evaluation(self, task_ids):
    self.task_ids = task_ids.clone()


class FakeRobot:
  def __init__(self, count):
    self.data = SimpleNamespace(
      root_link_pos_w=torch.zeros(count, 3),
      root_link_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]).repeat(count, 1),
      joint_pos=torch.zeros(count, 12),
      joint_vel=torch.zeros(count, 12),
      joint_acc=torch.zeros(count, 12),
      root_link_lin_vel_b=torch.zeros(count, 3),
      root_link_ang_vel_b=torch.zeros(count, 3),
      site_lin_vel_w=torch.zeros(count, 4, 3),
    )

  def find_sites(self, names, preserve_order):
    assert preserve_order
    return [0, 1, 2, 3], names


class FakeScene(dict):
  """Minimal scene mapping with the real environment's origin attribute."""


class FakeEnv:
  """Expose reset corruption and a sensor order unrelated to config order."""

  def __init__(self, fall_at=2):
    self.cfg = SimpleNamespace(auto_reset=False)
    self.num_envs = 5
    self.device = "cpu"
    self.step_dt = 0.02
    self.max_episode_length = 1000
    self.fall_at = fall_at
    self.command = FakeCommand(self.num_envs)
    self.command_manager = SimpleNamespace(get_term=lambda _: self.command)
    self.scene = FakeScene(
      {
        "robot": FakeRobot(self.num_envs),
        "feet_ground_contact": SimpleNamespace(
          data=SimpleNamespace(found=torch.zeros(self.num_envs, 4)),
        ),
        "limb_ground_contact": SimpleNamespace(
          primary_names=["RR_shank_collision", "base_collision", "FL_thigh_collision"],
          data=SimpleNamespace(found=torch.zeros(self.num_envs, 3)),
        ),
      }
    )
    self.scene.env_origins = torch.zeros(self.num_envs, 3)
    self.counter = torch.zeros(self.num_envs, dtype=torch.long)
    self.reset_counts = torch.zeros_like(self.counter)
    self.pending = torch.zeros(self.num_envs, dtype=torch.bool)
    self.reset_terminated = torch.zeros_like(self.pending)
    self.reset_time_outs = torch.zeros_like(self.pending)
    self.base_contact_steps = torch.tensor([0, 0, 50, 51, 0])

  def reset(self, env_ids=None):
    if env_ids is None:
      env_ids = torch.arange(self.num_envs)
    self.pending[env_ids] = False
    self.counter[env_ids] = 0
    self.reset_counts[env_ids] += 1
    data = self.scene["robot"].data
    data.root_link_pos_w[env_ids] = self.scene.env_origins[env_ids]
    # The evaluator must never sample these post-reset values.
    data.root_link_lin_vel_b[env_ids] = 99
    self.scene["limb_ground_contact"].data.found[env_ids] = 1


class FakeWrapped:
  def __init__(self, env):
    self.env = env

  def get_observations(self):
    return torch.zeros(self.env.num_envs, 1)

  def step(self, actions):
    env = self.env
    assert not env.pending.any(), "Must reset every done env before stepping"
    assert actions.shape == (env.num_envs, 12)
    env.counter += 1
    env.reset_terminated = torch.zeros_like(env.pending)
    env.reset_terminated[0] = env.counter[0] >= env.fall_at
    env.reset_time_outs = env.counter >= env.max_episode_length
    env.pending = env.reset_terminated | env.reset_time_outs
    data = env.scene["robot"].data
    data.root_link_lin_vel_b[:] = env.command.command
    data.root_link_lin_vel_b[-1] = 0  # Near-zero command can pass the paper metric.
    data.root_link_pos_w += data.root_link_lin_vel_b * env.step_dt
    # A new episode in env 0 has completely different dynamics. It is padding
    # and must contribute neither reward, contacts nor velocity to the report.
    data.root_link_lin_vel_b[env.reset_counts > 1] = 99
    feet = env.scene["feet_ground_contact"].data.found
    feet[:] = ((env.counter % 20) >= 6)[:, None]
    feet[-1] = 1  # Motionless robot keeps all feet planted.
    nonfoot = env.scene["limb_ground_contact"].data.found
    nonfoot.zero_()
    nonfoot[:, 1] = env.counter <= env.base_contact_steps
    nonfoot[1, 2] = env.counter[1] % 10 == 0
    nonfoot[env.reset_counts > 1] = 1
    reward = torch.ones(env.num_envs)
    reward[env.reset_counts > 1] = 999
    return self.get_observations(), reward, env.pending.long(), {}


def run_evaluation(env):
  return _EVAL.evaluate_batch(
    env,
    FakeWrapped(env),
    lambda _: torch.zeros(env.num_envs, 12),
    torch.arange(env.num_envs),
  )


def test_consecutive_batches_reset_sensor_buffer_created_in_inference_mode():
  class SensorBufferEnv(FakeEnv):
    def reset(self, env_ids=None):
      self.scene["feet_ground_contact"].data.found.zero_()
      return super().reset(env_ids)

  env = SensorBufferEnv()
  env.max_episode_length = 1
  wrapped = FakeWrapped(env)
  ordinary_step = wrapped.step

  def allocate_sensor_buffer_during_step(actions):
    sensor = env.scene["feet_ground_contact"]
    if not torch.is_inference(sensor.data.found):
      sensor.data.found = torch.zeros_like(sensor.data.found)
    return ordinary_step(actions)

  wrapped.step = allocate_sensor_buffer_during_step
  first = _EVAL.evaluate_batch(
    env, wrapped, lambda _: torch.zeros(env.num_envs, 12), torch.arange(env.num_envs)
  )
  sensor_buffer = env.scene["feet_ground_contact"].data.found
  assert torch.is_inference(sensor_buffer)
  second = _EVAL.evaluate_batch(
    env, wrapped, lambda _: torch.zeros(env.num_envs, 12), torch.arange(env.num_envs)
  )
  assert len(first) == len(second) == env.num_envs
  assert env.scene["feet_ground_contact"].data.found is sensor_buffer


@pytest.fixture(scope="module")
def evaluated():
  env = FakeEnv()
  return env, run_evaluation(env)


def test_first_fall_stops_accounting_before_reset(evaluated):
  env, rows = evaluated
  first = rows[0]
  assert env.reset_counts[0] > 1  # The inactive env did run more episodes.
  assert first["alive_steps"] == 1
  assert first["episode_steps"] == 2
  assert first["episodic_reward"] == 2.0
  assert first["epte_sp_forward"] == pytest.approx(0.999)
  assert first["actual_mean_vx_vy_wz"][0] == pytest.approx(0.1)
  assert first["any_nonfoot_contact_fraction"] == 0.0
  assert first["base_contact_fraction"] == 0.0
  assert first["fall"] and not first["timeout"]
  assert not first["horizon_reached"]
  assert not first["paper_style_success"]


def test_timeout_reaches_horizon_and_healthy_walking_is_accepted(evaluated):
  _, rows = evaluated
  healthy = rows[1]
  assert healthy["episode_steps"] == 1000
  assert healthy["alive_steps"] == 1000
  assert healthy["timeout"] and not healthy["fall"]
  assert healthy["horizon_reached"]
  assert healthy["epte_sp_forward"] == 0.0
  assert healthy["paper_style_success"]
  assert healthy["accepted_locomotion"]
  assert healthy["footfalls_per_second"] > 0
  # A thigh touching does not accidentally get interpreted as a base contact.
  assert healthy["any_nonfoot_contact_fraction"] == pytest.approx(0.1)
  assert healthy["base_contact_fraction"] == 0.0
  assert healthy["nonfoot_contact_fraction_by_geom"]["FL_thigh_collision"] == (
    pytest.approx(0.1)
  )


def test_belly_support_guard_is_separate_from_paper_success(evaluated):
  _, rows = evaluated
  boundary, excessive = rows[2:4]
  assert boundary["base_contact_fraction"] == pytest.approx(0.05)
  assert boundary["base_support_guard_passed"]
  assert boundary["accepted_locomotion"]
  assert excessive["base_contact_fraction"] == pytest.approx(0.051)
  assert excessive["paper_style_success"]
  assert excessive["gait_observed"] and excessive["motion_observed"]
  assert not excessive["base_support_guard_passed"]
  assert not excessive["accepted_locomotion"]


def test_near_zero_command_does_not_accept_motionless_robot(evaluated):
  _, rows = evaluated
  stationary = rows[4]
  assert stationary["epte_sp_forward"] == pytest.approx(0.2, abs=1e-5)
  assert stationary["paper_style_success"]
  assert stationary["actual_mean_vx_vy_wz"] == [0.0, 0.0, 0.0]
  assert stationary["footfalls_per_second"] == 0.0
  assert not stationary["motion_observed"]
  assert not stationary["gait_observed"]
  assert not stationary["accepted_locomotion"]


def test_summary_keeps_paper_and_locomotion_success_separate(evaluated):
  _, rows = evaluated
  summary = _EVAL.summarize(rows)
  assert summary["episodes"] == 5
  assert summary["falls"] == 1
  assert summary["timeouts"] == 4
  assert summary["horizon_reached_episodes"] == 4
  assert summary["base_support_guard_rejected_episodes"] == 1
  assert summary["paper_style_success_rate"] == 0.8
  assert summary["accepted_locomotion_rate"] == 0.4
  assert summary["mean_base_contact_fraction"] == pytest.approx(0.0202)


def test_flat_terrain_has_no_central_platform_diagnostic(evaluated):
  _, rows = evaluated
  assert all(not row["central_platform_applicable"] for row in rows)
  assert all(row["central_platform_outside_base_fraction"] is None for row in rows)
  assert all(row["ever_left_platform"] is None for row in rows)
  summary = _EVAL.summarize(rows)
  assert summary["central_platform_measured_episodes"] == 0
  assert summary["mean_outside_central_platform_base_fraction"] is None
  assert summary["ever_left_platform_rate"] is None


def test_platform_coverage_uses_tile_origins_and_only_valid_first_episode_steps():
  from pochi_rl.lp_acrl import ROUGH_TASK_CATALOG

  env = FakeEnv()
  env.max_episode_length = 5
  env.command.catalog = ROUGH_TASK_CATALOG
  env.scene.env_origins[:] = torch.tensor(
    [
      [100.0, -50.0, 0.0],
      [-36.0, 48.0, 0.0],
      [24.0, 72.0, 0.0],
      [120.0, -48.0, 0.0],
      [-72.0, -120.0, 0.0],
    ]
  )
  wrapped = FakeWrapped(env)
  ordinary_step = wrapped.step
  step_number = 0

  def place_base_and_inject_invalid(actions):
    nonlocal step_number
    result = ordinary_step(actions)
    step_number += 1
    data = env.scene["robot"].data
    data.root_link_pos_w[:] = env.scene.env_origins
    # Ascending stairs leaves only on its fall transition and after reset.
    data.root_link_pos_w[0, 0] += 0.7 if step_number >= 2 else 0.0
    # Descending stairs leaves twice, including the final timeout sample.
    data.root_link_pos_w[1, 1] += 0.7 if step_number in (3, 5) else 0.0
    # Random rough terrain has no center platform, regardless of position.
    data.root_link_pos_w[2, 0] += 2.0
    # Uphill leaves only on the invalid transition and inactive padding.
    data.root_link_pos_w[3, 0] -= 0.7 if step_number >= 3 else 0.0
    if step_number == 3:
      data.joint_vel[3, 0] = 201.0
    return result

  wrapped.step = place_base_and_inject_invalid
  rows = _EVAL.evaluate_batch(
    env,
    wrapped,
    lambda _: torch.zeros(env.num_envs, 12),
    torch.tensor([0, 4, 8, 12, 16]),
  )
  assert [row["terrain"] for row in rows] == list(ROUGH_TASK_CATALOG.terrains)
  assert rows[0]["alive_steps"] == 1 and env.reset_counts[0] > 1
  assert rows[0]["central_platform_outside_base_fraction"] == 0.0
  assert rows[0]["ever_left_platform"] is False
  assert rows[1]["central_platform_outside_base_fraction"] == pytest.approx(0.4)
  assert rows[1]["ever_left_platform"] is True
  assert rows[1]["timeout"]
  assert rows[2]["central_platform_applicable"] is False
  assert rows[2]["central_platform_outside_base_fraction"] is None
  assert rows[2]["ever_left_platform"] is None
  assert rows[3]["invalid_state"] and rows[3]["alive_steps"] == 2
  assert rows[3]["central_platform_outside_base_fraction"] == 0.0
  assert rows[3]["ever_left_platform"] is False
  assert rows[4]["central_platform_outside_base_fraction"] == 0.0
  summary = _EVAL.summarize(rows)
  assert summary["central_platform_applicable_episodes"] == 4
  assert summary["central_platform_measured_episodes"] == 4
  assert summary["mean_outside_central_platform_base_fraction"] == pytest.approx(0.1)
  assert summary["ever_left_platform_rate"] == 0.25
  per_terrain = {row["terrain"]: row for row in _EVAL.summarize_by_terrain(rows)}
  assert per_terrain["descending_stairs"]["ever_left_platform_rate"] == 1.0
  assert per_terrain["rough"]["ever_left_platform_rate"] is None
  json.dumps({"episodes": rows, "by_terrain": per_terrain}, allow_nan=False)


def test_autoreset_is_rejected_before_measuring():
  env = FakeEnv()
  env.cfg.auto_reset = True
  with pytest.raises(ValueError, match="auto_reset=False"):
    run_evaluation(env)


def test_supported_checkpoint_rewards_restore_exactly():
  from mjlab.envs import mdp
  from mjlab.managers.reward_manager import RewardTermCfg

  from pochi_rl.mjlab.lp_acrl_commands import _config_identity
  from pochi_rl.mjlab.lp_acrl_env_cfg import pochi_lp_acrl_flat_env_cfg

  saved_cfg = pochi_lp_acrl_flat_env_cfg()
  saved_cfg.rewards["track_lin_vel_xy_exp"].params["std"] = 0.15
  saved_cfg.rewards["track_ang_vel_z_exp"].params["std"] = 0.2
  saved_cfg.rewards["termination"] = RewardTermCfg(
    func=mdp.is_terminated, weight=-200.0
  )
  saved_rewards = _config_identity(saved_cfg.rewards)
  restored = pochi_lp_acrl_flat_env_cfg(play=True)
  _EVAL.restore_checkpoint_rewards(restored, saved_rewards)
  assert _config_identity(restored.rewards) == saved_rewards


def test_legacy_rough_checkpoint_omits_new_physics_assertion():
  from pochi_rl.mjlab.lp_acrl_commands import _config_identity
  from pochi_rl.mjlab.lp_acrl_env_cfg import pochi_lp_acrl_rough_env_cfg

  saved_cfg = pochi_lp_acrl_rough_env_cfg()
  saved_cfg.rewards.pop("physics_validity")
  saved_rewards = _config_identity(saved_cfg.rewards)
  restored = pochi_lp_acrl_rough_env_cfg(play=True)
  assert "physics_validity" in restored.rewards
  _EVAL.restore_checkpoint_rewards(restored, saved_rewards)
  assert "physics_validity" not in restored.rewards
  assert _config_identity(restored.rewards) == saved_rewards


def test_new_rough_checkpoint_retains_exact_physics_assertion():
  from pochi_rl.mjlab.lp_acrl_commands import _config_identity
  from pochi_rl.mjlab.lp_acrl_env_cfg import pochi_lp_acrl_rough_env_cfg

  saved_rewards = _config_identity(pochi_lp_acrl_rough_env_cfg().rewards)
  restored = pochi_lp_acrl_rough_env_cfg(play=True)
  _EVAL.restore_checkpoint_rewards(restored, saved_rewards)
  assert "physics_validity" in restored.rewards
  assert _config_identity(restored.rewards) == saved_rewards


def test_modified_physics_assertion_is_not_a_supported_reward_override():
  from pochi_rl.mjlab.lp_acrl_commands import _config_identity
  from pochi_rl.mjlab.lp_acrl_env_cfg import pochi_lp_acrl_rough_env_cfg

  saved_cfg = pochi_lp_acrl_rough_env_cfg()
  saved_cfg.rewards["physics_validity"].weight = 2.0
  with pytest.raises(ValueError, match="cannot report comparable episodic returns"):
    _EVAL.restore_checkpoint_rewards(
      pochi_lp_acrl_rough_env_cfg(play=True), _config_identity(saved_cfg.rewards)
    )


@pytest.mark.parametrize("changed", ["weight", "function", "parameter"])
def test_unsupported_checkpoint_reward_changes_cannot_silently_change_returns(changed):
  from pochi_rl.mjlab.lp_acrl_commands import _config_identity
  from pochi_rl.mjlab.lp_acrl_env_cfg import pochi_lp_acrl_flat_env_cfg
  from pochi_rl.mjlab.lp_acrl_rewards import track_yaw_velocity

  saved_cfg = pochi_lp_acrl_flat_env_cfg()
  reward = saved_cfg.rewards["track_lin_vel_xy_exp"]
  if changed == "weight":
    reward.weight = 2.0
  elif changed == "function":
    reward.func = track_yaw_velocity
  else:
    reward.params["command_name"] = "another_command"
  saved_rewards = _config_identity(saved_cfg.rewards)
  with pytest.raises(ValueError, match="cannot report comparable episodic returns"):
    _EVAL.restore_checkpoint_rewards(
      pochi_lp_acrl_flat_env_cfg(play=True), saved_rewards
    )


@pytest.mark.parametrize(
  "field", ["root_link_pos_w", "root_link_quat_w", "joint_pos", "joint_vel"]
)
def test_nonfinite_physics_cannot_pass_or_prevent_json_report(field):
  env = FakeEnv()
  wrapped = FakeWrapped(env)
  ordinary_step = wrapped.step
  injected = False

  def inject_nonfinite(actions):
    nonlocal injected
    result = ordinary_step(actions)
    if not injected and env.counter[1] == 950:
      getattr(env.scene["robot"].data, field)[1, 0] = float("nan")
      injected = True
    return result

  wrapped.step = inject_nonfinite
  rows = _EVAL.evaluate_batch(
    env, wrapped, lambda _: torch.zeros(env.num_envs, 12), torch.arange(env.num_envs)
  )
  # This was an otherwise healthy, moving, accurate episode >900 steps long.
  # Unchecked pose/joint NaNs previously still received paper/locomotion success.
  invalid = rows[1]
  assert invalid["invalid_state"]
  assert invalid["alive_steps"] == 949
  assert invalid["episode_steps"] == 950
  assert not invalid["paper_style_success"]
  assert not invalid["accepted_locomotion"]
  assert not invalid["horizon_reached"]
  assert invalid["actual_mean_vx_vy_wz"][0] == pytest.approx(0.1, abs=2e-6)
  if field == "root_link_pos_w":
    assert invalid["displacement_xy_m"] is None
  json.dumps({"episodes": rows, "summary": _EVAL.summarize(rows)}, allow_nan=False)


@pytest.mark.parametrize(
  ("field", "value"),
  [
    ("joint_vel", 201.0),
    ("joint_acc", 1_000_001.0),
    ("root_link_lin_vel_b", 21.0),
    ("root_link_ang_vel_b", 51.0),
    ("root_link_lin_vel_b", 1.0e38),
  ],
)
def test_impossible_finite_physics_fails_late_episode_without_reward_credit(
  field, value
):
  env = FakeEnv()
  wrapped = FakeWrapped(env)
  ordinary_step = wrapped.step
  injected = False

  def inject_impossible_state(actions):
    nonlocal injected
    result = ordinary_step(actions)
    if not injected and env.counter[1] == 950:
      getattr(env.scene["robot"].data, field)[1, 0] = value
      # A finite corrupted reward must not dominate a failed evaluation.
      result[1][1] = 1.0e38
      injected = True
    return result

  wrapped.step = inject_impossible_state
  rows = _EVAL.evaluate_batch(
    env, wrapped, lambda _: torch.zeros(env.num_envs, 12), torch.arange(env.num_envs)
  )
  invalid = rows[1]
  assert invalid["invalid_state"]
  assert invalid["alive_steps"] == 949
  assert invalid["episode_steps"] == 950
  assert invalid["episodic_reward"] == 949.0
  assert not invalid["paper_style_success"]
  assert not invalid["accepted_locomotion"]
  assert not invalid["horizon_reached"]
  assert invalid["actual_mean_vx_vy_wz"][0] == pytest.approx(0.1, abs=2e-6)
  json.dumps({"episodes": rows, "summary": _EVAL.summarize(rows)}, allow_nan=False)
