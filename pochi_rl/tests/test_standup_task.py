"""The learned stand-up task: safety envelope, start pose, and env sanity."""

from __future__ import annotations

import numpy as np
import pytest

from pochi_rl.control.standup import COLLAPSED_BASE_HEIGHT, COLLAPSED_JOINT_POS
from pochi_rl.robot import (
  JOINT_NAMES,
  NOMINAL_BASE_HEIGHT,
  RS02_NO_LOAD_SPEED_RAD_S,
  RS02_RATED_TORQUE_NM,
)
from pochi_rl.task_spec import POCHI_STANDUP_SPEC as S

# Joint limits from assets/pochi/pochi.xml.
LIMITS = {"hip_roll": 0.7, "hip_pitch": 1.6, "knee": 2.4}


def test_collapsed_pose_matches_the_simulated_collapse() -> None:
  """The hard-coded start pose is still what cutting the motors produces.

  The RL env poses thousands of robots per reset and cannot settle each one, so
  the collapsed pose is written down as a constant.  This is what stops that
  constant drifting away from the model it came from.
  """
  pytest.importorskip("mujoco")
  from pochi_rl.control.mujoco_driver import StandUpSim

  sim = StandUpSim()
  assert sim.base_height == pytest.approx(COLLAPSED_BASE_HEIGHT, abs=0.005)
  for name, expected in COLLAPSED_JOINT_POS.items():
    got = sim.joint_pos[JOINT_NAMES.index(name)]
    assert got == pytest.approx(expected, abs=0.01), name


def test_collapsed_pose_is_inside_the_joint_limits() -> None:
  for name, value in COLLAPSED_JOINT_POS.items():
    limit = LIMITS[name.split("_", 1)[1]]
    assert abs(value) <= limit, f"{name} = {value} outside +/-{limit}"


def test_safety_envelope_is_actually_restrictive() -> None:
  """The point of the task: a far smaller motor envelope than the RS02's."""
  assert S.safety.motor_speed_limit < 0.1 * RS02_NO_LOAD_SPEED_RAD_S
  assert S.safety.motor_effort_limit <= RS02_RATED_TORQUE_NM
  # The reward starts charging for speed well inside the hard cap, so the
  # policy never learns to sit on it.
  assert S.safety.soft_speed_limit < S.safety.motor_speed_limit


def test_slow_actuator_uses_the_safety_envelope() -> None:
  pytest.importorskip("mjlab")
  from pochi_rl.mjlab.entity_cfg import POCHI_ACTUATOR, POCHI_SLOW_ACTUATOR

  assert POCHI_SLOW_ACTUATOR.velocity_limit == S.safety.motor_speed_limit
  assert POCHI_SLOW_ACTUATOR.effort_limit == S.safety.motor_effort_limit
  # Everything else has to match the velocity task, or the two policies are
  # controlling different robots.
  assert POCHI_SLOW_ACTUATOR.stiffness == POCHI_ACTUATOR.stiffness
  assert POCHI_SLOW_ACTUATOR.damping == POCHI_ACTUATOR.damping
  assert POCHI_SLOW_ACTUATOR.armature == POCHI_ACTUATOR.armature
  assert POCHI_SLOW_ACTUATOR.saturation_effort == POCHI_ACTUATOR.saturation_effort


def test_task_is_registered() -> None:
  pytest.importorskip("mjlab")
  from mjlab.tasks.registry import list_tasks

  import pochi_rl  # noqa: F401

  assert "Pochi-StandUp-Flat-v0" in list_tasks()


def test_observation_layout_matches_the_velocity_task() -> None:
  """The two policies should see the same robot, differing only in the command.

  The velocity task is told a velocity to track; the stand-up task is told a
  height to be at and how far through the ramp it is.  Everything else is
  shared, so a policy hand-off does not need a different sensor stack.
  """
  pytest.importorskip("mjlab")
  from pochi_rl.mjlab.standup_env_cfg import _observation_terms
  from pochi_rl.mjlab.velocity_flat_env_cfg import (
    _observation_terms as velocity_terms,
  )
  from pochi_rl.task_spec import POCHI_TASK_SPEC

  standup = set(_observation_terms(add_noise=True))
  velocity = set(velocity_terms(add_noise=True))
  assert velocity - standup == {"velocity_commands"}
  assert standup - velocity == {"standup_reference"}
  assert S.policy_dim == POCHI_TASK_SPEC.observations.policy_dim - 3 + 2


def test_reference_ramp_starts_collapsed_and_ends_standing() -> None:
  from pochi_rl.control.standup import COLLAPSED_BASE_HEIGHT, reference_height

  settle, rise = S.reference.settle_s, S.reference.rise_s
  assert reference_height(0.0, settle, rise) == pytest.approx(COLLAPSED_BASE_HEIGHT)
  assert reference_height(settle, settle, rise) == pytest.approx(COLLAPSED_BASE_HEIGHT)
  assert reference_height(settle + rise, settle, rise) == pytest.approx(
    NOMINAL_BASE_HEIGHT
  )
  # Monotone, and slow enough that the rise never needs the safe motor envelope
  # to be anywhere near saturated.
  times = np.linspace(0.0, settle + rise + 2.0, 400)
  ref = np.array([reference_height(t, settle, rise) for t in times])
  assert np.all(np.diff(ref) >= -1e-12)
  assert np.max(np.abs(np.gradient(ref, times))) < 0.1  # m/s


def test_torch_reference_matches_the_scripted_one() -> None:
  """The reward's ramp and the scripted controller's are the same curve."""
  pytest.importorskip("mjlab")
  import torch

  from pochi_rl.control.standup import reference_height
  from pochi_rl.mjlab.rewards import standup_reference_height

  settle, rise = S.reference.settle_s, S.reference.rise_s
  dt = S.control.sim_dt * S.control.decimation
  steps = torch.arange(0, 500)

  class _FakeEnv:
    episode_length_buf = steps
    step_dt = dt

  got = standup_reference_height(_FakeEnv(), settle, rise).numpy()  # type: ignore[arg-type]
  want = np.array([reference_height(int(s) * dt, settle, rise) for s in steps])
  assert np.allclose(got, want, atol=1e-6)


def test_posture_penalty_is_off_while_the_robot_is_still_down() -> None:
  """The one shaping detail the manoeuvre lives or dies on.

  Ungated, the posture penalty bills a robot lying on the floor roughly as much
  per step as tracking the ramp is worth, and the policy learns to leap to the
  standing pose and wait for the reference to catch up.
  """
  pytest.importorskip("mjlab")
  import torch

  from pochi_rl.mjlab.rewards import standup_phase

  settle, rise = S.reference.settle_s, S.reference.rise_s
  dt = S.control.sim_dt * S.control.decimation

  class _FakeEnv:
    step_dt = dt

    def __init__(self, seconds: float) -> None:
      self.episode_length_buf = torch.tensor([int(seconds / dt)])

  assert float(standup_phase(_FakeEnv(0.0), settle, rise)) == 0.0
  assert float(standup_phase(_FakeEnv(settle), settle, rise)) == pytest.approx(0.0)
  assert float(standup_phase(_FakeEnv(settle + rise), settle, rise)) == pytest.approx(
    1.0, abs=0.01
  )
  # And it stays clamped after the ramp finishes, rather than growing.
  assert float(standup_phase(_FakeEnv(settle + rise + 5.0), settle, rise)) == 1.0


def _env(num_envs: int = 8):
  pytest.importorskip("mjlab")
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.tasks.registry import load_env_cfg

  import pochi_rl  # noqa: F401

  cfg = load_env_cfg("Pochi-StandUp-Flat-v0")
  cfg.scene.num_envs = num_envs
  cfg.events.pop("base_mass", None)
  return ManagerBasedRlEnv(cfg=cfg, device="cpu")


def test_env_resets_the_robot_onto_the_floor() -> None:
  env = _env()
  try:
    env.reset()
    height = env.scene["robot"].data.root_link_pos_w[:, 2]
    lo = COLLAPSED_BASE_HEIGHT + S.reset.base_height[0]
    hi = COLLAPSED_BASE_HEIGHT + S.reset.base_height[1]
    assert float(height.min()) >= lo - 0.02
    assert float(height.max()) <= hi + 0.02
  finally:
    env.close()


def test_env_random_rollout_nan_free() -> None:
  import torch

  env = _env(4)
  try:
    obs, _ = env.reset()
    assert obs["actor"].shape[1] == S.policy_dim
    for _ in range(50):
      actions = 2.0 * torch.rand(env.action_space.shape) - 1.0
      obs, reward, terminated, truncated, _ = env.step(actions)
      for value in (obs["actor"], obs["critic"], reward, terminated, truncated):
        assert torch.isfinite(value.float()).all()
  finally:
    env.close()


def test_scripted_manoeuvre_fits_inside_the_safety_envelope() -> None:
  """The behaviour the learned policy is being asked to reproduce is legal.

  If the scripted stand-up needed more speed or torque than the safe actuator
  can deliver, the task would be unlearnable by construction.
  """
  pytest.importorskip("mujoco")
  from pochi_rl.control.mujoco_driver import StandUpSim

  sim = StandUpSim()
  peak_speed = 0.0
  peak_torque = 0.0
  steps = int((sim.controller.cfg.total_duration_s + 2.0) / sim.model.opt.timestep)
  for _ in range(steps):
    sim.step()
    peak_speed = max(peak_speed, float(np.abs(sim.data.qvel[6:]).max()))
    peak_torque = max(peak_torque, float(np.abs(sim.data.actuator_force).max()))
  assert peak_speed < S.safety.soft_speed_limit
  assert peak_torque < S.safety.motor_effort_limit
