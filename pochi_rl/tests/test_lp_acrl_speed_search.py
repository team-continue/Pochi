from types import SimpleNamespace

import pytest
import torch

from pochi_rl.mjlab.lp_acrl_commands import _config_identity
from pochi_rl.mjlab.lp_acrl_omni_commands import OMNI_CATALOG
from pochi_rl.mjlab.lp_acrl_speed_commands import (
  MAX_COMMAND,
  SPEED_CATALOG,
  SpeedCatalog,
  SpeedLPACRLCurriculum,
  SpeedPhaseVelocityCommand,
)


def test_catalog_preserves_old_tasks_and_adds_both_signs_to_search_ceilings():
  bounds = SPEED_CATALOG.task_bounds().reshape(22, 10, 3, 2)
  assert SPEED_CATALOG.num_tasks == 220
  assert torch.allclose(bounds[:, :4].reshape(88, 3, 2), OMNI_CATALOG.task_bounds())
  assert torch.equal(bounds.abs().amax((0, 1, 3)), torch.tensor(MAX_COMMAND))
  for index, (_, signs) in enumerate(SPEED_CATALOG.directions()):
    for axis, sign in enumerate(signs):
      if sign == 0:
        assert not bounds[index, :, axis].count_nonzero()
      else:
        assert (sign * bounds[index, :, axis] > 0).all()
  with pytest.raises(ValueError):
    SpeedCatalog(axis_edges=((0.0, 1.0),) * 3)


def test_signed_progress_prefers_improvement_and_retains_all_old_modes():
  lp = SpeedLPACRLCurriculum(220, min_episodes=1)
  ids = torch.arange(220)
  lp.record_episodes(ids, torch.zeros(220))
  lp.end_stage()
  returns = torch.zeros(220)
  returns[9], returns[19] = 40, -40
  lp.record_episodes(ids, returns)
  lp.end_stage()
  assert lp.probabilities[9] > lp.probabilities[0] > lp.probabilities[19]
  reference = SPEED_CATALOG.retention_probabilities().reshape(22, 10)
  assert not reference[:, 4:].count_nonzero()
  names = [name for name, _ in SPEED_CATALOG.directions()]
  for name in set(names):
    assert torch.isclose(
      reference[[n == name for n in names]].sum(), reference.new_tensor(1 / 6)
    )
  effective = lp.effective_probabilities()
  assert torch.isclose(effective.sum(), effective.new_tensor(1.0))
  assert (effective >= 0.2 * reference.flatten()).all()
  restored = SpeedLPACRLCurriculum(220, min_episodes=1)
  restored.load_state_dict(lp.state_dict())
  assert torch.equal(restored.effective_probabilities(), effective)
  with pytest.raises(ValueError):
    restored.load_state_dict(SpeedLPACRLCurriculum(88).state_dict())


def test_speed_resampling_keeps_zero_stop_phase_and_pure_axes():
  cmd = object.__new__(SpeedPhaseVelocityCommand)
  cmd._env = SimpleNamespace(device="cpu", num_envs=220)
  cmd._task_bounds = SPEED_CATALOG.task_bounds()
  cmd.task_ids = torch.arange(220)
  cmd.episode_mode = torch.zeros(220, dtype=torch.long)
  cmd.episode_mode[0] = 1
  cmd.cfg = SimpleNamespace(adaptive=True, transition_interval_s=(4.0, 6.0))
  cmd.vel_command_b = torch.zeros(220, 3)
  cmd.vel_command_w = torch.zeros_like(cmd.vel_command_b)
  cmd._movement_command = torch.zeros_like(cmd.vel_command_b)
  cmd.is_standing_env = torch.zeros(220, dtype=torch.bool)
  cmd._transition_time_left = torch.zeros(220)
  cmd._forced_command = None
  cmd.phase_cycles = torch.zeros(220)
  cmd._resample_command(torch.arange(220))
  assert cmd.phase_cycles[0] == 0 and not cmd.vel_command_b[0].count_nonzero()
  assert not cmd.vel_command_b[20:40, (0, 2)].count_nonzero()
  assert (cmd.vel_command_b.abs() <= torch.tensor(MAX_COMMAND)).all()
  assert cmd.vel_command_b[19, 0] > 3.0


def test_speed_environment_preserves_physics_rewards_observations_and_zero_stop():
  from pochi_rl.mjlab.lp_acrl_blind_v18_env_cfg import pochi_lp_acrl_blind_v18_env_cfg
  from pochi_rl.mjlab.lp_acrl_blind_v19_env_cfg import pochi_lp_acrl_blind_v19_env_cfg
  from pochi_rl.mjlab.lp_acrl_speed_runner import expected_task_space

  old, new = pochi_lp_acrl_blind_v18_env_cfg(), pochi_lp_acrl_blind_v19_env_cfg()
  for field in (
    "scene",
    "sim",
    "rewards",
    "terminations",
    "observations",
    "events",
    "actions",
  ):
    assert _config_identity(getattr(old, field)) == _config_identity(
      getattr(new, field)
    )
  identity = expected_task_space()
  assert identity["axis_edges"] == SPEED_CATALOG.axis_edges
  assert identity["lateral_range"] == (-1.5, 1.5)
  assert "strength_edges" not in identity
  assert tuple(new.observations["actor"].terms)[-1] == "gait_phase"


def test_high_speed_screen_keeps_forward_input_through_all_side_and_yaw_switches(
  monkeypatch,
):
  import importlib
  from pathlib import Path

  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
  screen = importlib.import_module("eval_lp_acrl_speed_search")
  assert len(screen.PROFILES) == 88
  for name, specs in screen.PROFILE_SEGMENTS.items():
    if name.startswith("moving_"):
      assert len({command[0] for _, _, command in specs[1:5]}) == 1
      assert specs[1][2][0] != 0
      assert specs[2][2][1] == -specs[3][2][1]
      assert specs[2][2][2] == -specs[3][2][2]
  assert screen.profile_command("x_+4", 10) == (4.0, 0.0, 0.0)


def test_motor_utilization_uses_directional_curve_not_absolute_speed(monkeypatch):
  import importlib
  from pathlib import Path

  monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
  screen = importlib.import_module("eval_lp_acrl_speed_search")
  robot = SimpleNamespace(
    data=SimpleNamespace(
      qfrc_actuator=torch.zeros(1, 2),
      joint_pos=torch.zeros(1, 2),
      joint_pos_limits=torch.tensor([[[-1.0, 1.0], [-1.0, 1.0]]]),
    ),
    actuators=[
      SimpleNamespace(
        target_ids=torch.tensor([0, 1]),
        _joint_vel_clipped=torch.tensor([[21.45, -21.45]]),
      )
    ],
  )
  result = screen.motor_snapshot(robot, torch.zeros(1, 2))
  assert torch.allclose(result["motor_upper_nm"], torch.tensor([[8.5, 17.0]]))
  assert torch.allclose(result["motor_lower_nm"], torch.tensor([[-17.0, -8.5]]))
