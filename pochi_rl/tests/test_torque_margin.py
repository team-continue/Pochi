"""Safety projection and peak-accounting tests independent of GPU hardware."""
from pathlib import Path

import torch

from pochi_rl.mjlab.torque_margin import (
    PeakDcMotor, configuration, project_targets, safety_contract,
)


def test_projection_limits_pd_and_slew_for_feasible_states():
    generator = torch.Generator().manual_seed(13)
    q = torch.randn(128, 12, generator=generator) * .25
    qd = torch.randn(128, 12, generator=generator)
    center = q + 1.5*qd/140
    desired = torch.randn(128, 12, generator=generator)
    previous = center + .01
    limits = torch.tensor([-3., 3.]).expand(128, 12, 2)
    target, override, invalid = project_targets(desired, q, qd, previous, limits)
    assert not override.any() and not invalid.any()
    assert (140*(target-q)-1.5*qd).abs().max() <= 8.00002
    assert (target-previous).abs().max() <= .060001


def test_torque_takes_priority_over_infeasible_slew():
    q = torch.tensor([[1.]])
    target, override, invalid = project_targets(
        torch.tensor([[2.]]), q, torch.zeros_like(q), torch.zeros_like(q),
        torch.tensor([[[-3., 3.]]]),
    )
    assert override.all() and not invalid.any()
    torch.testing.assert_close(target, q+8/140)


def test_infeasible_travel_intersection_is_explicit():
    q = torch.tensor([[2.]])
    target, _, invalid = project_targets(
        q, q, torch.zeros_like(q), q, torch.tensor([[[-1., 1.]]]),
    )
    assert invalid.all()
    assert target.item() == 1.


def motor_fixture():
    motor = PeakDcMotor.__new__(PeakDcMotor)
    motor.saturation_effort = torch.full((2, 1), 8.)
    motor.force_limit = torch.full((2, 1), 8.)
    motor.velocity_limit_motor = torch.full((2, 1), 25.)
    motor._vel_at_effort_lim = torch.full((2, 1), 50.)
    motor._joint_vel_clipped = torch.zeros(2, 1)
    motor.peak_request = torch.zeros(2, 1)
    motor.peak_delta = torch.zeros(2, 1)
    motor.previous_request = torch.zeros(2, 1)
    motor.previous_valid = torch.zeros(2, 1, dtype=torch.bool)
    motor._delay_buffer = None
    return motor


def test_peak_is_before_clipping_and_keeps_boundary_delta():
    motor = motor_fixture()
    applied = motor._clip_effort(torch.tensor([[27.], [-12.]]))
    assert applied.abs().max() == 8.
    torch.testing.assert_close(motor.peak_request, torch.tensor([[27.], [12.]]))
    assert not motor.peak_delta.any()
    motor._clip_effort(torch.tensor([[-10.], [-8.]]))
    torch.testing.assert_close(motor.peak_delta, torch.tensor([[37.], [4.]]))
    motor.begin_control_step()
    motor._clip_effort(torch.tensor([[6.], [2.]]))
    torch.testing.assert_close(motor.peak_request, torch.tensor([[6.], [2.]]))
    torch.testing.assert_close(motor.peak_delta, torch.tensor([[16.], [10.]]))
    motor.reset(torch.tensor([0]))
    assert not motor.previous_valid[0].any()
    assert motor.previous_valid[1].all()


def test_configuration_is_opt_in_and_preserves_policy_interface(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/"scripts"))
    import v19_hardware_ablation as base
    original, _ = base.configuration("noise_penalties", 16, 42)
    cfg, _ = configuration(16, 42)
    old_motor = original.scene.entities["robot"].articulation.actuators[0]
    motor = cfg.scene.entities["robot"].articulation.actuators[0]
    assert old_motor.effort_limit == 17.
    assert motor.effort_limit == motor.saturation_effort == 8.
    assert (motor.stiffness, motor.damping) == (140., 1.5)
    assert tuple(cfg.observations['actor'].terms) == tuple(original.observations['actor'].terms)
    assert cfg.actions['joint_pos'].actuator_names == original.actions['joint_pos'].actuator_names
    assert cfg.observations['actor'].enable_corruption
    assert not cfg.observations['critic'].enable_corruption
    assert cfg.sim.mujoco.timestep*cfg.decimation == .02
    assert cfg.commands['base_velocity'].ranges.lin_vel_x == (-.5, .5)
    assert safety_contract()['last_action_semantics'].startswith('raw clipped')
