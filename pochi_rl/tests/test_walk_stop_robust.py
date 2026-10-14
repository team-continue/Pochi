"""Hardware-informed training semantics without a live GPU or robot."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pochi_rl.mjlab import torque_margin as margin
from pochi_rl.mjlab.walk_stop_robust import (
    HardwareVelocityObservation, ResponseMotor, configuration, relative_planar_tracking,
    safety_contract, stopped_body_motion, stopped_target_velocity,
)


def env_fixture(command, velocity=None, angular=None):
    command=torch.tensor(command,dtype=torch.float32)
    n=len(command)
    data=SimpleNamespace(root_link_lin_vel_b=torch.zeros(n,3) if velocity is None else torch.tensor(velocity,dtype=torch.float32),
                         root_link_ang_vel_b=torch.zeros(n,3) if angular is None else torch.tensor(angular,dtype=torch.float32))
    return SimpleNamespace(scene={'robot':SimpleNamespace(data=data)},num_envs=n,device='cpu',step_dt=.02,
                           episode_length_buf=torch.zeros(n,dtype=torch.long),
                           command_manager=SimpleNamespace(get_command=lambda name:command),
                           action_manager=SimpleNamespace(action=torch.zeros(n,12),prev_action=torch.zeros(n,12)))


def test_low_speed_tracking_pays_actual_motion_not_stationary_footfalls():
    env=env_fixture([[.1,0,0]]*3,[[0,0,0],[.1,0,0],[.2,0,0]])
    result=relative_planar_tracking(env)
    assert result[1]==1.
    assert result[0]<.02 and result[2]<.02
    torch.testing.assert_close(result[0],result[2])


def test_stop_motion_penalty_does_not_penalize_a_commanded_turn():
    env=env_fixture([[0,0,0],[0,0,.3]],angular=[[0,0,.3],[0,0,.3]])
    torch.testing.assert_close(stopped_body_motion(env),torch.tensor([1.,0.]))
    env.action_manager.action.fill_(.08)
    cost=stopped_target_velocity(env)
    assert cost[0]>0 and cost[1]==0


def test_velocity_proxy_caps_acceleration_and_does_not_update_twice_per_tick():
    env=env_fixture([[.1,0,0]])
    observation=HardwareVelocityObservation(SimpleNamespace(params={'randomize_bias':False}),env)
    torch.testing.assert_close(observation(env),torch.zeros(1,3))
    env.episode_length_buf+=1
    env.scene['robot'].data.root_link_lin_vel_b[:,0]=1.
    first=observation(env).clone()
    assert first[0,0]==pytest.approx(.06)
    torch.testing.assert_close(first,observation(env))
    observation.reset()
    torch.testing.assert_close(observation(env),torch.tensor([[1.,0.,0.]]))


def test_velocity_bias_is_persistent_within_episode_and_resets_by_environment():
    env=env_fixture([[0,0,0]]*3)
    observation=HardwareVelocityObservation(SimpleNamespace(params={}),env)
    observation.reset(); before=observation.bias.clone()
    assert (before.abs()<=.03).all()
    observation.reset(torch.tensor([1]))
    torch.testing.assert_close(observation.bias[[0,2]],before[[0,2]])
    assert not torch.equal(observation.bias[1],before[1])


def test_motor_response_records_unclipped_demand_once_then_respects_envelope():
    motor=ResponseMotor.__new__(ResponseMotor)
    motor.saturation_effort=torch.full((2,1),16.)
    motor.force_limit=torch.full((2,1),16.)
    motor.velocity_limit_motor=torch.full((2,1),42.9)
    motor._vel_at_effort_lim=torch.full((2,1),85.8)
    motor._joint_vel_clipped=torch.zeros(2,1)
    motor.peak_request=torch.zeros(2,1); motor.peak_delta=torch.zeros(2,1)
    motor.previous_request=torch.zeros(2,1);motor.previous_valid=torch.zeros(2,1,dtype=torch.bool)
    motor.filtered_effort=torch.zeros(2,1);motor.response_alpha=torch.full((2,1),.25)
    first=motor._clip_effort(torch.tensor([[30.],[-30.]]))
    torch.testing.assert_close(first,torch.tensor([[4.],[-4.]]))
    assert motor.peak_request.max()==30 and motor.peak_delta.max()==0
    motor._clip_effort(torch.tensor([[-30.],[30.]]))
    assert motor.peak_delta.max()==60
    assert motor.filtered_effort.abs().max()<=16


def test_configuration_is_opt_in_and_preserves_observation_and_action_semantics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    baseline,_=margin.configuration(8,42)
    cfg,_=configuration(8,42)
    actuator=cfg.scene.entities['robot'].articulation.actuators[0]
    assert baseline.actions['joint_pos'].torque_limit_nm==8
    assert cfg.actions['joint_pos'].torque_limit_nm==16
    assert cfg.actions['joint_pos'].target_rate_rad_s==6
    assert actuator.effort_limit==16 and actuator.delay_max_lag==8
    assert (actuator.stiffness,actuator.damping)==(140,1.5)
    assert tuple(cfg.observations['actor'].terms)==tuple(baseline.observations['actor'].terms)
    assert cfg.observations['critic'].terms['base_lin_vel'].func is baseline.observations['critic'].terms['base_lin_vel'].func
    assert cfg.observations['actor'].terms['base_lin_vel'].func is HardwareVelocityObservation
    assert cfg.commands['base_velocity'].standing_episode_fraction==.25
    assert cfg.commands['base_velocity'].transition_episode_fraction==.35
    assert margin.safety_contract()['effort_limit_nm']==8
    assert safety_contract()['required_target_guard']['walk_estimated_torque_limit_nm']==16


def test_nominal_video_does_not_claim_randomized_dynamics(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts'))
    nominal,_=configuration(8,42,play=True)
    robust,_=configuration(8,42,play=True,robust_play=True)
    assert nominal.scene.entities['robot'].articulation.actuators[0].delay_max_lag==0
    assert robust.scene.entities['robot'].articulation.actuators[0].delay_max_lag==8
    assert 'encoder_offsets' not in nominal.events and 'encoder_offsets' in robust.events
    assert nominal.observations['actor'].terms['base_lin_vel'].params['randomize_bias'] is False
