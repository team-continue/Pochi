"""Opt-in hardware-informed V19 walk/stop refinement; policy stays 50 -> 12.

Targets use the actual runtime 16 Nm / 6 rad/s guard. Dynamics uncertainty
addresses the 5.5 Hz stopped yaw loop; reward uses true velocity, while only
the actor receives a smoothed, biased velocity proxy. No terrain scan is added.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from mjlab.actuator.dc_actuator import DcMotorActuator
from mjlab.envs.mdp.dr import encoder_bias, effort_limits, joint_damping, joint_friction
from mjlab.managers import EventTermCfg, RewardTermCfg, SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg

from pochi_rl.mjlab import torque_margin as margin
from pochi_rl.mjlab.lp_acrl_phase_gait import (
    FREQUENCY_HZ, foot_targets, moving, phase_foot_lift_error,
)
from pochi_rl.robot import FOOT_GEOMS, JOINT_NAMES

TORQUE_LIMIT = 16.0
TARGET_RATE = 6.0
SOFT_THRESHOLD = 12.0
PROTOCOL = {**margin.PROTOCOL, 'version': 3,
            'standing_episode_fraction': .25, 'transition_episode_fraction': .35,
            'transition_interval_s': (2.0, 4.0),
            'auxiliary_task_sampling': 'equal directions/signs, bands 3..6 of the low-speed catalog'}


def exact_stop(env):
    return env.command_manager.get_command('base_velocity').abs().amax(-1) <= 1e-6


def relative_planar_tracking(env):
    command = env.command_manager.get_command('base_velocity')[:, :2]
    actual = env.scene['robot'].data.root_link_lin_vel_b[:, :2]
    # A 0.10 m/s request has a 0.05 m/s tracking width; stationary footfalls
    # cannot collect most of the tracking reward merely because speed is low.
    width = torch.maximum(command.norm(dim=-1)*.5, command.new_full((len(command),), .04))
    error = (actual-command).square().sum(-1)/width.square()
    return torch.exp(-error) * (~exact_stop(env))


def stopped_body_motion(env):
    data = env.scene['robot'].data
    linear = (data.root_link_lin_vel_b/.1).square().sum(-1)
    angular = (data.root_link_ang_vel_b/.3).square().sum(-1)
    return (linear+angular)*exact_stop(env)


def stopped_target_velocity(env):
    action = env.action_manager.action
    previous = env.action_manager.prev_action
    velocity = (action-previous)*.25/env.step_dt
    return velocity.square().sum(-1)*exact_stop(env)


def requested_excess(env):
    return (margin._action(env).raw_request.abs()-SOFT_THRESHOLD).clamp_min(0).square().sum(-1)


def peak_excess(env):
    return (margin._action(env).motor.peak_request-SOFT_THRESHOLD).clamp_min(0).square().sum(-1)


class speed_scaled_foot_lift(phase_foot_lift_error):
    def __call__(self, env, asset_cfg):
        sensor=env.scene['feet_ground_contact']
        names=[name.rsplit('/',1)[-1] for name in sensor.primary_names]
        order=[names.index(name) for name in FOOT_GEOMS]
        contact=sensor.data.found[:,order]>0
        z=env.scene[asset_cfg.name].data.site_pos_w[:,asset_cfg.site_ids,2]
        self.reference.copy_(torch.where(contact,z,self.reference))
        self.has_reference |= contact
        term=env.command_manager.get_term('base_velocity')
        command=term.vel_command_b
        height,_=foot_targets((term.phase_cycles+FREQUENCY_HZ*env.step_dt)%1.)
        speed=torch.maximum(command[:,:2].norm(dim=-1),command[:,2].abs()*.2)
        # 2 cm at very low speed, 3 cm at 0.10 m/s, 5 cm above 0.30 m/s.
        scale=(.4+speed*2.).clamp(.4,1.)
        height=height*scale[:,None]
        error=((z-self.reference-height)/.025).square().clamp_max(9.)
        return (error*self.has_reference).mean(-1)*moving(command)


class HardwareVelocityObservation:
    """4 Hz, 3 m/s² proxy with per-episode bias; update once per policy tick."""
    def __init__(self, cfg, env):
        with torch.inference_mode(False):
            self.value=torch.zeros((env.num_envs,3),device=env.device)
            self.bias=torch.zeros_like(self.value)
            self.valid=torch.zeros(env.num_envs,dtype=torch.bool,device=env.device)
            self.tick=torch.full((env.num_envs,),-1,dtype=torch.long,device=env.device)
        self.randomize=bool(cfg.params.get('randomize_bias',True))

    def reset(self, env_ids=None):
        ids=slice(None) if env_ids is None else env_ids
        self.value[ids]=0.;self.valid[ids]=False;self.tick[ids]=-1
        if self.randomize:
            self.bias[ids]=torch.empty_like(self.bias[ids]).uniform_(-.03,.03)
        else:self.bias[ids]=0.

    def __call__(self, env, randomize_bias=True):
        del randomize_bias
        actual=env.scene['robot'].data.root_link_lin_vel_b
        tick=env.episode_length_buf
        update=(tick!=self.tick)|(~self.valid)
        initial=torch.where(self.valid[:,None],self.value,actual)
        delta=-math.expm1(-2*math.pi*4*env.step_dt)*(actual-initial)
        magnitude=delta.norm(dim=-1,keepdim=True)
        delta=delta*(3.*env.step_dt/magnitude.clamp_min(1e-12)).clamp_max(1.)
        self.value.copy_(torch.where(update[:,None],initial+delta,self.value))
        self.tick.copy_(torch.where(update,tick,self.tick));self.valid |= update
        return self.value+self.bias


@dataclass(kw_only=True)
class ResponseMotorCfg(margin.PeakDcMotorCfg):
    reaction_time_max_s: float = .015
    physics_dt_s: float = .0025

    def build(self, entity, target_ids, target_names):
        return ResponseMotor(self,entity,target_ids,target_names)


class ResponseMotor(margin.PeakDcMotor):
    def initialize(self,*args,**kwargs):
        super().initialize(*args,**kwargs)
        self.filtered_effort=torch.zeros_like(self.stiffness)
        self.response_alpha=torch.ones_like(self.stiffness)

    def reset(self,env_ids=None):
        super().reset(env_ids)
        ids=slice(None) if env_ids is None else env_ids
        self.filtered_effort[ids]=0.
        tau=torch.empty_like(self.response_alpha[ids]).uniform_(0.,self.cfg.reaction_time_max_s)
        self.response_alpha[ids]=-torch.expm1(-self.cfg.physics_dt_s/tau.clamp_min(1e-9))

    def _clip_effort(self,effort):
        clipped=super()._clip_effort(effort)  # Peak/delta before clipping/filtering.
        self.filtered_effort.add_(self.response_alpha*(clipped-self.filtered_effort))
        # Reapply the torque-speed envelope after filtering, without recording
        # a second fictitious PD request in the substep peak/delta buffers.
        return DcMotorActuator._clip_effort(self,self.filtered_effort)


class StopVelocityCommand(margin.MarginVelocityCommand):
    def command_protocol_identity(self):
        return PROTOCOL

    def prepare_episode(self,env_ids):
        super().prepare_episode(env_ids)
        if not self._evaluation:
            ids=env_ids[self.episode_mode[env_ids]!=0]
            if len(ids):
                direction=torch.randint(22,(len(ids),),device=self.device)
                band=torch.randint(3,7,(len(ids),),device=self.device)
                self.task_ids[ids]=direction*10+band


@dataclass(kw_only=True)
class StopVelocityCfg(margin.MarginVelocityCfg):
    def build(self,env):
        self.__post_init__()
        return StopVelocityCommand(self,env)


def configuration(num_envs,seed,*,play=False,robust_play=False):
    cfg,agent=margin.configuration(num_envs,seed,play=play)
    uncertain=not play or robust_play
    actuator=cfg.scene.entities['robot'].articulation.actuators[0]
    cfg.scene.entities['robot'].articulation.actuators=(ResponseMotorCfg(**{
        **vars(actuator),'effort_limit':TORQUE_LIMIT,'saturation_effort':TORQUE_LIMIT,
        'delay_min_lag':0,'delay_max_lag':8 if uncertain else 0,
        'delay_update_period':400 if uncertain else 0,
        'reaction_time_max_s':.015 if uncertain else 0.,
    }),)
    cfg.actions['joint_pos']=margin.GuardedPositionCfg(**{
        **vars(cfg.actions['joint_pos']),'torque_limit_nm':TORQUE_LIMIT,'target_rate_rad_s':TARGET_RATE})
    cfg.commands['base_velocity']=StopVelocityCfg(**{
        **vars(cfg.commands['base_velocity']),'standing_episode_fraction':.25,
        'transition_episode_fraction':.35,'transition_interval_s':(2.,4.)})
    cfg.observations['actor'].terms['base_lin_vel'].func=HardwareVelocityObservation
    cfg.observations['actor'].terms['base_lin_vel'].params={'randomize_bias':uncertain}
    cfg.observations['actor'].terms['base_lin_vel'].noise=(UniformNoiseCfg(n_min=-.06,n_max=.06)
                                                        if not play else None)
    for name,weight in {
        'upright_error':-4.,'ang_vel_xy_l2':-.35,'termination':-400.,
        'joint_vel_l2':-.004,'standing_posture':2.,'standing_joint_velocity':-.45,
        'standing_missing_contacts':-4.,'phase_foot_lift_error':-2.,
        'moving_qualified_swing_occupancy':-3.,'signed_translation_progress':1.,
        'requested_pd_l2':-.00008,'substep_request_delta':-.0002,
    }.items():cfg.rewards[name].weight=weight
    cfg.rewards['phase_foot_lift_error'].func=speed_scaled_foot_lift
    cfg.rewards['moving_qualified_swing_occupancy'].params['min_lift']=.015
    cfg.rewards['valid_foot_cycle_age'].params['min_lift']=.015
    cfg.rewards['requested_pd_excess']=RewardTermCfg(func=requested_excess,weight=-.001)
    cfg.rewards['substep_peak_excess']=RewardTermCfg(func=peak_excess,weight=-.002)
    cfg.rewards.update({
        'relative_planar_tracking':RewardTermCfg(func=relative_planar_tracking,weight=2.),
        'stopped_body_motion':RewardTermCfg(func=stopped_body_motion,weight=-.2),
        'stopped_target_velocity':RewardTermCfg(func=stopped_target_velocity,weight=-.08),
    })
    cfg.terminations['fell_over'].params['limit_angle']=math.radians(45.)
    if uncertain:
        cfg.events['reset_joints'].params['position_range']=(-.12,.12)
        cfg.events['reset_root'].params['pose_range'].update(roll=(-.06,.06),pitch=(-.06,.06))
        joints=SceneEntityCfg('robot',joint_names=JOINT_NAMES,preserve_order=True)
        cfg.events.update({
            'encoder_offsets':EventTermCfg(func=encoder_bias,mode='reset',
                params={'bias_range':(-.025,.025),'asset_cfg':joints}),
            'joint_resistance':EventTermCfg(func=joint_friction,mode='startup',
                params={'ranges':(.025,.18),'asset_cfg':joints}),
            'joint_viscous':EventTermCfg(func=joint_damping,mode='startup',
                params={'ranges':(.005,.035),'asset_cfg':joints}),
            'motor_strength':EventTermCfg(func=effort_limits,mode='startup',
                params={'effort_limit_range':(.85,1.),'operation':'scale'}),
        })
    return cfg,agent


def safety_contract():
    return {**margin.safety_contract(),
        'effort_limit_nm':TORQUE_LIMIT,'saturation_effort_nm':TORQUE_LIMIT,
        'required_target_guard':{
            'walk_estimated_torque_limit_nm':TORQUE_LIMIT,'walk_target_rate_limit_rad_s':TARGET_RATE,
            'priority':'torque over slew; hardware aborts infeasible travel intersection'},
        'soft_torque_penalty_threshold_nm':SOFT_THRESHOLD,'command_protocol':PROTOCOL,
        'simulated_dynamics':{'target_delay_s':(0.,.020),'motor_reaction_s':(0.,.015),
                             'strength_fraction':(.85,1.),'encoder_offset_rad':(-.025,.025)},
        'actor_velocity_proxy':{'cutoff_hz':4.,'max_acceleration_m_s2':3.,'bias_m_s':(-.03,.03)},
        'fall_termination_deg':45.,'training_initial_joint_offset_rad':(-.12,.12),
        'training_initial_roll_pitch_rad':(-.06,.06)}
