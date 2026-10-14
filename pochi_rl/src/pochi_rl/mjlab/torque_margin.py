"""Opt-in low-speed V19 refinement with guarded targets and pre-limit costs.

The policy still observes its raw last action. Limits act on position targets,
in the encoder frame, exactly once per 50 Hz control step. Physics-step costs
measure the unbounded PD request after target filtering, before motor clipping.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from mjlab.actuator.dc_actuator import DcMotorActuator, DcMotorActuatorCfg
from mjlab.envs.mdp.actions.actions import JointPositionAction, JointPositionActionCfg
from mjlab.managers.reward_manager import RewardTermCfg

from pochi_rl.mjlab.lp_acrl_speed_commands import (
    COMMAND_CONTRACT, SpeedCatalog, SpeedPhaseVelocityCommand,
    SpeedPhaseVelocityCommandCfg,
)

TORQUE_LIMIT = 8.0
TARGET_RATE = 3.0
SOFT_TORQUE_THRESHOLD = 6.0
AXIS_EDGES = (
    (.001, .01, .02, .04, .06, .10, .16, .24, .32, .40, .50),
    (.0004, .003, .005, .01, .025, .04, .08, .12, .18, .24, .30),
    (.005, .02, .05, .10, .15, .20, .30, .40, .45, .50, .60),
)
CATALOG = SpeedCatalog(axis_edges=AXIS_EDGES)
PROTOCOL = {
    **COMMAND_CONTRACT,
    "version": 2,
    "max_abs_vx_vy_wz": (.50, .30, .60),
    "axis_edges": AXIS_EDGES,
    "fixed_episode_sampling": "0.8 signed-LP + 0.2 first-four low-speed bands",
    "auxiliary_task_sampling": "equal mode/sign/first four low-speed bands",
}


def project_targets(desired, q, qd, previous, limits, *, kp=140.0, kd=1.5,
                    torque_limit=TORQUE_LIMIT, rate=TARGET_RATE, dt=.02):
    """Return targets, slew overrides and infeasible joint/torque intersections.

    Torque wins over slew. If even the joint-travel intersection is infeasible,
    use the nearest travel bound; an explicit cost records that infeasibility.
    Hardware refuses this state, whereas simulation must remain vectorized.
    """
    center = q + kd * qd / kp
    low = torch.maximum(limits[..., 0], center - torque_limit / kp)
    high = torch.minimum(limits[..., 1], center + torque_limit / kp)
    infeasible = low > high
    travel_fallback = torch.clamp(center, min=limits[..., 0], max=limits[..., 1])
    low = torch.where(infeasible, travel_fallback, low)
    high = torch.where(infeasible, travel_fallback, high)
    slew_low = torch.maximum(low, previous - rate * dt)
    slew_high = torch.minimum(high, previous + rate * dt)
    override = slew_low > slew_high
    low = torch.where(override, low, slew_low)
    high = torch.where(override, high, slew_high)
    return torch.clamp(desired, min=low, max=high), override, infeasible


@dataclass(kw_only=True)
class PeakDcMotorCfg(DcMotorActuatorCfg):
    def build(self, entity, target_ids, target_names):
        return PeakDcMotor(self, entity, target_ids, target_names)


class PeakDcMotor(DcMotorActuator):
    def initialize(self, *args, **kwargs):
        super().initialize(*args, **kwargs)
        self.peak_request = torch.zeros_like(self.stiffness)
        self.peak_delta = torch.zeros_like(self.stiffness)
        self.previous_request = torch.zeros_like(self.stiffness)
        self.previous_valid = torch.zeros_like(self.stiffness, dtype=torch.bool)

    def _clip_effort(self, effort):
        # In-place buffers also accumulate during captured CUDA graph replay.
        self.peak_request.copy_(torch.maximum(self.peak_request, effort.abs()))
        delta = torch.where(self.previous_valid, (effort-self.previous_request).abs(), 0.)
        self.peak_delta.copy_(torch.maximum(self.peak_delta, delta))
        self.previous_request.copy_(effort)
        self.previous_valid.fill_(True)
        return super()._clip_effort(effort)

    def reset(self, env_ids=None):
        super().reset(env_ids)
        ids = slice(None) if env_ids is None else env_ids
        for value in (self.peak_request, self.peak_delta, self.previous_request):
            value[ids] = 0.
        self.previous_valid[ids] = False

    def begin_control_step(self):
        self.peak_request.zero_()
        self.peak_delta.zero_()


@dataclass(kw_only=True)
class GuardedPositionCfg(JointPositionActionCfg):
    torque_limit_nm: float = TORQUE_LIMIT
    target_rate_rad_s: float = TARGET_RATE

    def build(self, env):
        return GuardedPosition(self, env)


class GuardedPosition(JointPositionAction):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.previous_target = self._offset.clone()
        self.raw_request = torch.zeros_like(self._raw_actions)
        self.slew_override = torch.zeros_like(self._raw_actions, dtype=torch.bool)
        self.infeasible = torch.zeros_like(self.slew_override)
        self.dt = env.step_dt
        self.motor = self._entity.actuators[0]
        assert isinstance(self.motor, PeakDcMotor)
        assert tuple(self.motor._target_names) == tuple(self._target_names)

    def process_actions(self, actions):
        super().process_actions(actions)
        data = self._entity.data
        bias = data.encoder_bias[:, self._target_ids]
        q = data.joint_pos[:, self._target_ids] + bias
        qd = data.joint_vel[:, self._target_ids]
        kp, kd = self.motor.stiffness, self.motor.damping
        self.raw_request.copy_(kp * (self._processed_actions-q) - kd * qd)
        limits = data.soft_joint_pos_limits[:, self._target_ids] + bias.unsqueeze(-1)
        target, override, infeasible = project_targets(
            self._processed_actions, q, qd, self.previous_target, limits,
            kp=kp, kd=kd, torque_limit=self.cfg.torque_limit_nm,
            rate=self.cfg.target_rate_rad_s, dt=self.dt,
        )
        self._processed_actions = target
        self.previous_target.copy_(target)
        self.slew_override.copy_(override)
        self.infeasible.copy_(infeasible)
        self.motor.begin_control_step()

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        super().reset(ids)
        data = self._entity.data
        measured = data.joint_pos[:, self._target_ids] + data.encoder_bias[:, self._target_ids]
        self.previous_target[ids] = measured[ids]
        self.raw_request[ids] = 0.
        self.slew_override[ids] = False
        self.infeasible[ids] = False


def _action(env):
    return env.action_manager.get_term("joint_pos")


def requested_pd_l2(env):
    return _action(env).raw_request.square().sum(-1)


def requested_pd_excess(env):
    return (_action(env).raw_request.abs()-SOFT_TORQUE_THRESHOLD).clamp_min(0).square().sum(-1)


def substep_peak_excess(env):
    return (_action(env).motor.peak_request-SOFT_TORQUE_THRESHOLD).clamp_min(0).square().sum(-1)


def substep_request_delta(env):
    return _action(env).motor.peak_delta.square().sum(-1)


def guard_override(env):
    action = _action(env)
    return action.slew_override.float().sum(-1) + 10*action.infeasible.float().sum(-1)


class MarginVelocityCommand(SpeedPhaseVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.catalog = CATALOG
        self._task_indices = CATALOG.task_indices(self.device)
        self._task_bounds = CATALOG.task_bounds(self.device)

    def command_protocol_identity(self):
        return PROTOCOL

    def checkpoint_identity(self):
        return {**super().checkpoint_identity(), "axis_edges": AXIS_EDGES,
                "lateral_range": (-.3, .3), "command_protocol": PROTOCOL}


class MarginVelocityCfg(SpeedPhaseVelocityCommandCfg):
    def build(self, env):
        self.__post_init__()
        return MarginVelocityCommand(self, env)


def configuration(num_envs, seed, *, play=False):
    # Import from the standalone ablation launcher to retain its proven noise
    # and regularization configuration. Entrypoints add scripts/ to sys.path.
    import v19_hardware_ablation as base
    cfg, agent = base.configuration("noise_penalties", num_envs, seed, play=play)
    articulation = cfg.scene.entities["robot"].articulation
    assert len(articulation.actuators) == 1
    actuator = articulation.actuators[0]
    articulation.actuators = (PeakDcMotorCfg(**{
        **vars(actuator), "effort_limit": TORQUE_LIMIT, "saturation_effort": TORQUE_LIMIT,
    }),)
    cfg.actions["joint_pos"] = GuardedPositionCfg(**vars(cfg.actions["joint_pos"]))
    command = MarginVelocityCfg(**vars(cfg.commands["base_velocity"]))
    command.ranges.lin_vel_x = (-.5, .5)
    command.ranges.lin_vel_y = (-.3, .3)
    command.ranges.ang_vel_z = (-.6, .6)
    cfg.commands["base_velocity"] = command
    # Resolve contact/PD peaks at 400 Hz, retaining the policy's 50 Hz cadence.
    cfg.decimation = 8
    cfg.sim.mujoco.timestep = .0025
    for sensor in cfg.scene.sensors:
        if getattr(sensor, "history_length", None) == 4:
            sensor.history_length = cfg.decimation
    cfg.rewards.update({
        "requested_pd_l2": RewardTermCfg(func=requested_pd_l2, weight=-.0002),
        "requested_pd_excess": RewardTermCfg(func=requested_pd_excess, weight=-.005),
        "substep_peak_excess": RewardTermCfg(func=substep_peak_excess, weight=-.005),
        "substep_request_delta": RewardTermCfg(func=substep_request_delta, weight=-.0005),
        "guard_override": RewardTermCfg(func=guard_override, weight=-.02),
    })
    return cfg, agent


def safety_contract():
    return {
        "effort_limit_nm": TORQUE_LIMIT, "saturation_effort_nm": TORQUE_LIMIT,
        "kp": 140., "kd": 1.5, "policy_hz": 50, "physics_hz": 400,
        "required_target_guard": {
            "walk_estimated_torque_limit_nm": TORQUE_LIMIT,
            "walk_target_rate_limit_rad_s": TARGET_RATE,
            "priority": "torque over slew; hardware aborts infeasible travel intersection",
        },
        "soft_torque_penalty_threshold_nm": SOFT_TORQUE_THRESHOLD,
        "last_action_semantics": "raw clipped policy action, before target guard",
        "command_protocol": PROTOCOL,
        "scope": "Torque is a PD estimate; electrical current peaks are not simulated directly",
    }
