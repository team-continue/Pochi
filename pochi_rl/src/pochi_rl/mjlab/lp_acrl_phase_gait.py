"""Pochi extension: 1.5 Hz diagonal gait and continuous five-centimetre lift."""

from __future__ import annotations

import math

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import GaussianDistribution

from pochi_rl.mjlab.lp_acrl_commands import (
  LPACRLVelocityCommand,
  LPACRLVelocityCommandCfg,
)
from pochi_rl.mjlab.lp_acrl_quiet_stop import QuietStopMLPModel
from pochi_rl.mjlab.lp_acrl_symmetry import (
  mirror_blind_observations,
  mirror_joint_vectors,
)
from pochi_rl.robot import FOOT_GEOMS

FREQUENCY_HZ = 1.5
PEAK_LIFT_M = 0.05
SWING_FRACTION = 0.40
PHASE_CONTRACT = {
  "version": 1,
  "frequency_hz": FREQUENCY_HZ,
  "peak_lift_above_last_contact_m": PEAK_LIFT_M,
  "swing_fraction": SWING_FRACTION,
  "foot_order": ("FL", "FR", "RL", "RR"),
  "cycle_offsets": (0.0, 0.5, 0.5, 0.0),
  "standing": "all raw commands zero: clock reset and phase observation zero",
  "observation": "48 body inputs followed by sin/cos; no terrain scan",
  "reward": "continuous negative height/contact tracking costs; no touchdown bonus",
  "training_reset": "random phase on moving episode resets; seeded RNG",
  "play_reset": "phase zero",
}


def moving(command):
  return command.abs().amax(dim=-1) > 1.0e-6


def foot_targets(cycles: torch.Tensor):
  phase = (cycles[:, None] + cycles.new_tensor([0.0, 0.5, 0.5, 0.0])) % 1.0
  swing = phase < SWING_FRACTION
  height = PEAK_LIFT_M * torch.sin(math.pi * phase / SWING_FRACTION)
  height = torch.where(swing, height.clamp_min(0), 0.0)
  return height, ~swing


class PhaseVelocityCommand(LPACRLVelocityCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    with torch.inference_mode(False):
      self.phase_cycles = torch.zeros(self.num_envs, device=self.device)

  def checkpoint_identity(self):
    return {**super().checkpoint_identity(), "phase_guidance": PHASE_CONTRACT}

  def _resample_command(self, env_ids):
    super()._resample_command(env_ids)
    if hasattr(self, "phase_cycles"):
      phase = (
        torch.rand(len(env_ids), device=self.device)
        if self.cfg.adaptive
        else torch.zeros(len(env_ids), device=self.device)
      )
      self.phase_cycles[env_ids] = torch.where(
        moving(self.vel_command_b[env_ids]), phase, 0.0
      )

  def compute(self, dt):
    super().compute(dt)
    active = moving(self.vel_command_b)
    self.phase_cycles.copy_(
      torch.where(active, (self.phase_cycles + FREQUENCY_HZ * max(dt, 0.0)) % 1.0, 0.0)
    )


class PhaseVelocityCommandCfg(LPACRLVelocityCommandCfg):
  def build(self, env):
    self.__post_init__()
    return PhaseVelocityCommand(self, env)


def gait_phase(env):
  term = env.command_manager.get_term("base_velocity")
  angle = 2 * math.pi * term.phase_cycles
  return (
    torch.stack((angle.sin(), angle.cos()), dim=-1)
    * moving(term.vel_command_b)[:, None]
  )


class phase_foot_lift_error:
  """Last-contact-relative height tracking, with no ground query or event bonus."""

  def __init__(self, cfg, env):
    with torch.inference_mode(False):
      self.reference = torch.zeros((env.num_envs, 4), device=env.device)
      self.has_reference = torch.zeros_like(self.reference, dtype=torch.bool)

  def reset(self, env_ids=None):
    ids = slice(None) if env_ids is None else env_ids
    self.reference[ids] = 0.0
    self.has_reference[ids] = False

  def __call__(self, env, asset_cfg):
    sensor = env.scene["feet_ground_contact"]
    names = [name.rsplit("/", 1)[-1] for name in sensor.primary_names]
    order = [names.index(name) for name in FOOT_GEOMS]
    contact = sensor.data.found[:, order] > 0
    z = env.scene[asset_cfg.name].data.site_pos_w[:, asset_cfg.site_ids, 2]
    self.reference.copy_(torch.where(contact, z, self.reference))
    self.has_reference |= contact
    term = env.command_manager.get_term("base_velocity")
    # Reward is before command.compute; grade the end of the action's interval.
    height, _ = foot_targets((term.phase_cycles + FREQUENCY_HZ * env.step_dt) % 1.0)
    error = ((z - self.reference - height) / 0.025).square().clamp_max(9.0)
    return (error * self.has_reference).mean(-1) * moving(term.vel_command_b)


def phase_foot_contact_error(env):
  sensor = env.scene["feet_ground_contact"]
  names = [name.rsplit("/", 1)[-1] for name in sensor.primary_names]
  order = [names.index(name) for name in FOOT_GEOMS]
  contact = sensor.data.found[:, order] > 0
  term = env.command_manager.get_term("base_velocity")
  _, desired = foot_targets((term.phase_cycles + FREQUENCY_HZ * env.step_dt) % 1.0)
  return (contact != desired).float().mean(-1) * moving(term.vel_command_b)


def mirror_phase_observations(values):
  if values.shape[-1] != 50:
    raise ValueError("Phase gait requires48 body +2 phase observations")
  return torch.cat(
    (mirror_blind_observations(values[..., :48]), -values[..., 48:]), dim=-1
  )


def augment_phase_symmetry(env, obs=None, actions=None):
  del env
  if obs is not None:
    mirrored = obs.clone()
    for name in obs.keys():
      if name not in ("actor", "critic"):
        raise ValueError("Unexpected phase observation group")
      mirrored[name] = mirror_phase_observations(obs[name])
    obs = torch.cat((obs, mirrored), dim=0)
  if actions is not None:
    actions = torch.cat((actions, mirror_joint_vectors(actions)), dim=0)
  return obs, actions


class PhaseQuietStopMLP(QuietStopMLPModel):
  def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
    if (
      obs_set != "actor"
      or tuple(obs_groups[obs_set]) != ("actor",)
      or obs["actor"].shape[-1] != 50
      or output_dim != 12
    ):
      raise ValueError("Phase actor requires50 inputs and12 actions")
    MLPModel.__init__(self, obs, obs_groups, obs_set, output_dim, **kwargs)
    if type(self.distribution) is not GaussianDistribution:
      raise ValueError("Phase actor requires Gaussian distribution")

  def exploration_identity(self):
    return {**super().exploration_identity(), "raw_observation_width": 50}
