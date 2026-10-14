"""Flat, signed three-axis tasks with explicit pure lateral and yaw practice."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import product

import torch

from pochi_rl.lp_acrl import LPACRLCurriculum
from pochi_rl.mjlab.lp_acrl_phase_gait import (
  PhaseVelocityCommand,
  PhaseVelocityCommandCfg,
  moving,
)

MAX_COMMAND = (0.8, 0.5, 1.8)
STRENGTH_EDGES = (0.05, 0.25, 0.5, 0.75, 1.0)
COVERAGE_FRACTION = 0.5


@dataclass(frozen=True)
class OmniCatalog:
  """Each sign combination and strength band is a separate LP task."""

  max_command: tuple = MAX_COMMAND
  strength_edges: tuple = STRENGTH_EDGES
  terrains: tuple = ("flat",)
  num_levels: int = 1

  def directions(self):
    rows = []
    for name, active in (
      ("forward", (0,)),
      ("lateral", (1,)),
      ("yaw", (2,)),
      ("diagonal", (0, 1)),
      ("arc", (0, 2)),
      ("omni", (0, 1, 2)),
    ):
      for signs in product((-1, 1), repeat=len(active)):
        vector = [0, 0, 0]
        for axis, sign in zip(active, signs, strict=True):
          vector[axis] = sign
        rows.append((name, tuple(vector)))
    return tuple(rows)

  @property
  def num_tasks(self):
    return len(self.directions()) * (len(self.strength_edges) - 1)

  def task_indices(self, device="cpu"):
    return torch.tensor(
      [
        (direction, band, 0, 0)
        for direction in range(len(self.directions()))
        for band in range(len(self.strength_edges) - 1)
      ],
      dtype=torch.long,
      device=device,
    )

  def task_bounds(self, device="cpu"):
    rows = []
    for _, signs in self.directions():
      for lo, hi in zip(self.strength_edges[:-1], self.strength_edges[1:], strict=True):
        rows.append(
          [
            sorted((lo * maximum * sign, hi * maximum * sign))
            for maximum, sign in zip(self.max_command, signs, strict=True)
          ]
        )
    return torch.tensor(rows, device=device)

  def coverage_probabilities(self, device="cpu"):
    names = [name for name, _ in self.directions()]
    bands = len(self.strength_edges) - 1
    return torch.tensor(
      [1 / (6 * names.count(name) * bands) for name in names for _ in range(bands)],
      dtype=torch.float64,
      device=device,
    )


OMNI_CATALOG = OmniCatalog()
COMMAND_CONTRACT = {
  "version": 1,
  "max_abs_vx_vy_wz": MAX_COMMAND,
  "catalog_directions": OMNI_CATALOG.directions(),
  "strength_edges": STRENGTH_EDGES,
  "fixed_episode_sampling": "0.5 signed-LP softmax + 0.5 equal-mode coverage",
  "coverage_fraction": COVERAGE_FRACTION,
  "standing_episode_fraction": 0.2,
  "transition_episode_fraction": 0.2,
  "transition_interval_s": (4.0, 6.0),
  "auxiliary_task_sampling": "equal mode, signs and strength bands",
  "auxiliary_episode_lp_credit": False,
  "command_frame": "robot body vx,vy,wz; nonselected axes exactly zero",
  "terrain": "native infinite plane; no artificial terrain-edge timeout",
}


class CoverageLPACRLCurriculum(LPACRLCurriculum):
  """Keep original signed-LP statistics; add a declared sampling coverage floor."""

  def effective_probabilities(self):
    reference = OMNI_CATALOG.coverage_probabilities(self.device)
    return (1 - COVERAGE_FRACTION) * self.probabilities + COVERAGE_FRACTION * reference

  @torch.no_grad()
  def sample(self, num_samples, *, generator=None):
    if (
      isinstance(num_samples, bool)
      or not isinstance(num_samples, int)
      or num_samples < 0
    ):
      raise ValueError("num_samples must be a nonnegative integer")
    if num_samples == 0:
      return torch.empty(0, dtype=torch.long, device=self.device)
    return torch.multinomial(
      self.effective_probabilities(), num_samples, replacement=True, generator=generator
    )


class OmniPhaseVelocityCommand(PhaseVelocityCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.catalog = OMNI_CATALOG
    self._task_indices = self.catalog.task_indices(self.device)
    self._task_bounds = self.catalog.task_bounds(self.device)
    self.curriculum = CoverageLPACRLCurriculum(
      self.catalog.num_tasks,
      beta=cfg.beta,
      stage_steps=cfg.stage_steps,
      min_episodes=cfg.min_episodes,
      device=self.device,
    )

  def command_protocol_identity(self):
    return COMMAND_CONTRACT

  def checkpoint_identity(self):
    return {
      **super().checkpoint_identity(),
      **asdict(OMNI_CATALOG),
      "signed_yaw": True,
      "lateral_range": (-0.5, 0.5),
    }

  def prepare_episode(self, env_ids):
    super().prepare_episode(env_ids)
    if not self._evaluation:
      auxiliary = env_ids[self.episode_mode[env_ids] != 0]
      if len(auxiliary):
        self.task_ids[auxiliary] = torch.multinomial(
          self.catalog.coverage_probabilities(self.device),
          len(auxiliary),
          replacement=True,
        )

  def _resample_command(self, env_ids):
    if (self.task_ids[env_ids] < 0).any():
      raise RuntimeError("Pre-reset task selection must run before command reset")
    bounds = self._task_bounds[self.task_ids[env_ids]]
    values = bounds[:, :, 0] + torch.rand((len(env_ids), 3), device=self.device) * (
      bounds[:, :, 1] - bounds[:, :, 0]
    )
    self._movement_command[env_ids] = values
    self.vel_command_b[env_ids] = values
    self.is_standing_env[env_ids] = self.episode_mode[env_ids] == 1
    transition_ids = env_ids[self.episode_mode[env_ids] == 2]
    if len(transition_ids):
      self.is_standing_env[transition_ids] = (
        torch.rand(len(transition_ids), device=self.device) < 0.5
      )
      self._transition_time_left[transition_ids] = torch.empty(
        len(transition_ids), device=self.device
      ).uniform_(*self.cfg.transition_interval_s)
    self.vel_command_b[env_ids[self.is_standing_env[env_ids]]] = 0.0
    if self._forced_command is not None:
      self.vel_command_b[env_ids] = self._forced_command
    self.is_standing_env[env_ids] = ~moving(self.vel_command_b[env_ids])
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]
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
    stage = self.curriculum.stage_index
    super().compute(dt)
    if not self._evaluation and stage != self.curriculum.stage_index:
      effective = self.curriculum.effective_probabilities()
      self._env.extras["log"].update(
        {
          "LP_ACRL/coverage_probability_min": effective.min(),
          "LP_ACRL/coverage_probability_max": effective.max(),
          "LP_ACRL/coverage_entropy": -(effective * effective.log()).sum(),
        }
      )


class OmniPhaseVelocityCommandCfg(PhaseVelocityCommandCfg):
  def build(self, env):
    self.__post_init__()
    if (
      self.rough
      or self.standing_episode_fraction != 0.2
      or self.transition_episode_fraction != 0.2
    ):
      raise ValueError("Omni command requires the registered flat/stop protocol")
    return OmniPhaseVelocityCommand(self, env)


def planar_tracking_huber_error(env, delta=0.25):
  """Continuous command-relative xy error, including pure sideways movement."""
  command = env.command_manager.get_command("base_velocity")[:, :2]
  velocity = env.scene["robot"].data.root_link_lin_vel_b[:, :2]
  absolute = (command - velocity).abs()
  quadratic = absolute.clamp_max(delta)
  return (0.5 * quadratic.square() + delta * (absolute - quadratic)).sum(-1)
