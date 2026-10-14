"""V19: wide speed search with signed LP and explicit low-speed retention.

Search ceilings are task bounds, not assertions about achievable speed. The
first four bands exactly reproduce V18. No competence gate or event bonus is
introduced: signed learning progress chooses among all 220 tasks from startup.
"""

from dataclasses import asdict, dataclass

import torch

from pochi_rl.lp_acrl import LPACRLCurriculum
from pochi_rl.mjlab.lp_acrl_omni_commands import OmniCatalog
from pochi_rl.mjlab.lp_acrl_phase_gait import (
  PhaseVelocityCommand,
  PhaseVelocityCommandCfg,
  moving,
)

MAX_COMMAND = (4.0, 1.5, 4.0)
AXIS_EDGES = (
  (0.04, 0.20, 0.40, 0.60, 0.80, 1.00, 1.25, 1.50, 2.00, 3.00, 4.00),
  (0.025, 0.125, 0.250, 0.375, 0.500, 0.600, 0.750, 0.900, 1.100, 1.300, 1.500),
  (0.09, 0.45, 0.90, 1.35, 1.80, 2.00, 2.25, 2.50, 3.00, 3.50, 4.00),
)
RETENTION_FRACTION = 0.20


@dataclass(frozen=True)
class SpeedCatalog:
  axis_edges: tuple = AXIS_EDGES
  terrains: tuple = ("flat",)
  num_levels: int = 1

  def __post_init__(self):
    if len(self.axis_edges) != 3 or any(len(e) != 11 for e in self.axis_edges):
      raise ValueError("Require three ten-band speed axes")
    if any(
      not all(0 < lo < hi for lo, hi in zip(e[:-1], e[1:], strict=True))
      for e in self.axis_edges
    ):
      raise ValueError("Speed edges must be positive and strictly increasing")

  def directions(self):
    return OmniCatalog().directions()

  @property
  def num_tasks(self):
    return len(self.directions()) * 10

  def task_indices(self, device="cpu"):
    return torch.tensor(
      [(direction, band, 0, 0) for direction in range(22) for band in range(10)],
      dtype=torch.long,
      device=device,
    )

  def task_bounds(self, device="cpu"):
    return torch.tensor(
      [
        [
          sorted((edges[band] * sign, edges[band + 1] * sign))
          for edges, sign in zip(self.axis_edges, signs, strict=True)
        ]
        for _, signs in self.directions()
        for band in range(10)
      ],
      device=device,
    )

  def retention_probabilities(self, device="cpu"):
    names = [name for name, _ in self.directions()]
    return torch.tensor(
      [
        1 / (6 * names.count(name) * 4) if band < 4 else 0.0
        for name in names
        for band in range(10)
      ],
      dtype=torch.float64,
      device=device,
    )


SPEED_CATALOG = SpeedCatalog()
COMMAND_CONTRACT = {
  "version": 1,
  "max_abs_vx_vy_wz": MAX_COMMAND,
  "axis_edges": AXIS_EDGES,
  "catalog_directions": SPEED_CATALOG.directions(),
  "fixed_episode_sampling": (
    "0.8 signed-LP softmax over all bands + 0.2 V18-band retention"
  ),
  "retention_fraction": RETENTION_FRACTION,
  "standing_episode_fraction": 0.2,
  "transition_episode_fraction": 0.2,
  "transition_interval_s": (4.0, 6.0),
  "auxiliary_task_sampling": "equal mode/sign/first four V18 bands",
  "auxiliary_episode_lp_credit": False,
  "command_frame": "robot body vx,vy,wz; nonselected axes exactly zero",
  "terrain": "native infinite plane; no artificial terrain-edge timeout",
  "speed_ceilings_are_search_bounds": True,
  "competence_gates": False,
}


class SpeedLPACRLCurriculum(LPACRLCurriculum):
  def effective_probabilities(self):
    return (
      1 - RETENTION_FRACTION
    ) * self.probabilities + RETENTION_FRACTION * SPEED_CATALOG.retention_probabilities(
      self.device
    )

  @torch.no_grad()
  def sample(self, num_samples, *, generator=None):
    if (
      isinstance(num_samples, bool)
      or not isinstance(num_samples, int)
      or num_samples < 0
    ):
      raise ValueError("num_samples must be a nonnegative integer")
    if not num_samples:
      return torch.empty(0, dtype=torch.long, device=self.device)
    return torch.multinomial(
      self.effective_probabilities(), num_samples, replacement=True, generator=generator
    )


class SpeedPhaseVelocityCommand(PhaseVelocityCommand):
  def __init__(self, cfg, env):
    super().__init__(cfg, env)
    self.catalog = SPEED_CATALOG
    self._task_indices = self.catalog.task_indices(self.device)
    self._task_bounds = self.catalog.task_bounds(self.device)
    self.curriculum = SpeedLPACRLCurriculum(
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
      **asdict(SPEED_CATALOG),
      "signed_yaw": True,
      "lateral_range": (-1.5, 1.5),
    }

  def prepare_episode(self, env_ids):
    super().prepare_episode(env_ids)
    if not self._evaluation:
      auxiliary = env_ids[self.episode_mode[env_ids] != 0]
      if len(auxiliary):
        self.task_ids[auxiliary] = torch.multinomial(
          self.catalog.retention_probabilities(self.device),
          len(auxiliary),
          replacement=True,
        )

  def _resample_command(self, env_ids):
    if (self.task_ids[env_ids] < 0).any():
      raise RuntimeError("Pre-reset task selection must precede command reset")
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
        len(transition_ids),
        device=self.device,
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
      bands = self._task_indices[:, 1]
      self._env.extras["log"].update(
        {
          "LP_ACRL/speed_probability_above_v18": effective[bands >= 4].sum(),
          "LP_ACRL/speed_probability_top_band": effective[bands == 9].sum(),
          "LP_ACRL/speed_entropy": -(
            effective * effective.clamp_min(1e-300).log()
          ).sum(),
        }
      )


class SpeedPhaseVelocityCommandCfg(PhaseVelocityCommandCfg):
  def build(self, env):
    self.__post_init__()
    if (
      self.rough
      or self.standing_episode_fraction != 0.2
      or self.transition_episode_fraction != 0.2
    ):
      raise ValueError("Speed search requires registered flat/stop protocol")
    return SpeedPhaseVelocityCommand(self, env)
