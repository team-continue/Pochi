"""Explicit low-noise refinement contract; deterministic actor math is unchanged."""

from __future__ import annotations

from pathlib import Path

import torch

from pochi_rl.mjlab.lp_acrl_contact_runner import LPACRLContactOnPolicyRunner
from pochi_rl.mjlab.lp_acrl_quiet_stop import QuietStopMLPModel
from pochi_rl.mjlab.lp_acrl_symmetry import augment_blind_symmetry

MOVING_STD = 0.03
CORRUPTION_IDENTITY = {
  "version": 1,
  "groups": {"actor": False, "critic": False},
}


def expected_exploration_identity() -> dict:
  """Portable complete contract, also checked by the pre-environment evaluator."""
  return {
    "version": 1,
    "actor_class": f"{QuietStopMLPModel.__module__}:{QuietStopMLPModel.__qualname__}",
    "raw_observation_group": "actor",
    "raw_observation_width": 48,
    "command_columns": (9, 10, 11),
    "standing_max_abs_command": QuietStopMLPModel.STOP_THRESHOLD,
    "standing_action_std": QuietStopMLPModel.QUIET_STD,
    "moving_distribution": "GaussianDistribution",
    "moving_std_type": "scalar",
    "moving_std_range": (MOVING_STD, MOVING_STD),
    "moving_std_learned": False,
    "deterministic_inference": "unchanged inherited mean; no injected action",
  }


def expected_symmetry_identity() -> dict:
  """Keep the mean mirror regularizer without off-policy augmented PPO rows."""
  return {
    "data_augmentation_func": (
      f"{augment_blind_symmetry.__module__}:{augment_blind_symmetry.__qualname__}"
    ),
    "use_data_augmentation": False,
    "use_mirror_loss": True,
    "mirror_loss_coeff": 1.0,
  }


def validate_low_noise_checkpoint(saved: dict) -> None:
  """Check identities and the fixed parameter without constructing an environment."""
  metadata = saved.get("infos") or {}
  if metadata.get("lp_acrl_observation_corruption") != CORRUPTION_IDENTITY:
    raise ValueError("Checkpoint observation corruption differs or is missing")
  if metadata.get("lp_acrl_training_exploration") != expected_exploration_identity():
    raise ValueError("Checkpoint low-noise exploration differs or is missing")
  if metadata.get("lp_acrl_training_symmetry") != expected_symmetry_identity():
    raise ValueError("Checkpoint low-noise symmetry differs or is missing")
  LPACRLLowNoiseOnPolicyRunner._validate_std_parameter(
    (saved.get("actor_state_dict") or {}).get("distribution.std_param")
  )


class LPACRLLowNoiseOnPolicyRunner(LPACRLContactOnPolicyRunner):
  """Record observation corruption omitted by the legacy task-space identity."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.validate_low_noise_configuration()

  def observation_corruption_identity(self) -> dict:
    groups = self.env.unwrapped.cfg.observations
    if set(groups) != {"actor", "critic"}:
      raise ValueError("Low-noise task requires exactly actor and critic groups")
    identity = {
      "version": 1,
      "groups": {name: bool(group.enable_corruption) for name, group in groups.items()},
    }
    if identity != CORRUPTION_IDENTITY:
      raise ValueError("Low-noise task requires observation corruption disabled")
    return identity

  @staticmethod
  def _validate_std_parameter(value: torch.Tensor) -> None:
    if (
      not isinstance(value, torch.Tensor)
      or value.shape != (12,)
      or not value.is_floating_point()
      or not torch.isfinite(value).all()
      or not torch.allclose(
        value, torch.full_like(value, MOVING_STD), rtol=0, atol=1e-8
      )
    ):
      raise ValueError("Low-noise actor requires a fixed 12D std parameter of 0.03")

  def validate_low_noise_configuration(self) -> None:
    self.observation_corruption_identity()
    if self.exploration_identity() != expected_exploration_identity():
      raise ValueError("Low-noise exploration must use fixed moving 0.03/stopped 0.01")
    if self.symmetry_identity() != expected_symmetry_identity():
      raise ValueError("Low-noise PPO requires mirror loss without data augmentation")
    self._validate_std_parameter(self.alg._raw_actor.distribution.std_param)

  def save(self, path: str, infos=None) -> None:
    self.validate_low_noise_configuration()
    super().save(path, {
      **(infos or {}),
      "lp_acrl_observation_corruption": self.observation_corruption_identity(),
    })

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    self.validate_low_noise_configuration()
    saved = torch.load(Path(path), map_location="cpu", weights_only=False)
    validate_low_noise_checkpoint(saved)
    return super().load(path, load_cfg, strict, map_location)
