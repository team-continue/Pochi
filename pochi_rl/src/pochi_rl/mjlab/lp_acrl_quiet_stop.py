"""Opt-in quiet-stop exploration; deterministic policy and physics stay intact.

The conditional Normal is rebuilt from raw command observations both during
rollout sampling and PPO likelihood recomputation. Actions are never replaced
or scaled after sampling. Existing blind actor state dictionaries load strictly.
"""

from __future__ import annotations

from pathlib import Path

import torch
from rsl_rl.models import MLPModel
from rsl_rl.modules import GaussianDistribution, HiddenState
from rsl_rl.utils import unpad_trajectories
from tensordict import TensorDict
from torch.distributions import Normal

from pochi_rl.mjlab.lp_acrl_runner import (
  LPACRLOnPolicyRunner,
  validate_blind_symmetry_cfg,
)


class QuietStopMLPModel(MLPModel):
  """Keep a 48→12 mean policy; reduce only exact-stop sampling variance."""

  QUIET_STD = 0.01
  STOP_THRESHOLD = 1.0e-6

  def __init__(self, obs, obs_groups, obs_set, output_dim, **kwargs):
    if (
      obs_set != "actor"
      or tuple(obs_groups.get(obs_set, ())) != ("actor",)
      or obs["actor"].ndim != 2
      or obs["actor"].shape[-1] != 48
      or output_dim != 12
    ):
      raise ValueError("Quiet-stop actor requires actor-only48D input and12 actions")
    super().__init__(obs, obs_groups, obs_set, output_dim, **kwargs)
    if type(self.distribution) is not GaussianDistribution:
      raise ValueError("Quiet-stop actor requires the standard GaussianDistribution")

  def forward(
    self,
    obs: TensorDict,
    masks: torch.Tensor | None = None,
    hidden_state: HiddenState = None,
    stochastic_output: bool = False,
  ) -> torch.Tensor:
    # Match the inherited nonrecurrent unpadding before applying the raw gate.
    if masks is not None:
      obs = unpad_trajectories(obs, masks)
    mean = super().forward(obs, hidden_state=hidden_state, stochastic_output=False)
    if not stochastic_output:
      return mean
    assert isinstance(self.distribution, GaussianDistribution)
    self.distribution.update(mean)
    # Read BEFORE normalization. A zero physical command generally has a
    # nonzero normalized value, and reflected minibatches preserve this gate.
    standing = obs["actor"][..., 9:12].abs().amax(dim=-1) <= self.STOP_THRESHOLD
    moving_std = self.distribution.std
    std = torch.where(
      standing.unsqueeze(-1), torch.full_like(moving_std, self.QUIET_STD), moving_std
    )
    self.distribution._distribution = Normal(mean, std)
    return self.distribution.sample()

  def exploration_identity(self) -> dict:
    distribution = self.distribution
    assert isinstance(distribution, GaussianDistribution)
    parameter = (
      distribution.std_param if distribution.std_type == "scalar"
      else distribution.log_std_param
    )
    return {
      "version": 1,
      "actor_class": f"{type(self).__module__}:{type(self).__qualname__}",
      "raw_observation_group": "actor",
      "raw_observation_width": 48,
      "command_columns": (9, 10, 11),
      "standing_max_abs_command": self.STOP_THRESHOLD,
      "standing_action_std": self.QUIET_STD,
      "moving_distribution": "GaussianDistribution",
      "moving_std_type": distribution.std_type,
      "moving_std_range": tuple(distribution.std_range),
      "moving_std_learned": parameter.requires_grad,
      "deterministic_inference": "unchanged inherited mean; no injected action",
    }


class LPACRLQuietStopOnPolicyRunner(LPACRLOnPolicyRunner):
  """Record exploration semantics and reject incompatible stateful resumes."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    validate_blind_symmetry_cfg(self.env.unwrapped.cfg)
    if not isinstance(self.alg._raw_actor, QuietStopMLPModel):
      raise ValueError("Quiet-stop runner requires QuietStopMLPModel actor")

  def exploration_identity(self) -> dict:
    return self.alg._raw_actor.exploration_identity()

  def save(self, path: str, infos=None) -> None:
    super().save(path, {
      **(infos or {}),
      "lp_acrl_training_exploration": self.exploration_identity(),
    })

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    stateful = load_cfg is None or any(
      load_cfg.get(key, False) for key in ("optimizer", "iteration")
    )
    if stateful:
      saved = torch.load(Path(path), map_location="cpu", weights_only=False)
      identity = (saved.get("infos") or {}).get("lp_acrl_training_exploration")
      if identity != self.exploration_identity():
        raise ValueError("LP-ACRL exploration differs; cannot resume training state")
    return super().load(path, load_cfg, strict, map_location)
