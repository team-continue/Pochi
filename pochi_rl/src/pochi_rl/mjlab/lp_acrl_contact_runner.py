"""Require the validated contact backend when loading the corrected blind task.

The numerical backend is separate from the environment configuration identity:
its process-local CCD function binding must be recorded even for actor-only
playback. Older actors may only initialize an explicitly audited refinement.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch

from pochi_rl.mjlab.lp_acrl_quiet_stop import LPACRLQuietStopOnPolicyRunner
from pochi_rl.mjlab.warp_contact_backend import active_contact_backend_identity


class LPACRLContactOnPolicyRunner(LPACRLQuietStopOnPolicyRunner):
  """Quiet-stop PPO with a fail-closed contact-backend checkpoint contract."""

  def __init__(self, *args, **kwargs):
    self._contact_backend = deepcopy(active_contact_backend_identity())
    super().__init__(*args, **kwargs)

  def contact_backend_identity(self) -> dict:
    current = active_contact_backend_identity()
    if current != self._contact_backend:
      raise ValueError("Contact backend changed after runner construction")
    return deepcopy(current)

  def save(self, path: str, infos=None) -> None:
    metadata = {
      **(infos or {}),
      "lp_acrl_contact_backend": self.contact_backend_identity(),
    }
    super().save(path, metadata)

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    saved = torch.load(Path(path), map_location="cpu", weights_only=False)
    metadata = saved.get("infos") or {}
    if metadata.get("lp_acrl_contact_backend") != self.contact_backend_identity():
      raise ValueError(
        "Checkpoint contact backend differs or is missing; use an explicit "
        "refinement to initialize from a policy trained with the old backend"
      )
    return super().load(path, load_cfg, strict, map_location)
