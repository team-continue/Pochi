"""Action terms for Pochi.

The velocity task uses mjlab's stock joint-position action unchanged; only the
stand-up task needs a variant, and only because of where it centres.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
from mjlab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class CenteredJointPositionAction(JointPositionAction):
  """Joint-position action centred on a pose of the caller's choosing.

  The stock term offsets actions by the robot's default joint positions, so a
  zero action commands the standing stance.  That is the right centre for a
  gait, which lives near the stance, and the wrong one for standing up, which
  starts at the other end of the joint range: with the usual 0.25 rad scale the
  policy has to emit -3.6 just to hold the pose the episode begins in, which is
  3.6 sigma out on a unit-variance Gaussian and therefore never explored.  The
  resulting policy cannot stay down at all and leaps to a half-crouch before the
  reference has asked for anything.

  Raising the scale instead is worse: it buys the reach at the cost of control
  authority everywhere, and at 0.7 rad the robot thrashes itself over.
  Centring the action range on the *middle* of the manoeuvre costs nothing and
  puts both ends of it at about 1.8 sigma.
  """

  cfg: CenteredJointPositionActionCfg

  def __init__(
    self, cfg: CenteredJointPositionActionCfg, env: ManagerBasedRlEnv
  ) -> None:
    super().__init__(cfg, env)
    centre = torch.tensor(
      [cfg.centre_pose[name] for name in self._target_names],
      dtype=torch.float32,
      device=env.device,
    )
    # BaseAction types _offset as a scalar-or-tensor; the stock joint-position
    # term likewise replaces it with a per-joint tensor.
    self._offset = centre.unsqueeze(0).repeat(env.num_envs, 1)  # type: ignore[assignment]


@dataclass(kw_only=True)
class CenteredJointPositionActionCfg(JointPositionActionCfg):
  """``centre_pose`` maps joint name to the angle a zero action commands."""

  centre_pose: dict[str, float] = field(default_factory=dict)
  use_default_offset: bool = False

  def build(self, env: ManagerBasedRlEnv) -> CenteredJointPositionAction:
    return CenteredJointPositionAction(self, env)
