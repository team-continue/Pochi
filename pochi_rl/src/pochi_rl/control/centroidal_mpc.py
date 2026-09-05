"""Finite-horizon contact-force MPC for a level, slowly walking quadruped.

The state is [COM position, roll/pitch/yaw, COM velocity, world angular
velocity]. A single rigid body approximates the articulated robot. Every solve
optimizes all twelve contact-force components over the horizon; only the first
control is applied. Scheduled swing forces are zero, stance forces obey a
conservative friction pyramid and unilateral normal-force bounds.

OSQP interface: https://osqp.org/docs/interfaces/python.html
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import numpy as np
import osqp
from scipy import sparse


@dataclass(frozen=True)
class MPCConfig:
  horizon: int = 16
  dt: float = 0.05
  friction: float = 0.5
  max_normal_force: float = 100.0
  force_cost: float = 2e-4

  def __post_init__(self) -> None:
    if self.horizon < 2 or not isinstance(self.horizon, int):
      raise ValueError("horizon must be an integer >= 2")
    for name in ("dt", "friction", "max_normal_force", "force_cost"):
      if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
        raise ValueError(f"{name} must be finite and positive")


def skew(v: np.ndarray) -> np.ndarray:
  x, y, z = v
  return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


class CentroidalMPC:
  def __init__(
    self, mass: float, inertia: np.ndarray, cfg: MPCConfig | None = None
  ) -> None:
    self.cfg = cfg or MPCConfig()
    self.mass = mass
    self.inertia = np.asarray(inertia)
    n = self.cfg.horizon
    # Position, orientation, linear velocity, angular velocity.
    weights = np.array([30, 60, 800, 800, 1000, 150, 40, 50, 80, 10, 10, 15])
    state_weights = np.tile(weights, n)
    state_weights[-12:] *= 2
    self._weights = state_weights
    self._P = sparse.diags(
      2 * np.r_[state_weights, np.full(12 * n, self.cfg.force_cost)],
      format="csc",
    )
    self._Ad = np.eye(12)
    self._Ad[:6, 6:] = np.eye(6) * self.cfg.dt
    mu = self.cfg.friction / np.sqrt(2)  # pyramid inside the circular cone
    cone = np.array(
      [
        [1, 0, -mu],
        [-1, 0, -mu],
        [0, 1, -mu],
        [0, -1, -mu],
        [0, 0, 1],
      ]
    )
    self._cone = sparse.kron(sparse.eye(4 * n), cone, format="csc")
    self._previous: np.ndarray | None = None
    self.prediction = np.zeros((n, 12))
    self.forces = np.zeros((n, 4, 3))
    self.status = "not solved"
    self.solve_ms = 0.0
    self.failures = 0

  def solve(
    self,
    state: np.ndarray,
    reference: np.ndarray,
    feet: np.ndarray,
    contacts: np.ndarray,
    rotation: np.ndarray,
  ) -> np.ndarray:
    """Return first-step world ground reactions (4, 3).

    ``reference`` is (N, 12), ``feet`` (N, 4, 3), ``contacts`` (N, 4).
    Linearization uses measured attitude/inertia and extrapolated COM moment arms.
    """
    start = perf_counter()
    n, dt = self.cfg.horizon, self.cfg.dt
    inv_inertia = np.linalg.inv(rotation @ self.inertia @ rotation.T)
    blocks = []
    for k in range(n):
      acc = np.zeros((6, 12))
      for leg in range(4):
        acc[:3, 3 * leg : 3 * leg + 3] = np.eye(3) / self.mass
        acc[3:, 3 * leg : 3 * leg + 3] = inv_inertia @ skew(
          feet[k, leg] - (state[:3] + k * dt * state[6:9])
        )
      blocks.append(sparse.csc_matrix(np.vstack((0.5 * dt**2 * acc, dt * acc))))
    dynamics = sparse.eye(12 * n, format="csc") - sparse.kron(
      sparse.diags(np.ones(n - 1), -1, shape=(n, n)), self._Ad, format="csc"
    )
    equality = sparse.hstack((dynamics, -sparse.block_diag(blocks)), format="csc")
    gravity = np.zeros(12)
    gravity[2], gravity[8] = -0.5 * 9.81 * dt**2, -9.81 * dt
    rhs = np.tile(gravity, n)
    rhs[:12] += self._Ad @ state
    lower = np.full((n, 4, 5), -np.inf)
    upper = np.zeros((n, 4, 5))
    lower[:, :, 4] = 0
    upper[:, :, 4] = contacts * self.cfg.max_normal_force
    constraint = sparse.vstack(
      (
        equality,
        sparse.hstack((sparse.csc_matrix((20 * n, 12 * n)), self._cone)),
      ),
      format="csc",
    )
    solver = osqp.OSQP()
    solver.setup(
      P=self._P,
      q=np.r_[-2 * self._weights * reference.ravel(), np.zeros(12 * n)],
      A=constraint,
      l=np.r_[rhs, lower.ravel()],
      u=np.r_[rhs, upper.ravel()],
      verbose=False,
      eps_abs=1e-4,
      eps_rel=1e-4,
      max_iter=4000,
      polishing=False,
    )
    if self._previous is not None:
      solver.warm_start(x=self._previous)
    result = solver.solve(raise_error=False)
    self.status = result.info.status
    self.solve_ms = (perf_counter() - start) * 1000
    if result.info.status_val not in (1, 2) or not np.isfinite(result.x).all():
      self.failures += 1
      # A failed solve must not leave old forces on a newly airborne foot.
      force = np.zeros((4, 3))
      support = contacts[0].astype(bool)
      force[support, 2] = min(
        self.mass * 9.81 / max(1, support.sum()), self.cfg.max_normal_force
      )
      return force
    self.prediction = result.x[: 12 * n].reshape(n, 12).copy()
    self.forces = result.x[12 * n :].reshape(n, 4, 3).copy()
    self._previous = np.r_[
      np.vstack((self.prediction[1:], self.prediction[-1:])).ravel(),
      np.concatenate((self.forces[1:], self.forces[-1:])).ravel(),
    ]
    force = self.forces[0].copy()
    force[~contacts[0].astype(bool)] = 0
    return force
