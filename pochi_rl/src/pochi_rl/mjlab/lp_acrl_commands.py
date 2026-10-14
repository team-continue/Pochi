"""Episode-scoped task sampling and reset integration for LP-ACRL.

mjlab invokes curriculum terms before resetting physics, rewards and commands.
That ordering is essential: credit the completed episode to its old task, then
place the robot on its next terrain before its reset pose is written.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import TYPE_CHECKING, Any

import torch
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp.velocity_command import (
  UniformVelocityCommand,
  UniformVelocityCommandCfg,
)

from pochi_rl.lp_acrl import (
  FLAT_TASK_CATALOG,
  ROUGH_TASK_CATALOG,
  LPACRLCurriculum,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _config_identity(value: Any) -> Any:
  """Stable plain values, without callable pickles or memory-address reprs."""
  if value is None or isinstance(value, (bool, int, float, str)):
    return value
  if isinstance(value, SceneEntityCfg):
    # Manager construction resolves local indices in place. Physical selection
    # is named, and indices may differ when the terrain geometry is rebuilt.
    return {
      "name": value.name,
      **{
        field.name: _config_identity(getattr(value, field.name))
        for field in fields(value)
        if field.name.endswith("_names") or field.name == "preserve_order"
      },
    }
  if is_dataclass(value):
    return {
      "type": f"{type(value).__module__}.{type(value).__qualname__}",
      **{
        field.name: _config_identity(getattr(value, field.name))
        for field in fields(value)
      },
    }
  if isinstance(value, dict):
    return {key: _config_identity(item) for key, item in value.items()}
  if isinstance(value, (tuple, list)):
    return tuple(_config_identity(item) for item in value)
  if isinstance(value, slice):
    return {"slice": (value.start, value.stop, value.step)}
  if callable(value) and hasattr(value, "__qualname__"):
    return f"{value.__module__}.{value.__qualname__}"
  raise TypeError(f"Unsupported checkpoint config value: {type(value).__name__}")


class LPACRLVelocityCommand(UniformVelocityCommand):
  """One fixed velocity/terrain task per episode, with signed LP sampling."""

  cfg: LPACRLVelocityCommandCfg

  def __init__(self, cfg: LPACRLVelocityCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    self.catalog = ROUGH_TASK_CATALOG if cfg.rough else FLAT_TASK_CATALOG
    self.curriculum = LPACRLCurriculum(
      self.catalog.num_tasks,
      beta=cfg.beta,
      stage_steps=cfg.stage_steps,
      min_episodes=cfg.min_episodes,
      device=self.device,
    )
    self.task_ids = torch.full(
      (self.num_envs,), -1, device=self.device, dtype=torch.long
    )
    self._task_indices = self.catalog.task_indices(device=self.device)
    self._task_bounds = self.catalog.task_bounds(device=self.device)
    self._evaluation = not cfg.adaptive
    self._forced_task_ids: torch.Tensor | None = None
    self._forced_command: torch.Tensor | None = None
    self.last_episode_returns = torch.zeros(self.num_envs, device=self.device)
    self.last_episode_steps = torch.zeros_like(self.task_ids)
    self.last_completed_task_ids = torch.full_like(self.task_ids, -1)
    # Auxiliary episodes use task IDs only for terrain placement. Their whole
    # return is excluded from LP, including movement segments of stop/start.
    self.episode_mode = torch.zeros_like(self.task_ids)
    self.last_completed_episode_mode = torch.full_like(self.task_ids, -1)
    self.last_episode_lp_credited = torch.zeros(
      self.num_envs, device=self.device, dtype=torch.bool
    )
    self._movement_command = torch.zeros_like(self.vel_command_b)
    self._transition_time_left = torch.zeros(self.num_envs, device=self.device)

  def command_protocol_identity(self) -> dict[str, Any] | None:
    """Describe changed sampling semantics while preserving legacy identity."""
    cfg = self.cfg
    if not any((
      cfg.standing_episode_fraction,
      cfg.transition_episode_fraction,
      cfg.zero_yaw_probability,
      cfg.zero_lateral_probability,
    )):
      return None
    return {
      "version": 1,
      "standing_episode_fraction": cfg.standing_episode_fraction,
      "transition_episode_fraction": cfg.transition_episode_fraction,
      "transition_interval_s": tuple(cfg.transition_interval_s),
      "transition_straight_probability": cfg.transition_straight_probability,
      "transition_forward_range": tuple(cfg.transition_forward_range),
      "zero_yaw_probability_in_lowest_bin": cfg.zero_yaw_probability,
      "zero_lateral_probability": cfg.zero_lateral_probability,
      "auxiliary_episode_lp_credit": False,
      "auxiliary_terrain_sampling": "uniform catalog",
    }

  def checkpoint_identity(self) -> dict[str, Any]:
    """Identify task semantics and physical settings independent of batch size.

    Noise enablement may differ in play. Terrain seed is separately checked on
    full training resume so actor evaluation can use a held-out terrain seed.
    """
    cfg = self._env.cfg
    terrain = cfg.scene.terrain
    generator = terrain.terrain_generator if terrain is not None else None
    generator_identity = _config_identity(generator)
    if generator_identity is not None:
      generator_identity.pop("seed", None)
    robot = cfg.scene.entities["robot"]
    identity = {
      **asdict(self.catalog),
      "signed_forward": True,
      "signed_yaw": self.cfg.rough,
      "lateral_range": (-0.1, 0.1) if self.cfg.rough else (0.0, 0.0),
      "environment": {
        "physics": _config_identity(cfg.sim.mujoco),
        "decimation": cfg.decimation,
        "episode_length_s": cfg.episode_length_s,
        "scale_rewards_by_dt": cfg.scale_rewards_by_dt,
        "actions": _config_identity(cfg.actions),
        "robot_articulation": _config_identity(robot.articulation),
        "robot_default_pose": _config_identity(robot.init_state),
        "rewards": _config_identity(cfg.rewards),
        "terminations": _config_identity(cfg.terminations),
        "events": _config_identity(cfg.events),
        "sensors": _config_identity(cfg.scene.sensors),
        # Preserve term order: equal observation dimensions alone do not make
        # policies compatible when the meaning/order of their inputs changes.
        "observations": {
          name: tuple(
            (term_name, _config_identity(term_cfg))
            for term_name, term_cfg in group.terms.items()
          )
          for name, group in cfg.observations.items()
        },
        "terrain_type": None if terrain is None else terrain.terrain_type,
        "terrain_generator": generator_identity,
      },
    }
    protocol = self.command_protocol_identity()
    if protocol is not None:
      identity["command_protocol"] = protocol
    return identity

  def curriculum_state_dict(self) -> dict[str, Any]:
    terrain = self._env.cfg.scene.terrain
    generator = terrain.terrain_generator if terrain is not None else None
    return {
      "task_space": self.checkpoint_identity(),
      "training_terrain_seed": None if generator is None else generator.seed,
      "curriculum": self.curriculum.state_dict(),
    }

  def load_curriculum_state_dict(self, state: dict[str, Any]) -> None:
    if state.get("task_space") != self.checkpoint_identity():
      raise ValueError("LP-ACRL checkpoint task space differs from this environment")
    terrain = self._env.cfg.scene.terrain
    generator = terrain.terrain_generator if terrain is not None else None
    seed = None if generator is None else generator.seed
    if "training_terrain_seed" not in state or state["training_terrain_seed"] != seed:
      raise ValueError("LP-ACRL terrain seed differs; cannot resume its curriculum")
    self.curriculum.load_state_dict(state["curriculum"])

  def set_evaluation(
    self,
    task_ids: int | torch.Tensor | None = None,
    command: tuple[float, float, float] | torch.Tensor | None = None,
  ) -> None:
    """Freeze learning progress; apply optional fixed tasks on the next reset.

    Call ``env.reset()`` afterwards so terrain, root pose and command agree.
    A scalar task is broadcast; a tensor supplies one task per environment.
    """
    self._evaluation = True
    if task_ids is not None:
      ids = torch.as_tensor(task_ids, device=self.device, dtype=torch.long)
      ids = ids.expand(self.num_envs).clone()
      self._validate_task_ids(ids)
      self._forced_task_ids = ids
    if command is not None:
      values = torch.as_tensor(command, device=self.device, dtype=torch.float32)
      if values.shape != (3,) or not torch.isfinite(values).all():
        raise ValueError("Evaluation command must contain three finite values")
      self._forced_command = values

  def force_task_ids(self, env_ids: torch.Tensor, task_ids: torch.Tensor) -> None:
    """Assign evaluation tasks; caller then resets those environments."""
    self._evaluation = True
    ids = torch.as_tensor(task_ids, device=self.device, dtype=torch.long)
    self._validate_task_ids(ids)
    if self._forced_task_ids is None:
      self._forced_task_ids = self.task_ids.clamp(min=0).clone()
    self._forced_task_ids[env_ids] = ids

  def _validate_task_ids(self, ids: torch.Tensor) -> None:
    if ((ids < 0) | (ids >= self.catalog.num_tasks)).any():
      raise ValueError("LP-ACRL task ID outside the task catalog")

  def prepare_episode(self, env_ids: torch.Tensor) -> None:
    """Capture completed returns and choose terrain before any manager resets."""
    completed = (
      (self.task_ids[env_ids] >= 0)
      & (self._env.episode_length_buf[env_ids] > 0)
      & self._env.termination_manager.dones[env_ids]
    )
    completed_ids = env_ids[completed]
    if len(completed_ids):
      # mjlab's logged Episode_Reward values are averages divided by the maximum
      # horizon. They are not per-episode returns and cannot be used for LP.
      returns = torch.stack(
        [
          value[completed_ids]
          for value in self._env.reward_manager._episode_sums.values()
        ]
      ).sum(dim=0)
      self.last_episode_returns[completed_ids] = returns
      self.last_episode_steps[completed_ids] = self._env.episode_length_buf[
        completed_ids
      ]
      self.last_completed_task_ids[completed_ids] = self.task_ids[completed_ids]
      self.last_completed_episode_mode[completed_ids] = self.episode_mode[
        completed_ids
      ]
      eligible = (self.episode_mode[completed_ids] == 0) & (not self._evaluation)
      self.last_episode_lp_credited[completed_ids] = eligible
      if not self._evaluation:
        self.curriculum.record_episodes(
          self.task_ids[completed_ids[eligible]], returns[eligible]
        )
        if self.command_protocol_identity() is not None:
          for mode, name in ((0, "fixed"), (1, "standing"), (2, "stop_start")):
            selected = self.episode_mode[completed_ids] == mode
            self._env.extras["log"][f"LP_ACRL/episode_fraction_{name}"] = (
              selected.float().mean()
            )
            if selected.any():
              self._env.extras["log"][f"LP_ACRL/return_{name}"] = (
                returns[selected].mean()
              )

    if self._forced_task_ids is None:
      self.task_ids[env_ids] = self.curriculum.sample(len(env_ids))
    else:
      self.task_ids[env_ids] = self._forced_task_ids[env_ids]

    self.episode_mode[env_ids] = 0
    if not self._evaluation and (
      self.cfg.standing_episode_fraction + self.cfg.transition_episode_fraction > 0
    ):
      draw = torch.rand(len(env_ids), device=self.device)
      standing = draw < self.cfg.standing_episode_fraction
      transitions = (~standing) & (
        draw
        < self.cfg.standing_episode_fraction + self.cfg.transition_episode_fraction
      )
      self.episode_mode[env_ids[standing]] = 1
      self.episode_mode[env_ids[transitions]] = 2
      auxiliary_ids = env_ids[standing | transitions]
      self.task_ids[auxiliary_ids] = torch.randint(
        self.catalog.num_tasks, (len(auxiliary_ids),), device=self.device
      )

    if self.cfg.rough:
      terrain = self._env.scene.terrain
      assert terrain is not None and terrain.terrain_origins is not None
      coordinates = self._task_indices[self.task_ids[env_ids]]
      types, levels = coordinates[:, 2], coordinates[:, 3]
      terrain.terrain_types[env_ids] = types
      terrain.terrain_levels[env_ids] = levels
      terrain.env_origins[env_ids] = terrain.terrain_origins[levels, types]

  def _resample_command(self, env_ids: torch.Tensor) -> None:
    # The pre-reset curriculum hook already sampled tasks and placed terrain.
    if (self.task_ids[env_ids] < 0).any():
      raise RuntimeError("LP-ACRL pre-reset task hook did not run before command reset")
    bounds = self._task_bounds[self.task_ids[env_ids]]
    sample = bounds[:, :, 0] + torch.rand((len(env_ids), 2), device=self.device) * (
      bounds[:, :, 1] - bounds[:, :, 0]
    )
    signs = torch.randint(0, 2, sample.shape, device=self.device) * 2 - 1
    self.vel_command_b[env_ids, 0] = sample[:, 0] * signs[:, 0]
    self.vel_command_b[env_ids, 1] = (
      torch.empty(len(env_ids), device=self.device).uniform_(-0.1, 0.1)
      if self.cfg.rough
      else 0.0
    )
    self.vel_command_b[env_ids, 2] = sample[:, 1] * signs[:, 1]
    if self.cfg.zero_yaw_probability:
      zero_yaw = (bounds[:, 1, 0] == 0) & (
        torch.rand(len(env_ids), device=self.device) < self.cfg.zero_yaw_probability
      )
      self.vel_command_b[env_ids[zero_yaw], 2] = 0.0
    if self.cfg.zero_lateral_probability:
      zero_lateral = (
        torch.rand(len(env_ids), device=self.device)
        < self.cfg.zero_lateral_probability
      )
      self.vel_command_b[env_ids[zero_lateral], 1] = 0.0
    transition_ids = env_ids[self.episode_mode[env_ids] == 2]
    if len(transition_ids) and self.cfg.transition_straight_probability:
      straight = (
        torch.rand(len(transition_ids), device=self.device)
        < self.cfg.transition_straight_probability
      )
      straight_ids = transition_ids[straight]
      magnitude = torch.empty(len(straight_ids), device=self.device).uniform_(
        *self.cfg.transition_forward_range
      )
      direction = torch.where(self.vel_command_b[straight_ids, 0] < 0, -1.0, 1.0)
      self.vel_command_b[straight_ids, 0] = direction * magnitude
      self.vel_command_b[straight_ids, 1:] = 0.0
    self._movement_command[env_ids] = self.vel_command_b[env_ids]
    self.is_standing_env[env_ids] = self.episode_mode[env_ids] == 1
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
    self.is_standing_env[env_ids] = (
      self.vel_command_b[env_ids].abs().amax(dim=1) <= 1e-6
    )
    self.vel_command_w[env_ids] = self.vel_command_b[env_ids]

  def create_gui(self, *args, **kwargs) -> None:
    super().create_gui(*args, **kwargs)
    if self.cfg.manual_control_on_play and self._evaluation:
      self._joystick_enabled.value = True
      # The inherited sliders start at zero. Apply them immediately so a
      # paused viewer cannot display zero while retaining a sampled command.
      self.compute(0.0)

  def compute(self, dt: float) -> None:
    # Never invoke the uniform command's timer or heading updates: a task's
    # command remains fixed through its entire episode, even across LP stages.
    joystick = getattr(self, "_joystick_enabled", None)
    if self._evaluation and joystick is not None and joystick.value:
      # Honor the inherited Viser controls in playback, including paused
      # resets (dt=0). No GUI exists during training or batch evaluation.
      index = self._joystick_get_env_idx()
      for axis, slider in enumerate(self._joystick_sliders):
        self.vel_command_b[index, axis] = slider.value
      self.vel_command_w[index] = self.vel_command_b[index]
      self.is_standing_env[index] = self.vel_command_b[index].abs().max() <= 1e-6
    if dt <= 0:
      return
    active = self._env.episode_length_buf > 0
    normalizer = max(self._env.max_episode_length, 1)
    self.metrics["error_vel_xy"] += (
      torch.norm(
        self.vel_command_b[:, :2] - self.robot.data.root_link_lin_vel_b[:, :2], dim=1
      )
      * active
      / normalizer
    )
    self.metrics["error_vel_yaw"] += (
      (self.vel_command_b[:, 2] - self.robot.data.root_link_ang_vel_b[:, 2]).abs()
      * active
      / normalizer
    )
    if not self._evaluation:
      update = self.curriculum.advance()
      if update is not None:
        probs = self.curriculum.probabilities
        self._env.extras["log"].update(
          {
            "LP_ACRL/stage": self.curriculum.stage_index,
            "LP_ACRL/probability_min": probs.min(),
            "LP_ACRL/probability_max": probs.max(),
            "LP_ACRL/entropy": -(probs * probs.clamp(min=1e-20).log()).sum(),
            "LP_ACRL/mean_signed_progress": self.curriculum.learning_progress.mean(),
          }
        )
    if not self._evaluation and self.cfg.transition_episode_fraction > 0:
      transitions = self.episode_mode == 2
      self._transition_time_left[transitions] -= dt
      switch_ids = (
        transitions & (self._transition_time_left <= 0)
      ).nonzero(as_tuple=False).flatten()
      if len(switch_ids):
        self.is_standing_env[switch_ids] = ~self.is_standing_env[switch_ids]
        self.vel_command_b[switch_ids] = self._movement_command[switch_ids]
        self.vel_command_b[switch_ids[self.is_standing_env[switch_ids]]] = 0.0
        self.vel_command_w[switch_ids] = self.vel_command_b[switch_ids]
        self._transition_time_left[switch_ids] = torch.empty(
          len(switch_ids), device=self.device
        ).uniform_(*self.cfg.transition_interval_s)


@dataclass(kw_only=True)
class LPACRLVelocityCommandCfg(UniformVelocityCommandCfg):
  rough: bool = False
  adaptive: bool = True
  beta: float = 5.0
  stage_steps: int = 2400
  min_episodes: int = 2
  standing_episode_fraction: float = 0.0
  transition_episode_fraction: float = 0.0
  transition_interval_s: tuple[float, float] = (4.0, 6.0)
  transition_straight_probability: float = 0.0
  transition_forward_range: tuple[float, float] = (0.1, 0.4)
  zero_yaw_probability: float = 0.0
  zero_lateral_probability: float = 0.0
  manual_control_on_play: bool = False

  def __post_init__(self) -> None:
    super().__post_init__()
    for name in (
      "standing_episode_fraction", "transition_episode_fraction",
      "transition_straight_probability", "zero_yaw_probability",
      "zero_lateral_probability",
    ):
      value = getattr(self, name)
      if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{name} must be finite and between zero and one")
    if self.standing_episode_fraction + self.transition_episode_fraction > 1:
      raise ValueError("Auxiliary episode fractions must sum to at most one")
    for name in ("transition_interval_s", "transition_forward_range"):
      lo, hi = getattr(self, name)
      if not (math.isfinite(lo) and math.isfinite(hi) and 0 < lo <= hi):
        raise ValueError(f"{name} must contain finite ordered positive bounds")

  def build(self, env: ManagerBasedRlEnv) -> LPACRLVelocityCommand:
    self.__post_init__()
    return LPACRLVelocityCommand(self, env)


def pre_reset_task(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice,
  command_name: str = "base_velocity",
) -> dict[str, float | torch.Tensor]:
  """mjlab curriculum hook; must remain enabled during play for terrain resets."""
  if isinstance(env_ids, slice):
    env_ids = torch.arange(env.num_envs, device=env.device)[env_ids]
  command = env.command_manager.get_term(command_name)
  assert isinstance(command, LPACRLVelocityCommand)
  command.prepare_episode(env_ids)
  return {"stage": float(command.curriculum.stage_index)}
