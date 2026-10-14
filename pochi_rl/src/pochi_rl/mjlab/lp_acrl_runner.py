"""Independent PPO runner and checkpoint format for the LP-ACRL experiments."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import MethodType

import torch
from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from pochi_rl.lp_acrl.physics import (
  PHYSICS_BODY_FIELDS,
  PHYSICS_LIMITS,
  physics_fault_masks,
)
from pochi_rl.mjlab.lp_acrl_symmetry import BLIND_OBSERVATION_TERMS
from pochi_rl.mjlab.runner import PochiOnPolicyRunner
from pochi_rl.robot.pochi_constants import JOINT_NAMES

PAPER_URL = "https://arxiv.org/abs/2601.17428"
REPLAY_HISTORY_LIMIT = 16
REPLAY_STATE_FIELDS = (
  "qpos", "qvel", "act", "mocap_pos", "mocap_quat", "ctrl", "qacc_warmstart"
)


def _capture_with_replay_inputs(guard, data) -> None:
  """Extend this instance's pre-step history with controls and solver warmstart."""
  guard._lp_acrl_original_capture(data)
  if guard.enabled and guard.buffer:
    guard.buffer[-1]["ctrl"] = data.ctrl.clone()
    guard.buffer[-1]["qacc_warmstart"] = data.qacc_warmstart.clone()


def _fault_replay_payload(env, ids: torch.Tensor, directory: Path) -> dict:
  """Copy only selected worlds and a bounded history; never serialize the model."""
  sim = getattr(env, "sim", None)
  guard = getattr(sim, "nan_guard", None)
  history = []
  for state in list(getattr(guard, "buffer", ()))[-REPLAY_HISTORY_LIMIT:]:
    frame = {"step": int(state["step"])}
    for name in REPLAY_STATE_FIELDS:
      value = state.get(name)
      if value is not None:
        frame[name] = value[ids.to(value.device)].detach().cpu()
    history.append(frame)

  model = getattr(sim, "model", None)
  parameters = {}
  parameter_shapes = {}
  skipped_parameters = {}
  for name, rank in (("body_mass", 2), ("geom_friction", 3)):
    value = getattr(model, name, None)
    if value is None:
      continue
    parameter_shapes[name] = list(value.shape)
    if value.ndim == rank - 1:
      value = value.unsqueeze(0)
    if value.ndim != rank or value.shape[0] not in (1, env.num_envs):
      skipped_parameters[name] = "Unexpected model tensor shape"
      continue
    rows = ids.to(value.device) if value.shape[0] > 1 else torch.zeros_like(ids)
    parameters[name] = value[rows.to(value.device)].detach().cpu()

  manifest_path = directory / "run.json"
  source_manifest = None
  manifest_error = None
  if manifest_path.exists():
    try:
      source_manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
      manifest_error = str(exc)
  return {
    "env_ids": ids.detach().cpu(),
    "history": history,
    "history_limit": REPLAY_HISTORY_LIMIT,
    "capture_timing": "Before each physics substep; chronological oldest first",
    "guard_enabled": bool(getattr(guard, "enabled", False)),
    "guard_state_spec": getattr(guard, "state_spec", None),
    "guard_state_size": getattr(guard, "state_size", None),
    "guard_step_counter": getattr(guard, "step_counter", None),
    "control_decimation": getattr(getattr(env, "cfg", None), "decimation", None),
    "model_parameters": parameters,
    "model_parameter_original_shapes": parameter_shapes,
    "skipped_model_parameters": skipped_parameters,
    "model_parameter_timing": (
      "At failure; older history frames may precede an episode reset/randomization"
    ),
    "source_manifest_path": str(manifest_path),
    "source_manifest": source_manifest,
    "source_manifest_error": manifest_error,
  }


@dataclass
class LPACRLSymmetricPpoCfg(RslRlPpoAlgorithmCfg):
  """Expose RSL-RL's supported symmetry extension through mjlab's dataclass."""

  symmetry_cfg: dict | None = None


def pochi_lp_acrl_ppo_runner_cfg(
  *, rough: bool = False, blind: bool = False
) -> RslRlOnPolicyRunnerCfg:
  """Pochi choices: the paper does not publish network/PPO hyperparameters."""
  algorithm = LPACRLSymmetricPpoCfg if blind else RslRlPpoAlgorithmCfg
  symmetry = {
    "symmetry_cfg": {
      "data_augmentation_func": (
        "pochi_rl.mjlab.lp_acrl_symmetry:augment_blind_symmetry"
      ),
      "use_data_augmentation": True,
      "use_mirror_loss": True,
      "mirror_loss_coeff": 1.0,
    }
  } if blind else {}
  return RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 0.35 if blind else 1.0,
        "std_type": "scalar",
        **({"std_range": (0.10, 1.0e6)} if blind else {}),
      },
    ),
    critic=RslRlModelCfg(
      hidden_dims=(256, 128, 64),
      activation="elu",
      obs_normalization=True,
    ),
    algorithm=algorithm(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.005,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
      **symmetry,
    ),
    experiment_name=(
      "pochi_lp_acrl_blind" if blind
      else "pochi_lp_acrl_rough" if rough else "pochi_lp_acrl_flat"
    ),
    run_name="scratch",
    logger="tensorboard",
    upload_model=False,
    resume=False,
    clip_actions=4.0,
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=3000,
  )


def _log_with_all_episode_keys(logger, *args, **kwargs):
  """Make later rollout diagnostics visible without adding artificial samples.

  RSL-RL 5.2 iterates only the first episode-info dictionary's keys. That first
  control step often has no completed episodes, hiding later failures and LP
  updates. Empty tensors expose the union of keys but do not bias the means.
  """
  if logger.ep_extras:
    first = dict(logger.ep_extras[0])
    for packet in logger.ep_extras[1:]:
      for key in packet:
        first.setdefault(key, torch.empty(0, device=logger.device))
    logger.ep_extras[0] = first
  return logger._lp_acrl_original_log(*args, **kwargs)


def validate_blind_symmetry_cfg(cfg) -> None:
  """Reject reordered inputs/actions before applying the hard-coded reflection."""
  for name in ("actor", "critic"):
    if tuple(cfg.observations[name].terms) != BLIND_OBSERVATION_TERMS:
      raise ValueError("Blind symmetry requires the verified48D observation order")
  if tuple(cfg.actions["joint_pos"].actuator_names) != tuple(JOINT_NAMES):
    raise ValueError("Blind symmetry requires FL,FR,RL,RR joint order")


class LPACRLOnPolicyRunner(PochiOnPolicyRunner):
  """Persist task statistics and refuse accidental legacy-policy loading.

  A single GPU is deliberate: this implementation does not aggregate episodic
  task statistics across distributed workers. Each checkpoint includes the
  curriculum as well as the usual actor, critic, optimizer and normalizers.
  """

  def __init__(self, *args, **kwargs):
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
      raise ValueError("LP-ACRL currently supports one GPU per training run")
    super().__init__(*args, **kwargs)
    if self.alg.symmetry is not None:
      validate_blind_symmetry_cfg(self.env.unwrapped.cfg)
    # Source snapshots and hashes live in run.json/source. RSL-RL discovers
    # this repository through its in-tree venv and otherwise copies 127 MiB
    # of unrelated generated CAD mesh changes into every experiment.
    self.logger.git_status_repos = []
    self.logger._lp_acrl_original_log = self.logger.log
    self.logger.log = MethodType(_log_with_all_episode_keys, self.logger)
    guard = self.env.unwrapped.sim.nan_guard
    if guard.enabled and not hasattr(guard, "_lp_acrl_original_capture"):
      guard._lp_acrl_original_capture = guard.capture
      guard.capture = MethodType(_capture_with_replay_inputs, guard)

  def add_git_repo_to_log(self, repo_file_path: str) -> None:
    """The dedicated launcher records compact, reproducible source snapshots."""
    return None

  @property
  def command_term(self):
    return self.env.unwrapped.command_manager.get_term("base_velocity")

  def learn(
    self, num_learning_iterations: int, init_at_random_ep_len: bool = False
  ) -> None:
    # A randomly shortened first episode must not become a low-return sample
    # in the LP estimator. All episodes start with the full physical horizon.
    try:
      super().learn(num_learning_iterations, init_at_random_ep_len=False)
    except ValueError as exc:
      self._record_failure(reason=str(exc))
      raise

  def _record_failure(self, reason: str | None = None) -> None:
    """Preserve finite explosions as well as nonfinite state; never repair it."""
    env = self.env.unwrapped
    fields = {f"observation/{k}": v for k, v in env.obs_buf.items()}
    robot = env.scene["robot"]
    for name in (
      "root_link_pos_w",
      "root_link_quat_w",
      "root_link_lin_vel_b",
      "root_link_ang_vel_b",
      "joint_pos",
      "joint_vel",
      "joint_acc",
    ):
      fields[name] = getattr(robot.data, name)
    for name in PHYSICS_BODY_FIELDS:
      value = getattr(robot.data, name, None)
      if value is not None:
        fields[name] = value
    fields["action"] = env.action_manager.action
    previous_action = getattr(env.action_manager, "prev_action", None)
    if previous_action is not None:
      fields["previous_action"] = previous_action
    fields["reward"] = env.reward_buf
    if "terrain_scan" in getattr(env.scene, "sensors", env.scene):
      scan = env.scene["terrain_scan"].data
      fields["ray_distances"] = scan.distances
      fields["ray_hit_pos_w"] = scan.hit_pos_w
      fields["ray_frame_pos_w"] = scan.frame_pos_w
    bad = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    counts = {}
    for name, value in fields.items():
      mask = ~torch.isfinite(value)
      counts[name] = int(mask.sum())
      bad |= mask.reshape(env.num_envs, -1).any(dim=1)
    faults = physics_fault_masks(robot.data)
    for mask in faults.values():
      bad |= mask
    stored_ids = getattr(env, "_lp_acrl_physics_fault_ids", None)
    if stored_ids is not None:
      bad[stored_ids] = True
    ids = bad.nonzero(as_tuple=False).flatten()[:32]
    summary = {
      "reason": reason or "LP-ACRL training rejected invalid state",
      "physics_limits": PHYSICS_LIMITS,
      "physics_fault_counts": {name: int(mask.sum()) for name, mask in faults.items()},
      "physics_fault_reasons": {
        str(env_id): [name for name, mask in faults.items() if bool(mask[env_id])]
        for env_id in ids.tolist()
      },
      "total_fault_envs": int(bad.sum()),
      "saved_env_limit": 32,
      "iteration": self.current_learning_iteration,
      "step": env.common_step_counter,
      "nonfinite_counts": counts,
      "env_ids": ids.tolist(),
      "task_ids": self.command_term.task_ids[ids].tolist(),
    }
    directory = Path(self.logger.log_dir)
    replay = _fault_replay_payload(env, ids, directory)
    summary["replay"] = {
      "history_frames": len(replay["history"]),
      "history_limit": REPLAY_HISTORY_LIMIT,
      "history_steps": [frame["step"] for frame in replay["history"]],
      "guard_state_spec": replay["guard_state_spec"],
      "guard_state_size": replay["guard_state_size"],
      "model_parameter_original_shapes": replay["model_parameter_original_shapes"],
      "skipped_model_parameters": replay["skipped_model_parameters"],
      "previous_action_available": previous_action is not None,
      "source_manifest_path": replay["source_manifest_path"],
      "source_manifest_error": replay["source_manifest_error"],
    }
    (directory / "failure.json").write_text(json.dumps(summary, indent=2) + "\n")
    torch.save(
      {
        **{name: value[ids].detach().cpu() for name, value in fields.items()},
        "replay": replay,
      },
      directory / "failure.pt",
    )

  def save(self, path: str, infos=None) -> None:
    infos = {
      **(infos or {}),
      "lp_acrl": self.command_term.curriculum_state_dict(),
      "paper": PAPER_URL,
      "lp_acrl_checkpoint_version": 1,
      "lp_acrl_training_symmetry": self.symmetry_identity(),
    }
    super().save(path, infos)

  def symmetry_identity(self) -> dict | None:
    symmetry = getattr(self.alg, "symmetry", None)
    if symmetry is None:
      return None
    func = symmetry.data_augmentation_func
    return {
      "data_augmentation_func": f"{func.__module__}:{func.__qualname__}",
      "use_data_augmentation": symmetry.use_data_augmentation,
      "use_mirror_loss": symmetry.use_mirror_loss,
      "mirror_loss_coeff": symmetry.mirror_loss_coeff,
    }

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    saved = torch.load(Path(path), map_location="cpu", weights_only=False)
    metadata = saved.get("infos") or {}
    if metadata.get("lp_acrl_checkpoint_version") != 1:
      raise ValueError(
        "Expected an independent LP-ACRL checkpoint. Legacy velocity, deliberate "
        "walk and stand-up weights cannot initialize this experiment."
      )
    if "lp_acrl" not in metadata:
      raise ValueError("LP-ACRL checkpoint is missing curriculum state")
    actor_only = load_cfg is not None and not load_cfg.get("optimizer", False)
    if not actor_only and metadata.get("lp_acrl_training_symmetry") != (
      self.symmetry_identity()
    ):
      raise ValueError("LP-ACRL training symmetry differs; cannot fully resume")
    saved_identity = metadata["lp_acrl"].get("task_space")
    current_identity = self.command_term.checkpoint_identity()
    if actor_only and isinstance(saved_identity, dict):
      # Rewards do not enter actor inference. This permits ordinary `play`
      # for a checkpoint trained with explicit reward-kernel adaptations.
      # The evaluator reconstructs the saved rewards to report honest returns.
      saved_identity = {
        **saved_identity,
        "environment": {
          key: value
          for key, value in saved_identity["environment"].items()
          if key != "rewards"
        },
      }
      current_identity = {
        **current_identity,
        "environment": {
          key: value
          for key, value in current_identity["environment"].items()
          if key != "rewards"
        },
      }
    if saved_identity != current_identity:
      raise ValueError("LP-ACRL checkpoint task space differs from this environment")
    # Playback can use different sampler settings because it does not train.
    # Full resume validates/restores the entire partial curriculum stage.
    old_state = self.command_term.curriculum_state_dict()
    if not actor_only:
      self.command_term.load_curriculum_state_dict(metadata["lp_acrl"])
    try:
      infos = super().load(path, load_cfg, strict, map_location)
    except Exception:
      self.command_term.load_curriculum_state_dict(old_state)
      raise
    return infos
