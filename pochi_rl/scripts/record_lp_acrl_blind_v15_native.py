"""Record a V15 actor on original rough task360 without touchdown event reward.

The target removes only balanced_touchdown from the V12 environment.
Only the explicit flat catalog/terrain projection then differs from the policy.
The original recorder supplies every action, metric, contact refresh and frame.
This bridge never loads optimizer/LP state or rewrites checkpoint metadata.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

POLICY_TASK = "Pochi-LP-ACRL-Blind-v15"
ENVIRONMENT_TASK = "Pochi-LP-ACRL-Blind-v12"
RECORDER_SHA256 = "5104cea91546e859d408b9dcc616b71bab8a862f68ba5a4061568322b3274368"


def digest(path: Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_native_bridge(saved: dict, rough_task_space: dict, backend: dict) -> dict:
  """Validate full source/target identities; allow only the declared projection."""
  from pochi_rl.mjlab.lp_acrl_flat_focus_identity import _first_difference
  from pochi_rl.mjlab.lp_acrl_no_touchdown_refinement import (
    expected_no_touchdown_rough_task_space,
    expected_no_touchdown_task_space,
    validate_no_touchdown_checkpoint,
  )

  count = validate_no_touchdown_checkpoint(saved)
  infos = saved.get("infos", {})
  if infos.get("lp_acrl_checkpoint_version") != 1:
    raise ValueError("Native bridge requires an independent LP-ACRL checkpoint")
  if infos.get("lp_acrl_contact_backend") != backend:
    raise ValueError("Native bridge contact backend mismatch")
  curriculum = infos.get("lp_acrl", {})
  if (
    "training_terrain_seed" not in curriculum
    or curriculum["training_terrain_seed"] is not None
  ):
    raise ValueError("Flat training terrain seed must be present and None")
  flat = curriculum.get("task_space")
  rough_mismatch = _first_difference(
    expected_no_touchdown_rough_task_space(), rough_task_space,
  )
  flat_mismatch = _first_difference(expected_no_touchdown_task_space(), flat)
  if rough_mismatch is not None or flat_mismatch is not None:
    raise ValueError(
      "Native bridge identity differs outside declared flat projection: "
      + str(rough_mismatch or flat_mismatch)
    )
  return {
    "policy_task": POLICY_TASK,
    "evaluation_environment_task": ENVIRONMENT_TASK,
    "policy_training_task_space": flat,
    "evaluation_task_space": rough_task_space,
    "source_completed_updates": count,
    "saved_training_terrain_seed_is_none": True,
    "allowed_difference": (
      "Only the complete canonical rough22-to-V15 plane22/five-task projection; "
      "all nonterrain physics, motors, observations, actions, rewards and "
      "termination identities match. Original rough task360 remains real."
    ),
    "evaluation_environment_reward_removal": {
      "key": "balanced_touchdown",
      "all_other_original_v12_rewards_retained": True,
    },
    "checkpoint_identity_rewritten": False,
    "actor_and_normalizer_only": True,
    "optimizer_and_curriculum_loaded": False,
  }


def load_native_actor(runner, checkpoint: Path, saved: dict, rough_identity: dict,
                      backend: dict, device: str):
  """Use the native strict actor loader only after the complete bridge gate."""
  from rsl_rl.runners import OnPolicyRunner

  bridge = validate_native_bridge(saved, rough_identity, backend)
  OnPolicyRunner.load(
    runner, str(checkpoint), load_cfg={"actor": True}, strict=True,
    map_location=device,
  )
  return runner.get_inference_policy(device=device), bridge


def parse_args(argv=None):
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--device", default="cuda:2")
  parser.add_argument("--seed", type=int, choices=(1123,), default=1123)
  args = parser.parse_args(argv)
  if not args.checkpoint.is_file():
    parser.error("--checkpoint must name an existing V15 checkpoint")
  if args.output.suffix.lower() != ".mp4":
    parser.error("--output must end in .mp4")
  for path in (
    args.output, args.output.with_suffix(".json"),
    args.output.with_suffix(".provenance.json"),
  ):
    if path.exists():
      parser.error("Refusing to replace a previous take, trace or provenance")
  return args


def main() -> None:
  args = parse_args()
  recorder_path = Path(__file__).with_name("record_lp_acrl_blind_demo.py")
  if digest(recorder_path) != RECORDER_SHA256:
    raise ValueError("Frozen native recorder source changed")
  os.environ["MUJOCO_GL"] = "egl"
  import imageio.v2 as imageio
  import torch
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.utils.torch import configure_torch_backends
  from record_lp_acrl_blind_demo import FPS, _json_safe, record_demo

  from pochi_rl.lp_acrl.physics import PHYSICS_LIMITS
  from pochi_rl.mjlab.lp_acrl_blind_v12 import register_blind_v12
  from pochi_rl.mjlab.lp_acrl_no_touchdown_refinement import (
    CORRUPTION_IDENTITY,
    STATE_KEY,
    expected_exploration_identity,
    expected_no_touchdown_rough_task_space,
    expected_symmetry_identity,
  )
  from pochi_rl.mjlab.warp_contact_backend import (
    active_contact_backend_identity,
    install_contact_backend,
  )

  backend = install_contact_backend()
  if active_contact_backend_identity() != backend:
    raise ValueError("Installed backend does not match the active backend")
  saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  checksum = digest(args.checkpoint)
  preflight = validate_native_bridge(
    saved, expected_no_touchdown_rough_task_space(), backend,
  )
  register_blind_v12()
  configure_torch_backends()
  cfg = load_env_cfg(ENVIRONMENT_TASK, play=True)
  if "balanced_touchdown" not in cfg.rewards:
    raise ValueError("Original V12 environment is missing touchdown reward")
  del cfg.rewards["balanced_touchdown"]
  cfg.scene.num_envs = 1
  cfg.seed = args.seed
  cfg.auto_reset = False
  cfg.commands["base_velocity"].debug_vis = False
  cfg.viewer.width, cfg.viewer.height = 960, 540
  cfg.viewer.max_extra_envs = 0
  cfg.viewer.distance, cfg.viewer.elevation, cfg.viewer.azimuth = 1.6, -20.0, 130.0
  agent = load_rl_cfg(ENVIRONMENT_TASK)
  args.output.parent.mkdir(parents=True, exist_ok=True)
  provenance_path = args.output.with_suffix(".provenance.json")
  provenance = {
    "status": "running", "started_at_utc": datetime.now(UTC).isoformat(),
    "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checksum,
    "policy_task": POLICY_TASK, "evaluation_environment_task": ENVIRONMENT_TASK,
    "native_task_id": 360, "seed": args.seed,
    "fresh_native_seed1123": True, "prior_v13_native_seed123_reused": False,
    "contact_backend": backend, "canonical_bridge_checked_before_env": True,
    "source_sha256": {
      Path(__file__).name: digest(Path(__file__)),
      recorder_path.name: RECORDER_SHA256,
    },
  }
  with provenance_path.open("x") as stream:
    json.dump(provenance, stream, indent=2)
    stream.write("\n")
  env = None
  try:
    env = ManagerBasedRlEnv(cfg=cfg, device=args.device, render_mode="rgb_array")
    term = env.command_manager.get_term("base_velocity")
    rough_identity = term.checkpoint_identity()
    actual = validate_native_bridge(saved, rough_identity, backend)
    if actual != preflight:
      raise ValueError("Constructed native rough environment differs from preflight")
    if term.catalog.num_tasks != 600:
      raise ValueError("Native task360 requires the original 600-task catalog")
    index = term.catalog.task_indices()[360].tolist()
    if index[2:] != [0, 0] or term.catalog.terrains[0] != "ascending_stairs":
      raise ValueError("Native task360 must be level0 ascending stairs")
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = load_runner_cls(ENVIRONMENT_TASK)(
      wrapped, asdict(agent), device=args.device,
    )
    policy, bridge = load_native_actor(
      runner, args.checkpoint, saved, rough_identity, backend, args.device,
    )
    if digest(args.checkpoint) != checksum:
      raise ValueError("Checkpoint changed during native actor loading")
    with imageio.get_writer(
      str(args.output), fps=FPS, codec="libx264", quality=7, macro_block_size=2,
    ) as writer:
      report = record_demo(env, wrapped, policy, writer=writer)
    if (
      active_contact_backend_identity() != backend
      or digest(args.checkpoint) != checksum
    ):
      raise ValueError("Backend or source checkpoint changed during native recording")
    report.update({
      "task": ENVIRONMENT_TASK, "policy_task": POLICY_TASK,
      "evaluation_environment_task": ENVIRONMENT_TASK,
      "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checksum,
      "seed": args.seed, "device": args.device, "video": str(args.output.resolve()),
      "height_scan": False, "observation_dim": 48, "action_dim": 12,
      "action_clip": agent.clip_actions, "physics_limits": PHYSICS_LIMITS,
      "physics_dt_s": cfg.sim.mujoco.timestep, "decimation": cfg.decimation,
      "physics_solver": cfg.sim.mujoco.solver, "contact_backend": backend,
      "observation_corruption": CORRUPTION_IDENTITY,
      "training_exploration": expected_exploration_identity(),
      "training_symmetry": expected_symmetry_identity(),
      "no_touchdown_state": saved["infos"][STATE_KEY], "native_actor_bridge": bridge,
      "source_sha256": provenance["source_sha256"],
      "native_measurements_profiles_actions_and_thresholds_unchanged": True,
    })
    args.output.with_suffix(".json").write_text(
      json.dumps(_json_safe(report), allow_nan=False) + "\n"
    )
    provenance.update(
      status="completed", report_sha256=digest(args.output.with_suffix(".json")),
    )
    print(json.dumps({k: report[k] for k in ("complete", "stop_reason", "segments")},
                     indent=2, allow_nan=False))
  except BaseException as error:
    provenance.update(
      status="failed", error_type=type(error).__name__, error=str(error),
    )
    raise
  finally:
    try:
      if env is not None:
        env.close()
    finally:
      provenance["finished_at_utc"] = datetime.now(UTC).isoformat()
      provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")


if __name__ == "__main__":
  main()
