"""Validated-plane standing/walking diagnostic, protocol plane-v2.

Reuse the frozen behavior profiles, metrics and thresholds without editing them.
The only new physical override is a verified single infinite plane at z=0.
Fresh post-final-forward contacts are aligned with sampled robot state.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

_FROZEN_PATH = Path(__file__).with_name("eval_lp_acrl_behavior.py")
_SPEC = importlib.util.spec_from_file_location("frozen_lp_behavior", _FROZEN_PATH)
assert _SPEC is not None and _SPEC.loader is not None
frozen = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(frozen)
PROFILES = frozen.PROFILES
FOOT_ORDER = frozen.FOOT_ORDER
MIN_AIR_TIME_S = frozen.MIN_AIR_TIME_S
STAND_LIMITS = frozen.STAND_LIMITS
WALK_LIMITS = frozen.WALK_LIMITS
PROTOCOL = "pochi-lp-acrl-behavior-plane-v2"
PLANE_OVERRIDE_REASON = (
  "Replace terrain subfamilies with flat placeholders preserving the task-origin "
  "table, then convert exactly one terrain geom into an infinite horizontal "
  "plane at z=0 and disable collision on every other terrain geom. Verify the "
  "compiled ground before rollout. Original 24m boxes caused Warp collision "
  "misses and default-PD jitter; original training-platform and plane controls "
  "stood stably. Historical box results remain geometry-confounded evidence. "
  "Robot CAD, mass, motors, physics, action/observation identities and profile "
  "criteria are unchanged. Rewards are not graded. Checkpoint is unmodified."
)


def install_single_plane(spec):
  """Change terrain geoms only; retain robot, task origins and sensor bindings."""
  import mujoco

  terrain = spec.body("terrain")
  if terrain is None:
    raise ValueError("Diagnostic scene must contain terrain body")
  geoms = list(terrain.geoms)
  if not geoms:
    raise ValueError("Diagnostic terrain contains no geometry")
  for index, geom in enumerate(geoms):
    if index == 0:
      geom.type = mujoco.mjtGeom.mjGEOM_PLANE
      geom.size = (0.0, 0.0, 0.1)
      geom.pos[2] = 0.0
      geom.quat = (1.0, 0.0, 0.0, 0.0)
      geom.contype = 1
      geom.conaffinity = 1
    else:
      geom.contype = 0
      geom.conaffinity = 0


def verify_single_plane(model):
  """Fail if compiled collision geometry differs from the documented override."""
  import mujoco

  terrain_id = model.body("terrain").id
  ids = np.flatnonzero(model.geom_bodyid == terrain_id)
  colliding = [
    int(gid) for gid in ids if model.geom_contype[gid] or model.geom_conaffinity[gid]
  ]
  if len(colliding) != 1:
    raise ValueError("Plane protocol requires exactly one colliding terrain geom")
  gid = colliding[0]
  if model.geom_type[gid] != mujoco.mjtGeom.mjGEOM_PLANE:
    raise ValueError("Plane protocol ground must be an infinite plane")
  data = mujoco.MjData(model)
  mujoco.mj_kinematics(model, data)
  normal = data.geom_xmat[gid].reshape(3, 3)[:, 2]
  if not np.allclose(normal, (0, 0, 1), atol=1e-8):
    raise ValueError("Diagnostic plane must be horizontal and upward-facing")
  if not np.isclose(data.geom_xpos[gid, 2], 0.0, atol=1e-8):
    raise ValueError("Diagnostic plane must have top z=0")
  return {
    "kind": "single_infinite_horizontal_plane",
    "plane_geom_id": gid,
    "plane_geom_name": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid),
    "top_z_m": float(data.geom_xpos[gid, 2]),
    "normal_world": normal.tolist(),
    "collision_disabled_other_terrain_geoms": len(ids) - 1,
    "robot_geometry_override": False,
    "checkpoint_scene_callback_not_in_legacy_identity": (
      "Explicit diagnostic override verified against compiled ground; "
      "not represented as an identical training scene"
    ),
  }


class FreshContactStepWrapper:
  """Invalidate only contact read caches after finalforward; no extra physics."""

  def __init__(self, env, wrapped):
    self.env = env
    self.wrapped = wrapped

  def step(self, actions):
    result = self.wrapped.step(actions)
    for name in ("feet_ground_contact", "limb_ground_contact"):
      self.env.scene[name]._invalidate_cache()
    return result


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--task", default="Pochi-LP-ACRL-Blind-v0")
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--device", default="cuda:2")
  parser.add_argument("--num-envs", type=int, default=32)
  parser.add_argument("--seed", type=int, default=456)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  if args.output.exists():
    parser.error("--output already exists; preserve every earlier diagnostic result")
  if args.num_envs < 4 or args.num_envs % 4:
    parser.error("--num-envs must be a positive multiple of four")
  if not args.checkpoint.is_file():
    parser.error("--checkpoint must name an existing file")
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.rl import RslRlVecEnvWrapper
  from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
  from mjlab.terrains.primitive_terrains import BoxFlatTerrainCfg
  from mjlab.utils.torch import configure_torch_backends
  from rsl_rl.runners import OnPolicyRunner

  import pochi_rl  # noqa: F401
  from pochi_rl.robot import PAYLOAD_MASS_KG, PAYLOAD_POS

  configure_torch_backends()
  saved = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
  infos = saved.get("infos", {})
  if infos.get("lp_acrl_checkpoint_version") != 1:
    parser.error("Expected an independent LP-ACRL checkpoint")
  identity = infos["lp_acrl"]["task_space"]
  physical = identity["environment"]
  cfg = load_env_cfg(args.task, play=True)
  for key, value in physical["physics"].items():
    if key != "type":
      setattr(cfg.sim.mujoco, key, value)
  cfg.decimation = physical["decimation"]
  saved_sensors = {sensor["name"]: sensor for sensor in physical["sensors"]}
  for sensor in cfg.scene.sensors:
    if sensor.name in saved_sensors and "history_length" in saved_sensors[sensor.name]:
      sensor.history_length = saved_sensors[sensor.name]["history_length"]
  generator = cfg.scene.terrain.terrain_generator
  if generator is not None:
    generator.sub_terrains = {
      name: BoxFlatTerrainCfg(size=family.size, proportion=family.proportion)
      for name, family in generator.sub_terrains.items()
    }
  if cfg.scene.spec_fn is not None:
    raise ValueError("Refuse to replace an existing scene-spec callback")
  cfg.scene.spec_fn = install_single_plane
  cfg.seed = args.seed
  cfg.scene.num_envs = args.num_envs
  cfg.auto_reset = False
  cfg.commands["base_velocity"].debug_vis = False
  agent = load_rl_cfg(args.task)
  env = ManagerBasedRlEnv(cfg=cfg, device=args.device)
  try:
    term = env.command_manager.get_term("base_velocity")
    overrides = frozen.validate_behavior_identity(identity, term.checkpoint_identity())
    geometry = verify_single_plane(env.sim.mj_model)
    overrides["compiled_ground_override"] = geometry
    overrides["reason"] = PLANE_OVERRIDE_REASON
    wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
    runner = load_runner_cls(args.task)(wrapped, asdict(agent), device=args.device)
    # The recorded narrow identity check above replaces only the custom runner's
    # equality check; this is the native strict actor+normalizer loading API.
    OnPolicyRunner.load(
      runner,
      str(args.checkpoint),
      load_cfg={"actor": True},
      strict=True,
      map_location=args.device,
    )
    rows = frozen.evaluate_profiles(
      env,
      FreshContactStepWrapper(env, wrapped),
      runner.get_inference_policy(device=args.device),
    )
    robot = env.scene["robot"]
    masses = env.sim.model.body_mass
    bodies = robot.indexing.body_ids.long()
    base_local, _ = robot.find_bodies(("base_link",), preserve_order=True)
    base_global = bodies[base_local[0]]
    total_mass = masses[:, bodies].sum(-1).detach().cpu().tolist()
    base_mass = masses[:, base_global].detach().cpu().tolist()
    summary = {
      profile: {
        "episodes": sum(row["profile"] == profile for row in rows),
        "passed": sum(row["profile"] == profile and row["passed"] for row in rows),
        "falls": sum(row["profile"] == profile and row["fall"] for row in rows),
        "invalid": sum(
          row["profile"] == profile and row["invalid_state"] for row in rows
        ),
      }
      for profile in PROFILES
    }
    report = {
      "protocol": PROTOCOL,
      "source_sha256": {
        str(path.name): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (Path(__file__), Path(frozen.__file__))
      },
      "task": args.task,
      "checkpoint": str(args.checkpoint.resolve()),
      "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
      "seed": args.seed,
      "num_envs": env.num_envs,
      "device": args.device,
      "control_dt_s": env.step_dt,
      "horizon_s": 20.0,
      "identity_overrides": overrides,
      "observation_terms": {
        group: list(group_cfg.terms) for group, group_cfg in cfg.observations.items()
      },
      "height_scan_present": any(
        "height" in name.lower()
        for group in cfg.observations.values()
        for name in group.terms
      ),
      "payload": {
        "central_ballast_already_in_robot_asset_kg": PAYLOAD_MASS_KG,
        "position_in_base_frame_m": PAYLOAD_POS,
        "realized_robot_mass_kg": total_mass,
        "realized_base_mass_kg": base_mass,
      },
      "metric_conventions": {
        "purpose": "Flat-terrain behavior diagnostics; not paper scores",
        "profile_commands": (
          "stand 0; forward +.25; backward -.25; "
          "stop/start 0,+.25,0,+.25 in 5s blocks; lateral/yaw 0"
        ),
        "transient_exclusion_s": {
          "constant_profiles": 2.0,
          "stop_start_each_segment": 1.0,
        },
        "standing_limits": STAND_LIMITS,
        "walking_limits": WALK_LIMITS,
        "criteria_origin": (
          "Explicit Pochi engineering acceptance conventions; "
          "not paper criteria or hardware guarantees"
        ),
        "sampling": (
          "50Hz post-control-step, before manual terminal reset; first episode "
          "only; invalid and fall samples excluded"
        ),
        "contact": (
          "Fresh post-final-forward contact sensor found flag; "
          "sensor caches invalidated "
          "after env.step without advancing physics or air-time histories; "
          "brief sub-control-step contacts may be missed"
        ),
        "footfall_minimum_observed_air_time_s": MIN_AIR_TIME_S,
        "bouts": (
          "Observed run lengths; boundary bouts marked censored; "
          "no inference about unobserved swing phases"
        ),
        "drift": "Distance from first valid post-transient sample in each segment",
        "foot_order": FOOT_ORDER,
        "rewards": (
          "Reward values are not reported or graded; "
          "nonfinite reward still invalidates a transition"
        ),
      },
      "summary": summary,
      "episodes": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"output": str(args.output), "summary": summary}, indent=2))
  finally:
    env.close()


if __name__ == "__main__":
  main()
