"""Export the pinned V19 mean actor on CPU, with its deployment IO contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import torch
from tensordict import TensorDict

from pochi_rl.mjlab.lp_acrl_blind_v19_env_cfg import (
  pochi_lp_acrl_blind_v19_env_cfg,
  pochi_lp_acrl_blind_v19_ppo_runner_cfg,
)
from pochi_rl.mjlab.lp_acrl_phase_gait import PHASE_CONTRACT, PhaseQuietStopMLP
from pochi_rl.mjlab.lp_acrl_speed_runner import validate_speed_checkpoint
from pochi_rl.robot import DEFAULT_JOINT_POS, JOINT_NAMES

ROOT = Path(__file__).resolve().parents[1]
OBS_DIMS = (3, 3, 3, 3, 12, 12, 12, 2)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--checkpoint", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  args = parser.parse_args()
  torch.set_num_threads(1)
  checkpoint = args.checkpoint.resolve(strict=True)
  pins = json.loads((checkpoint.parent / "run.json").read_text())["source_sha256"]
  for name, digest in pins.items():
    if hashlib.sha256((ROOT / name).read_bytes()).hexdigest() != digest:
      raise ValueError(f"Frozen training source changed: {name}")
  saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
  count = validate_speed_checkpoint(saved)
  cfg = pochi_lp_acrl_blind_v19_env_cfg(play=True)
  agent = pochi_lp_acrl_blind_v19_ppo_runner_cfg()
  actor_cfg = asdict(agent.actor)
  example = TensorDict({"actor": torch.zeros(1, 50)}, batch_size=[1])
  actor = PhaseQuietStopMLP(
    example,
    {"actor": ["actor"]},
    "actor",
    12,
    **{
      key: actor_cfg[key]
      for key in ("hidden_dims", "activation", "obs_normalization", "distribution_cfg")
    },
  ).eval()
  actor.load_state_dict(saved["actor_state_dict"], strict=True)
  args.output.mkdir(parents=True, exist_ok=True)
  model_path = args.output / f"policy_v19_model{count - 1}.torchscript.pt"
  manifest_path = args.output / "policy_contract.json"
  replay_path = args.output / "export_validation_inputs.pt"
  if any(p.exists() for p in (model_path, manifest_path, replay_path)):
    raise FileExistsError("Export output already exists; select a new directory")
  exported = torch.jit.script(actor.as_jit().eval())
  exported.save(str(model_path))
  reloaded = torch.jit.load(str(model_path), map_location="cpu").eval()
  rng = torch.Generator().manual_seed(20261009)
  inputs = torch.randn(4096, 50, generator=rng)
  inputs[:512, 9:12] = 0.0
  inputs[:512, 48:50] = 0.0
  with torch.inference_mode():
    expected = actor(TensorDict({"actor": inputs}, batch_size=[len(inputs)]))
    actual = reloaded(inputs)
  torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
  torch.save({"observations": inputs, "actions": expected}, replay_path)
  independent = subprocess.check_output(
    [
      sys.executable,
      "-c",
      "import json,sys,torch; torch.set_num_threads(1); torch.set_grad_enabled(False); "
      "m=torch.jit.load(sys.argv[1],map_location='cpu').eval(); "
      "d=torch.load(sys.argv[2],map_location='cpu',weights_only=True); "
      "a=m(d['observations']); "
      "torch.testing.assert_close(a,d['actions'],rtol=0,atol=1e-6); "
      "print(json.dumps({'max_abs_error':float((a-d['actions']).abs().max())}))",
      str(model_path.resolve()),
      str(replay_path.resolve()),
    ],
    text=True,
  )
  layout, start = [], 0
  for name, dim in zip(cfg.observations["actor"].terms, OBS_DIMS, strict=True):
    layout.append({"name": name, "start": start, "stop": start + dim})
    start += dim
  assert start == 50
  actuator = cfg.scene.entities["robot"].articulation.actuators[0]
  # Read from 192.168.201.172:/home/continue/Pochi on 2026-10-09.
  # These are deployment mappings, not changes to the frozen training frame.
  motor_ids = (2, 1, 0, 11, 10, 9, 5, 4, 3, 8, 7, 6)
  motor_sign = (-1, -1, -1, -1, 1, 1, 1, -1, -1, 1, 1, 1)
  manifest = {
    "task": "Pochi-LP-ACRL-Blind-v19",
    "checkpoint": str(checkpoint),
    "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
    "completed_updates": count,
    "model": model_path.name,
    "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    "input": {"dtype": "float32", "shape": ["batch", 50], "layout": layout},
    "output": {"dtype": "float32", "shape": ["batch", 12], "kind": "mean action"},
    "normalization": "Included in model: (raw_obs - saved_mean)/(saved_std + 0.01)",
    "policy_hz": 1 / (cfg.sim.mujoco.timestep * cfg.decimation),
    "phase": PHASE_CONTRACT,
    "phase_clock": (
      "Start at 0; advance 1.5*dt cycles while max(abs(command))>1e-6; "
      "reset to 0 at stop; observation [sin(2*pi*cycles),cos(2*pi*cycles)] "
      "while moving, [0,0] at stop"
    ),
    "joint_names": JOINT_NAMES,
    "default_joint_pos_policy_frame": [DEFAULT_JOINT_POS[name] for name in JOINT_NAMES],
    "action_scale": cfg.actions["joint_pos"].scale,
    "clip_actions": agent.clip_actions,
    "last_action": (
      "Previous clipped action in policy coordinates, before motor sign "
      "conversion; initially 12 zeros"
    ),
    "body_frame": (
      "x forward, y left, z up; rad/s angular velocity; unit gravity direction, "
      "not acceleration scaled by 9.81"
    ),
    "hardware_mapping_from_remote_source": {
      "host": "192.168.201.172",
      "motor_ids_in_policy_order": motor_ids,
      "motor_sign_in_policy_order": motor_sign,
      "q_policy": "motor_sign * calibrated_q_motor",
      "q_target_motor": (
        "motor_sign * (q_default_policy + 0.25*clip(action,-4,4)); "
        "apply calibrated encoder offsets if required"
      ),
      "encoder_zero_calibration_verified": False,
    },
    "simulation_pd": {"kp": actuator.stiffness, "kd": actuator.damping},
    "real_robot_validation": (
      "Not performed; do not treat export equality as hardware validation"
    ),
    "validation": {
      "observations": len(inputs),
      "standing_observations": 512,
      "in_process_max_abs_error": float((actual - expected).abs().max()),
      "torch_only_process": json.loads(independent),
      "frozen_sources_verified": len(pins),
    },
  }
  manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
  print(
    json.dumps(
      {
        "model": str(model_path),
        "contract": str(manifest_path),
        "validation": manifest["validation"],
      }
    )
  )


if __name__ == "__main__":
  main()
