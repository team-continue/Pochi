"""V14 native bridge retains original rough task semantics and strict actor loading."""

import importlib.util
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from pochi_rl.mjlab.lp_acrl_duty_balance_refinement import (
  CORRUPTION_IDENTITY,
  STATE_KEY,
  expected_duty_balance_rough_task_space,
  expected_duty_balance_task_space,
  expected_exploration_identity,
  expected_symmetry_identity,
)

_PATH = (
  Path(__file__).resolve().parents[1] / "scripts/record_lp_acrl_blind_v14_native.py"
)
_SPEC = importlib.util.spec_from_file_location("v14_native_bridge", _PATH)
_BRIDGE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_BRIDGE)


def fixture():
  rough = expected_duty_balance_rough_task_space()
  backend = {"patch_id": "exact process-local identity"}
  saved = {
    "iter": 100,
    "actor_state_dict": {"distribution.std_param": torch.full((12,), .15)},
    "optimizer_state_dict": {"param_groups": [{"lr": 2.25e-5}]},
    "infos": {
      "lp_acrl_checkpoint_version": 1,
      "lp_acrl_contact_backend": deepcopy(backend),
      "lp_acrl_observation_corruption": deepcopy(CORRUPTION_IDENTITY),
      "lp_acrl_training_exploration": expected_exploration_identity(),
      "lp_acrl_training_symmetry": expected_symmetry_identity(),
      "env_state": {"common_step_counter": 2424},
      "lp_acrl": {
        "task_space": expected_duty_balance_task_space(),
        "training_terrain_seed": None,
      },
      STATE_KEY: {
        "version": 1, "completed_updates": 101, "learning_rate": 2.25e-5,
        "initialization": "model_only_fresh_optimizer_lp",
      },
    },
  }
  return saved, rough, backend


def test_complete_bridge_is_explicit_and_never_mutates_source():
  saved, rough, backend = fixture()
  old = deepcopy(saved)
  result = _BRIDGE.validate_native_bridge(saved, rough, backend)
  assert result["policy_task"] == "Pochi-LP-ACRL-Blind-v14"
  assert result["evaluation_environment_task"] == "Pochi-LP-ACRL-Blind-v12"
  assert result["policy_training_task_space"]["terrains"] == ("flat",)
  assert result["evaluation_task_space"]["terrains"][0] == "ascending_stairs"
  assert not result["checkpoint_identity_rewritten"]
  assert not result["optimizer_and_curriculum_loaded"]
  assert saved["infos"] == old["infos"]
  assert torch.equal(saved["actor_state_dict"]["distribution.std_param"],
                     old["actor_state_dict"]["distribution.std_param"])


@pytest.mark.parametrize(
  "change", ["backend", "version", "physics", "motor", "observations", "actions",
             "rewards", "termination", "catalog", "yaw", "level_type", "old_state",
             "std", "target_physics", "target_rewards", "missing_seed", "seed42",
             "missing_raw", "wrong_raw", "target_missing_raw", "target_wrong_raw"],
)
def test_non_scope_changes_rejected_before_actor_load(change):
  saved, rough, backend = fixture()
  flat = saved["infos"]["lp_acrl"]["task_space"]
  env = flat["environment"]
  if change == "missing_raw":
    env["rewards"].pop("moving_contact_duty_variance")
  elif change == "wrong_raw":
    env["rewards"]["moving_contact_duty_variance"]["weight"] = -4.
  elif change == "target_missing_raw":
    rough["environment"]["rewards"].pop("moving_contact_duty_variance")
  elif change == "target_wrong_raw":
    reward = rough["environment"]["rewards"]["moving_contact_duty_variance"]
    reward["params"]["tau_s"] = .25
  elif change == "backend":
    saved["infos"]["lp_acrl_contact_backend"] = {}
  elif change == "version":
    del saved["infos"]["lp_acrl_checkpoint_version"]
  elif change == "catalog":
    flat["terrains"] = ("ascending_stairs",)
  elif change == "yaw":
    flat["yaw_edges"] = (0., .2)
  elif change == "level_type":
    flat["num_levels"] = True
  elif change == "old_state":
    saved["infos"]["lp_acrl_flat_focus_state"] = {"version": 1}
  elif change == "std":
    saved["actor_state_dict"]["distribution.std_param"].fill_(.03)
  elif change == "missing_seed":
    saved["infos"]["lp_acrl"].pop("training_terrain_seed")
  elif change == "seed42":
    saved["infos"]["lp_acrl"]["training_terrain_seed"] = 42
  elif change == "target_physics":
    rough["environment"]["physics"]["gravity"] = (0., 0., -1.)
  elif change == "target_rewards":
    rough["environment"]["rewards"].pop("yaw_tracking_huber")
  else:
    key = {
      "motor": "robot_articulation", "termination": "terminations",
    }.get(change, change)
    env[key] = {"changed": True}
  with pytest.raises(ValueError):
    _BRIDGE.validate_native_bridge(saved, rough, backend)


def test_strict_base_actor_load_after_bridge_without_curriculum(monkeypatch, tmp_path):
  from rsl_rl.runners import OnPolicyRunner

  saved, rough, backend = fixture()
  calls = []
  sentinel = object()
  runner = SimpleNamespace(get_inference_policy=lambda **kw: sentinel)
  checkpoint = tmp_path / "model100.pt"
  checkpoint.write_bytes(b"unchanged source checkpoint")

  def actor_load(self, path, **kwargs):
    calls.append((self, path, kwargs))

  monkeypatch.setattr(OnPolicyRunner, "load", actor_load)
  policy, bridge = _BRIDGE.load_native_actor(
    runner, checkpoint, saved, rough, backend, "cpu",
  )
  assert policy is sentinel
  assert calls == [(runner, str(checkpoint), {
    "load_cfg": {"actor": True}, "strict": True, "map_location": "cpu",
  })]
  assert bridge["actor_and_normalizer_only"]
  assert checkpoint.read_bytes() == b"unchanged source checkpoint"
  saved["infos"]["lp_acrl"]["task_space"]["environment"]["actions"] = {}
  with pytest.raises(ValueError):
    _BRIDGE.load_native_actor(runner, checkpoint, saved, rough, backend, "cpu")
  assert len(calls) == 1


@pytest.mark.parametrize("suffix", [".mp4", ".json", ".provenance.json"])
def test_every_previous_take_artifact_prevents_overwrite(tmp_path, suffix):
  checkpoint = tmp_path / "model.pt"
  checkpoint.write_bytes(b"checkpoint")
  output = tmp_path / "first_take.mp4"
  previous = output.with_suffix(suffix)
  previous.write_bytes(b"previous evidence")
  with pytest.raises(SystemExit):
    _BRIDGE.parse_args(["--checkpoint", str(checkpoint), "--output", str(output)])
  assert previous.read_bytes() == b"previous evidence"


def test_native_seed_remains123_and_frozen_recorder_is_unmodified(tmp_path):
  checkpoint = tmp_path / "model.pt"
  checkpoint.write_bytes(b"checkpoint")
  args = ["--checkpoint", str(checkpoint), "--output", str(tmp_path / "take.mp4")]
  assert _BRIDGE.parse_args(args).seed == 123
  with pytest.raises(SystemExit):
    _BRIDGE.parse_args([*args, "--seed", "789"])
  recorder = _PATH.with_name("record_lp_acrl_blind_demo.py")
  assert _BRIDGE.digest(recorder) == _BRIDGE.RECORDER_SHA256
