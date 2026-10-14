"""V8 validates its complete reward contract before the frozen actor loader."""

import importlib.util
import json
import sys
from copy import deepcopy
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from pochi_rl.mjlab.lp_acrl_low_noise import (
  CORRUPTION_IDENTITY,
  expected_exploration_identity,
  expected_symmetry_identity,
)

_PATH = (
  Path(__file__).resolve().parents[1] / "scripts/eval_lp_acrl_behavior_blind_v8.py"
)
_SPEC = importlib.util.spec_from_file_location("blind_v8_eval_wrapper", _PATH)
_WRAPPER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_WRAPPER)


def setup(monkeypatch, tmp_path):
  calls = []
  identity = {"patch_id": "fixed", "epsilon": 1e-7}
  saved = {
    "infos": {
      "lp_acrl_contact_backend": identity.copy(),
      "lp_acrl_observation_corruption": deepcopy(CORRUPTION_IDENTITY),
      "lp_acrl_training_exploration": expected_exploration_identity(),
      "lp_acrl_training_symmetry": expected_symmetry_identity(),
    },
    "actor_state_dict": {"distribution.std_param": torch.full((12,), 0.03)},
  }
  rewards = {
    "standing_posture": {"weight": 1.5, "params": {"std": 0.15}},
    "moving_common_hip_pitch_deviation": {"weight": -4.0, "params": {}},
  }
  saved["infos"]["lp_acrl"] = {
    "task_space": {"environment": {"rewards": deepcopy(rewards)}}
  }
  checkpoint = tmp_path / "model.pt"
  checkpoint.write_bytes(b"immutable checkpoint")
  output = tmp_path / "result.json"
  backend = ModuleType("pochi_rl.mjlab.warp_contact_backend")
  backend.install_contact_backend = lambda: calls.append("install") or identity.copy()
  backend.active_contact_backend_identity = lambda: identity.copy()
  task = ModuleType("pochi_rl.mjlab.lp_acrl_blind_v8")
  task.register_blind_v8 = lambda: calls.append("register")
  cfg_module = ModuleType("pochi_rl.mjlab.lp_acrl_blind_v8_env_cfg")
  cfg_module.pochi_lp_acrl_blind_v8_env_cfg = lambda **kw: (
    calls.append("factory") or SimpleNamespace(rewards=deepcopy(rewards))
  )
  plane = ModuleType("eval_lp_acrl_behavior_plane")

  def evaluate():
    calls.append("evaluate")
    output.write_text(
      json.dumps(
        {
          "checkpoint_sha256": _WRAPPER.hashlib.sha256(
            checkpoint.read_bytes()
          ).hexdigest(),
          "source_sha256": {},
          "summary": {"stand": {"passed": 8}},
        }
      )
    )

  plane.main = evaluate
  for module in (backend, task, cfg_module, plane):
    monkeypatch.setitem(sys.modules, module.__name__, module)
  monkeypatch.setattr(
    torch, "load", lambda *args, **kwargs: calls.append("load") or saved
  )
  monkeypatch.setattr(
    sys, "argv", ["wrapper", "--checkpoint", str(checkpoint), "--output", str(output)]
  )
  return calls, saved, output


def test_valid_contract_checked_before_registration_and_reported(monkeypatch, tmp_path):
  calls, _, output = setup(monkeypatch, tmp_path)
  _WRAPPER.main()
  assert calls == ["install", "load", "factory", "register", "evaluate"]
  report = json.loads(output.read_text())
  assert report["summary"] == {"stand": {"passed": 8}}
  assert report["reward_identity_validation"][
    "full_declared_reward_configuration_checked"
  ]
  assert report["observation_corruption"] == CORRUPTION_IDENTITY
  assert report["training_exploration"] == json.loads(
    json.dumps(expected_exploration_identity())
  )
  assert report["training_symmetry"] == expected_symmetry_identity()
  assert report["low_noise_validation"]["saved_fixed_moving_std_checked"] is True
  assert sys.argv[-2:] == ["--task", "Pochi-LP-ACRL-Blind-v8"]


@pytest.mark.parametrize(
  "change",
  [
    "backend",
    "missing_noise",
    "actor_noise",
    "critic_noise",
    "missing_exploration",
    "learned_std",
    "wrong_columns",
    "symmetry",
    "std",
    "nan",
    "shape",
  ],
)
def test_incompatible_contract_never_reaches_environment(monkeypatch, tmp_path, change):
  calls, saved, output = setup(monkeypatch, tmp_path)
  info = saved["infos"]
  if change == "backend":
    del info["lp_acrl_contact_backend"]
  elif change == "missing_noise":
    del info["lp_acrl_observation_corruption"]
  elif change in ("actor_noise", "critic_noise"):
    info["lp_acrl_observation_corruption"]["groups"][change.split("_")[0]] = True
  elif change == "missing_exploration":
    del info["lp_acrl_training_exploration"]
  elif change == "learned_std":
    info["lp_acrl_training_exploration"]["moving_std_learned"] = True
  elif change == "wrong_columns":
    info["lp_acrl_training_exploration"]["command_columns"] = (0, 1, 2)
  elif change == "symmetry":
    info["lp_acrl_training_symmetry"]["use_data_augmentation"] = True
  elif change == "shape":
    saved["actor_state_dict"]["distribution.std_param"] = torch.full((1, 12), 0.03)
  else:
    saved["actor_state_dict"]["distribution.std_param"].fill_(
      float("nan") if change == "nan" else 0.12
    )
  with pytest.raises(ValueError):
    _WRAPPER.main()
  assert calls == ["install", "load"]
  assert not output.exists()


def test_existing_result_preserved_before_any_installation(monkeypatch, tmp_path):
  calls, _, output = setup(monkeypatch, tmp_path)
  output.write_text("previous evidence")
  with pytest.raises(SystemExit):
    _WRAPPER.main()
  assert not calls
  assert output.read_text() == "previous evidence"


@pytest.mark.parametrize(
  "change", ["v5_parent", "v6_parent", "wrong_weight", "other_reward", "extra_reward"]
)
def test_entire_reward_identity_checked_before_registration(
  monkeypatch, tmp_path, change
):
  calls, saved, output = setup(monkeypatch, tmp_path)
  rewards = saved["infos"]["lp_acrl"]["task_space"]["environment"]["rewards"]
  if change == "v5_parent":
    del rewards["moving_common_hip_pitch_deviation"]
  elif change == "v6_parent":
    rewards["moving_default_joint_deviation"] = rewards.pop(
      "moving_common_hip_pitch_deviation"
    )
  elif change == "wrong_weight":
    rewards["moving_common_hip_pitch_deviation"]["weight"] = -1
  elif change == "other_reward":
    rewards["standing_posture"]["weight"] = 0.5
  else:
    rewards["undeclared"] = {"weight": 1}
  with pytest.raises(ValueError, match="rewards must exactly match"):
    _WRAPPER.main()
  assert calls == ["install", "load", "factory"]
  assert not output.exists()
