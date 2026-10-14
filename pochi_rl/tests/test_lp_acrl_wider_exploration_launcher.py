"""V10 requires the pinned V8 checkpoint and unchanged source/backend provenance."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPT = (
  Path(__file__).resolve().parents[1]
  / "scripts/train_lp_acrl_blind_wider_exploration_resume.py"
)
SPEC = importlib.util.spec_from_file_location("wider_exploration_launcher", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_exact_parent_checkpoint_pin_rejects_different_bytes(tmp_path, monkeypatch):
  assert launcher.SOURCE_CHECKPOINT_SHA256 == (
    "347e67f25be0a9f03f65d4fae54da8a311cf799520ebcdfd7a392d4d5e02bfa2"
  )
  path = tmp_path / "model_300.pt"
  path.write_bytes(b"parent checkpoint")
  digest = launcher.sha256(path)
  monkeypatch.setattr(launcher, "SOURCE_CHECKPOINT_SHA256", digest)
  assert launcher.validate_checkpoint_pin(path) == digest
  path.write_bytes(b"different parent checkpoint")
  with pytest.raises(ValueError, match="exact preregistered"):
    launcher.validate_checkpoint_pin(path)


def _parent(project):
  names = (
    "assets/pochi/pochi.xml",
    "assets/pochi/meshes/foot.obj",
    "src/pochi_rl/mjlab/lp_acrl_blind_v8_env_cfg.py",
    "src/pochi_rl/mjlab/lp_acrl_quiet_stop.py",
    "src/pochi_rl/mjlab/lp_acrl_common_hip_rewards.py",
    "scripts/parent_launcher.py",
  )
  for name in names:
    path = project / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"source {name}\n")
  return {
    "task": launcher.SOURCE_TASK,
    "source_sha256": {name: launcher.sha256(project / name) for name in names},
  }


@pytest.mark.parametrize(
  "changed",
  [
    "assets/pochi/meshes/foot.obj",
    "src/pochi_rl/mjlab/lp_acrl_quiet_stop.py",
    "scripts/parent_launcher.py",
  ],
)
def test_all_parent_runtime_and_asset_hashes_are_required(tmp_path, changed):
  parent = _parent(tmp_path)
  assert launcher.validate_parent_sources(parent, tmp_path) == parent["source_sha256"]
  (tmp_path / changed).write_text("changed\n")
  with pytest.raises(ValueError, match="differs"):
    launcher.validate_parent_sources(parent, tmp_path)


@pytest.mark.parametrize("fault", ["family", "mesh", "runtime", "escape"])
def test_unverifiable_parent_is_rejected(tmp_path, fault):
  project = tmp_path / "project"
  parent = _parent(project)
  if fault == "family":
    parent["task"] = "Pochi-LP-ACRL-Blind-v3"
  elif fault == "mesh":
    del parent["source_sha256"]["assets/pochi/meshes/foot.obj"]
  elif fault == "runtime":
    del parent["source_sha256"]["src/pochi_rl/mjlab/lp_acrl_blind_v8_env_cfg.py"]
  else:
    outside = tmp_path / "outside.py"
    outside.write_text("not a project source\n")
    name = "src/pochi_rl/escape.py"
    (project / name).symlink_to(outside)
    parent["source_sha256"][name] = launcher.sha256(outside)
  with pytest.raises(ValueError):
    launcher.validate_parent_sources(parent, project)


def _backend_fixture():
  backend = {
    "patch_id": launcher.BACKEND_PATCH,
    "discrete_epa_epsilon": 1e-7,
    "versions": {"mujoco-warp": "3.8.0.2"},
    "patched_sha256": "a" * 64,
  }
  parent = {"task": launcher.SOURCE_TASK, "contact_backend": deepcopy(backend)}
  saved = {"infos": {"lp_acrl_contact_backend": deepcopy(backend)}}
  return parent, saved, backend


def test_corrected_backend_is_identical_without_a_physics_transition():
  parent, saved, backend = _backend_fixture()
  assert launcher.backend_transition(parent, saved, backend) == {
    "source": backend,
    "target": backend,
    "unchanged": True,
  }


@pytest.mark.parametrize(
  "fault", ["legacy", "parent_missing", "checkpoint_missing", "version", "epsilon"]
)
def test_missing_or_changed_backend_is_rejected(fault):
  parent, saved, backend = _backend_fixture()
  if fault == "legacy":
    parent["task"] = "Pochi-LP-ACRL-Blind-v3"
  elif fault == "parent_missing":
    parent.pop("contact_backend")
  elif fault == "checkpoint_missing":
    saved["infos"].clear()
  elif fault == "version":
    saved["infos"]["lp_acrl_contact_backend"]["versions"] = {}
  else:
    backend["discrete_epa_epsilon"] = 1e-15
  with pytest.raises(ValueError):
    launcher.backend_transition(parent, saved, backend)


@pytest.mark.parametrize(
  ("updates", "expected"),
  [(1, (301, 302, 301)), (3, (301, 304, 303)), (1000, (301, 1301, 1300))],
)
def test_post_update_cursor_and_final_checkpoint_numbering(updates, expected):
  assert launcher.continuation_window(300, updates) == expected


@pytest.mark.parametrize(
  ("iteration", "updates"),
  [(299, 1000), (301, 1000), (True, 1000), (300, 0), (300, 1001), (300, True)],
)
def test_unregistered_continuation_window_is_rejected(iteration, updates):
  with pytest.raises(ValueError):
    launcher.continuation_window(iteration, updates)


def test_full_batch_and_fresh_reset_seed_are_preserved_with_explicit_smoke_exception():
  parent = {"seed": 42, "num_envs": 4096}
  launcher.validate_reset_request(parent, 42, 4096, False)
  launcher.validate_reset_request(parent, 42, 32, True)
  with pytest.raises(ValueError, match="seed"):
    launcher.validate_reset_request(parent, 43, 4096, False)
  with pytest.raises(ValueError, match="environment count"):
    launcher.validate_reset_request(parent, 42, 32, False)
