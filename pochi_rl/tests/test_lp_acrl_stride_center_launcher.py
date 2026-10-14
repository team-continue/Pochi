"""V11 pins its V10 model-only source, backend, assets and saved adaptive LR."""

import importlib.util
from copy import deepcopy
from pathlib import Path

import pytest

SCRIPT = (
  Path(__file__).resolve().parents[1]
  / "scripts/train_lp_acrl_blind_stride_center_refine.py"
)
SPEC = importlib.util.spec_from_file_location("stride_center_refine_launcher", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def test_exact_parent_checkpoint_pin_rejects_different_bytes(tmp_path, monkeypatch):
  assert launcher.SOURCE_CHECKPOINT_SHA256 == (
    "57d46865166d719541ebccdd4a56e7ba6096ca85021a597105a92c65dccad646"
  )
  path = tmp_path / "model_400.pt"
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
    "src/pochi_rl/mjlab/lp_acrl_blind_v10_env_cfg.py",
    "src/pochi_rl/mjlab/lp_acrl_quiet_stop.py",
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
  ["assets/pochi/meshes/foot.obj", "src/pochi_rl/mjlab/lp_acrl_quiet_stop.py"],
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
    del parent["source_sha256"]["src/pochi_rl/mjlab/lp_acrl_blind_v10_env_cfg.py"]
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


def _lr_checkpoint():
  rate = 7.593750000000004e-05
  return {
    "iter": 400,
    "infos": {
      "lp_acrl_wider_exploration_state": {
        "version": 1,
        "completed_updates": 401,
        "learning_rate": rate,
      }
    },
    "optimizer_state_dict": {"param_groups": [{"lr": rate}]},
  }


def test_learning_rate_comes_from_saved_adaptive_state_and_agrees_with_adam():
  assert launcher.source_learning_rate(_lr_checkpoint()) == 7.593750000000004e-05


@pytest.mark.parametrize(
  "fault",
  [
    "missing",
    "original_factory_lr",
    "nan",
    "bool",
    "empty_groups",
    "mismatched_group",
    "multiple_inconsistent_groups",
    "iteration",
    "count",
  ],
)
def test_cannot_silently_reset_or_diverge_from_pinned_actual_learning_rate(fault):
  saved = _lr_checkpoint()
  state = saved["infos"]["lp_acrl_wider_exploration_state"]
  groups = saved["optimizer_state_dict"]["param_groups"]
  if fault == "missing":
    state.pop("learning_rate")
  elif fault == "original_factory_lr":
    state["learning_rate"] = groups[0]["lr"] = 1e-4
  elif fault == "nan":
    state["learning_rate"] = float("nan")
  elif fault == "bool":
    state["learning_rate"] = True
  elif fault == "empty_groups":
    groups.clear()
  elif fault == "mismatched_group":
    groups[0]["lr"] = 1e-4
  elif fault == "multiple_inconsistent_groups":
    groups.append({"lr": 1e-4})
  elif fault == "iteration":
    saved["iter"] = 500
  else:
    state["completed_updates"] = 400
  with pytest.raises(ValueError, match="adaptive LR"):
    launcher.source_learning_rate(saved)


@pytest.mark.parametrize("path", ["/tmp/nan_dumps", "/dev/shm", "/data/nan_dumps"])
def test_guard_and_compile_storage_must_stay_in_task_ram(path):
  with pytest.raises(ValueError, match="below /dev/shm"):
    launcher.require_ram_path(Path(path), new=False)
