"""The new launcher records a narrow backend change and protects provenance."""

import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
  Path(__file__).resolve().parents[1]
  / "scripts/train_lp_acrl_blind_qualified_refine.py"
)
SPEC = importlib.util.spec_from_file_location("qualified_refine_launcher", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


def _parent(project):
  names = (
    "assets/pochi/pochi.xml",
    "assets/pochi/meshes/foot.obj",
    "src/pochi_rl/mjlab/lp_acrl_blind_v3_env_cfg.py",
    "src/pochi_rl/mjlab/lp_acrl_quiet_stop.py",
  )
  for name in names:
    path = project / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"original source {name}\n")
  return {
    "task": launcher.SOURCE_TASK,
    "source_sha256": {name: launcher.sha256(project / name) for name in names},
  }


def _backend():
  return {
    "patch_id": launcher.BACKEND_PATCH,
    "source_sha256": "a" * 64,
    "patched_sha256": "b" * 64,
    "versions": {"mujoco-warp": "3.8.0.2", "warp-lang": "1.13.0"},
    "discrete_epa_epsilon": 1e-7,
    "scope": "mujoco_warp._src.collision_convex.ccd (process-local explicit binding)",
  }


def test_all_recorded_runtime_sources_and_assets_are_verified(tmp_path):
  parent = _parent(tmp_path)
  assert launcher.validate_parent_sources(parent, tmp_path) == parent["source_sha256"]
  (tmp_path / "src/pochi_rl/mjlab/lp_acrl_quiet_stop.py").write_text("changed\n")
  with pytest.raises(ValueError, match="differs"):
    launcher.validate_parent_sources(parent, tmp_path)


def test_robot_mesh_change_is_rejected_even_with_unchanged_config(tmp_path):
  parent = _parent(tmp_path)
  (tmp_path / "assets/pochi/meshes/foot.obj").write_text("different mesh\n")
  with pytest.raises(ValueError, match="physical asset differs"):
    launcher.validate_parent_sources(parent, tmp_path)


@pytest.mark.parametrize("missing", ["xml", "mesh", "runtime"])
def test_incomplete_parent_provenance_is_rejected(tmp_path, missing):
  parent = _parent(tmp_path)
  remove = {
    "xml": "assets/pochi/pochi.xml",
    "mesh": "assets/pochi/meshes/foot.obj",
    "runtime": "src/pochi_rl/mjlab/lp_acrl_quiet_stop.py",
  }[missing]
  del parent["source_sha256"][remove]
  with pytest.raises(ValueError, match="lacks required"):
    launcher.validate_parent_sources(parent, tmp_path)


@pytest.mark.parametrize(
  "task", ["Pochi-LP-ACRL-Blind-v2", "Pochi-LP-ACRL-Blind-v4", None]
)
def test_only_v3_parent_is_authorized(tmp_path, task):
  parent = _parent(tmp_path)
  parent["task"] = task
  with pytest.raises(ValueError, match="Blind-v3"):
    launcher.validate_parent_sources(parent, tmp_path)
  with pytest.raises(ValueError, match="Blind-v3"):
    launcher.backend_transition(parent, {}, _backend())


def test_parent_paths_cannot_escape_project_even_with_matching_hash(tmp_path):
  project = tmp_path / "project"
  parent = _parent(project)
  outside = tmp_path / "outside.py"
  outside.write_text("outside project\n")
  alias = project / "src/pochi_rl/mjlab/escape.py"
  alias.symlink_to(outside)
  parent["source_sha256"]["src/pochi_rl/mjlab/escape.py"] = launcher.sha256(outside)
  with pytest.raises(ValueError, match="differs"):
    launcher.validate_parent_sources(parent, project)


def test_transition_preserves_legacy_metadata_limitation_and_exact_target():
  backend = _backend()
  result = launcher.backend_transition({"task": launcher.SOURCE_TASK}, {}, backend)
  assert result["target"] == backend
  assert result["source"]["discrete_epa_epsilon"] == 1e-15
  assert result["source"]["checkpoint_backend_metadata_available"] is False
  assert result["environment_identity_relaxation"] == ["environment.rewards"]
  assert result["backend_change_recorded_separately"] is True


@pytest.mark.parametrize(
  "field,value",
  [
    ("patch_id", "another-patch"),
    ("discrete_epa_epsilon", 1e-6),
    ("versions", {}),
    ("scope", ""),
    ("source_sha256", "no-hash"),
    ("patched_sha256", "G" * 64),
  ],
)
def test_unapproved_or_unidentified_backend_is_rejected(field, value):
  backend = _backend()
  backend[field] = value
  with pytest.raises(ValueError, match="backend"):
    launcher.backend_transition({"task": launcher.SOURCE_TASK}, {}, backend)


def test_existing_backend_checkpoint_cannot_be_mislabeled_as_legacy_transition():
  parent = {"task": launcher.SOURCE_TASK}
  with pytest.raises(ValueError, match="legacy v3 checkpoint"):
    launcher.backend_transition(
      parent, {"infos": {"lp_acrl_contact_backend": _backend()}}, _backend()
    )
  parent["contact_backend"] = _backend()
  with pytest.raises(ValueError, match="unpatched legacy"):
    launcher.backend_transition(parent, {}, _backend())


def _backend_paths(folder):
  paths = {}
  for key in ("installed_collision_gjk", "generated_collision_gjk", "installer"):
    path = folder / f"{key}.py"
    path.write_text(f"{key}\n")
    paths[key] = path
  backend = _backend()
  backend["source_sha256"] = launcher.sha256(paths["installed_collision_gjk"])
  backend["patched_sha256"] = launcher.sha256(paths["generated_collision_gjk"])
  return paths, backend


def test_archive_preserves_original_generated_and_installer_and_refuses_overwrite(
  tmp_path,
):
  paths, backend = _backend_paths(tmp_path)
  output = tmp_path / "run"
  output.mkdir()
  records = launcher.archive_backend_sources(output, paths, backend)
  assert records.keys() == paths.keys()
  for key, record in records.items():
    assert (output / record["archived_path"]).read_bytes() == paths[key].read_bytes()
    assert record["sha256"] == launcher.sha256(paths[key])
  with pytest.raises(FileExistsError):
    launcher.archive_backend_sources(output, paths, backend)


@pytest.mark.parametrize(
  "corruption", ["missing_original", "changed_original", "changed_generated"]
)
def test_archive_rejects_missing_or_changed_backend_source_before_copy(
  tmp_path, corruption
):
  paths, backend = _backend_paths(tmp_path)
  if corruption == "missing_original":
    paths.pop("installed_collision_gjk")
  else:
    key = (
      "installed_collision_gjk"
      if corruption == "changed_original"
      else "generated_collision_gjk"
    )
    paths[key].write_text("changed after validation\n")
  output = tmp_path / "run"
  output.mkdir()
  with pytest.raises(ValueError, match="source"):
    launcher.archive_backend_sources(output, paths, backend)
  assert not (output / "contact_backend_sources").exists()


@pytest.mark.parametrize(
  "path", ["/tmp/pochi_dump", "/dev/shm", "/dev/shm/../disk_dump"]
)
def test_compiled_models_and_temporary_compilation_cannot_use_disk_paths(path):
  with pytest.raises(ValueError, match="below /dev/shm"):
    launcher.require_ram_path(Path(path), new=False)


def test_guard_directory_must_be_new_and_cannot_follow_disk_symlink(
  tmp_path, monkeypatch
):
  # Exercise the path logic with private fixture directories, never shared RAM.
  ram = tmp_path / "ram"
  ram.mkdir()
  monkeypatch.setattr(launcher, "RAM_ROOT", ram)
  assert launcher.require_ram_path(ram / "new_guard", new=True) == ram / "new_guard"
  existing = ram / "existing_guard"
  existing.mkdir()
  with pytest.raises(FileExistsError):
    launcher.require_ram_path(existing, new=True)
  outside = tmp_path / "disk"
  outside.mkdir()
  (ram / "symlink").symlink_to(outside, target_is_directory=True)
  with pytest.raises(ValueError, match="below /dev/shm"):
    launcher.require_ram_path(ram / "symlink", new=False)
