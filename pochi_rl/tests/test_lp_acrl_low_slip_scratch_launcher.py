"""Scratch rejects learned state and archives the exact robot without GPU use."""

import ast
import importlib.util
import subprocess
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from rsl_rl.modules import EmpiricalNormalization

from pochi_rl.lp_acrl.curriculum import LPACRLCurriculum

SCRIPT = (
  Path(__file__).resolve().parents[1]
  / "scripts/train_lp_acrl_blind_low_slip_scratch.py"
)
SPEC = importlib.util.spec_from_file_location("low_slip_scratch_launcher", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)


@pytest.mark.parametrize(
  "options",
  [
    ["--init-checkpoint", "prior.pt"],
    ["--checkpoint", "prior.pt"],
    ["--load", "prior.pt"],
    ["--resume"],
  ],
)
def test_checkpoint_initialization_and_resume_arguments_are_rejected_before_imports(
  tmp_path, options
):
  target = tmp_path / "must_not_create"
  result = subprocess.run(
    [sys.executable, str(SCRIPT), "--output", str(target), *options],
    capture_output=True,
    text=True,
    timeout=15,
  )
  assert result.returncode == 2 and "unrecognized arguments" in result.stderr
  assert not target.exists()


def test_scratch_launcher_contains_no_model_or_checkpoint_load_call():
  tree = ast.parse(SCRIPT.read_text())
  forbidden = [
    node
    for node in ast.walk(tree)
    if isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and node.func.attr in ("load", "load_state_dict")
  ]
  assert forbidden == []


class _Model(torch.nn.Module):
  def __init__(self, output):
    super().__init__()
    self.obs_normalizer = EmpiricalNormalization(48)
    self.linear = torch.nn.Linear(48, output)


def _runner():
  actor, critic = _Model(12), _Model(1)
  schedule = {
    "completed_updates": 0,
    "next_moving_std": 0.35,
    "training_resume_allowed": True,
  }
  alg = SimpleNamespace(
    _raw_actor=actor,
    _raw_critic=critic,
    learning_rate=1e-3,
    optimizer=torch.optim.Adam([*actor.parameters(), *critic.parameters()], lr=1e-3),
    schedule_state=lambda: schedule,
  )
  return SimpleNamespace(
    alg=alg,
    current_learning_iteration=0,
    command_term=SimpleNamespace(curriculum=LPACRLCurriculum(600)),
  )


def test_fresh_models_normalizers_optimizer_and_lp_recorded_without_mutation():
  runner = _runner()
  before = {k: v.clone() for k, v in runner.alg._raw_actor.state_dict().items()}
  report = launcher.assert_fresh_scratch(runner)
  assert report["checkpoint_loaded"] is False
  assert report["initial_actor_critic_normalizer_count"] == 0
  assert len(report["initial_model_state_sha256"]["actor"]) == 64
  assert (
    report["initial_model_state_sha256"]["actor"]
    != report["initial_model_state_sha256"]["critic"]
  )
  assert all(
    torch.equal(v, before[k]) for k, v in runner.alg._raw_actor.state_dict().items()
  )


@pytest.mark.parametrize(
  "fault",
  [
    "optimizer",
    "iteration",
    "lp",
    "count",
    "mean",
    "variance",
    "std",
    "schedule",
    "inference",
    "lr",
    "model_nan",
  ],
)
def test_any_used_learning_or_normalization_state_is_rejected(fault):
  runner = _runner()
  model = runner.alg._raw_actor
  if fault == "optimizer":
    runner.alg.optimizer.state[next(model.parameters())] = {}
  elif fault == "iteration":
    runner.current_learning_iteration = 1
  elif fault == "lp":
    runner.command_term.curriculum.previous_valid[0] = True
  elif fault == "count":
    model.obs_normalizer.count.fill_(1)
  elif fault == "mean":
    model.obs_normalizer._mean.fill_(0.1)
  elif fault == "variance":
    model.obs_normalizer._var.fill_(0.5)
  elif fault == "std":
    model.obs_normalizer._std.fill_(0.5)
  elif fault == "schedule":
    runner.alg.schedule_state()["completed_updates"] = 1
  elif fault == "inference":
    runner.alg.schedule_state()["training_resume_allowed"] = False
  elif fault == "lr":
    runner.alg.learning_rate = 1e-4
  else:
    with torch.no_grad():
      model.linear.weight[0, 0] = float("nan")
  with pytest.raises(ValueError):
    launcher.assert_fresh_scratch(runner)


def _assets(tmp_path):
  project = tmp_path / "project"
  folder = project / "assets/pochi/meshes"
  folder.mkdir(parents=True)
  (folder.parent / "pochi.xml").write_bytes(b"<mujoco/>\n")
  (folder / "foot.obj").write_bytes(b"v 0 0 0\n" * 100)
  return project


def test_lossless_robot_archive_is_verified_and_pins_every_source_file(tmp_path):
  project = _assets(tmp_path)
  archive, hashes = launcher.archive_robot_assets(project, tmp_path)
  try:
    with tarfile.open(archive, "r:gz") as bundle:
      assert set(bundle.getnames()) == set(hashes)
      for name in hashes:
        assert bundle.extractfile(name).read() == (project / name).read_bytes()
        assert hashes[name] == launcher.sha256(project / name)
  finally:
    archive.unlink()


def test_unchanged_assets_produce_identical_archive_bytes_for_safe_deduplication(
  tmp_path,
):
  project = _assets(tmp_path)
  first, hashes = launcher.archive_robot_assets(project, tmp_path)
  second, repeated = launcher.archive_robot_assets(project, tmp_path)
  try:
    assert hashes == repeated
    assert first.read_bytes() == second.read_bytes()
  finally:
    first.unlink()
    second.unlink()


@pytest.mark.parametrize("fault", ["no_xml", "no_mesh", "escape"])
def test_incomplete_or_escaping_robot_assets_rejected(tmp_path, fault):
  project = _assets(tmp_path)
  xml = project / "assets/pochi/pochi.xml"
  mesh = project / "assets/pochi/meshes/foot.obj"
  if fault == "no_xml":
    xml.unlink()
  elif fault == "no_mesh":
    mesh.unlink()
  else:
    outside = tmp_path / "outside.obj"
    outside.write_bytes(mesh.read_bytes())
    mesh.unlink()
    mesh.symlink_to(outside)
  with pytest.raises(ValueError):
    launcher.archive_robot_assets(project, tmp_path)


def test_ram_guard_rejects_disk_paths_and_existing_directory(tmp_path, monkeypatch):
  ram = tmp_path / "ram"
  ram.mkdir()
  monkeypatch.setattr(launcher, "RAM_ROOT", ram)
  with pytest.raises(ValueError):
    launcher.require_ram_path(tmp_path / "disk", new=True)
  (ram / "existing").mkdir()
  with pytest.raises(FileExistsError):
    launcher.require_ram_path(ram / "existing", new=True)
