"""Pinned, opt-in contact backend source/binding lifecycle regressions."""

import hashlib
import importlib.util
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from pochi_rl.mjlab import warp_contact_backend as backend


@pytest.fixture
def isolated_backend(monkeypatch, tmp_path):
  installed = importlib.util.find_spec("mujoco_warp._src.collision_gjk")
  assert installed is not None and installed.origin is not None
  content = Path(installed.origin).read_bytes()
  source = tmp_path / "installed" / "mujoco_warp" / "_src" / "collision_gjk.py"
  source.parent.mkdir(parents=True)
  source.write_bytes(content)
  original = SimpleNamespace(__file__=str(source), ccd=object())
  convex = SimpleNamespace(ccd=original.ccd)
  utility = SimpleNamespace(_KERNEL_CACHE={})
  monkeypatch.setattr(backend, "_STATE", None)
  monkeypatch.delitem(sys.modules, backend._MODULE_NAME, raising=False)
  monkeypatch.setattr(backend, "_dependencies", lambda: (original, convex, utility))

  def load(path, verified_bytes):
    assert path.read_bytes() == verified_bytes
    module = ModuleType(backend._MODULE_NAME)
    module.ccd = object()
    monkeypatch.setitem(sys.modules, backend._MODULE_NAME, module)
    return module

  monkeypatch.setattr(backend, "_load_generated_module", load)
  return SimpleNamespace(
    source=source,
    content=content,
    original=original,
    convex=convex,
    utility=utility,
    cache=tmp_path / "cache",
  )


def test_installed_source_and_exact_verified_single_line_patch(isolated_backend):
  source = isolated_backend.content
  generated = backend._patch_source(source)
  assert hashlib.sha256(source).hexdigest() == backend.SOURCE_SHA256
  assert hashlib.sha256(generated).hexdigest() == backend.PATCHED_SHA256
  assert generated == source.replace(backend._OLD, backend._NEW, 1)
  assert generated.splitlines()[:16] == source.splitlines()[:16]
  assert b"Licensed under the Apache License" in generated


@pytest.mark.parametrize("content", [b"", b"unknown source", b"epsilon = 1e-7"])
def test_unpinned_source_rejected(content):
  with pytest.raises(RuntimeError, match="pinned SHA256"):
    backend._patch_source(content)


def test_duplicate_literal_rejected_even_with_matching_input_hash(monkeypatch):
  content = backend._OLD + b"\n" + backend._OLD
  monkeypatch.setattr(backend, "SOURCE_SHA256", hashlib.sha256(content).hexdigest())
  with pytest.raises(RuntimeError, match="exactly one"):
    backend._patch_source(content)


def test_generated_hash_is_independently_checked(isolated_backend, monkeypatch):
  monkeypatch.setattr(backend, "PATCHED_SHA256", "0" * 64)
  with pytest.raises(RuntimeError, match="Generated contact backend"):
    backend._patch_source(isolated_backend.content)


def test_active_and_archive_require_explicit_install(isolated_backend):
  with pytest.raises(RuntimeError, match="not installed"):
    backend.active_contact_backend_identity()
  with pytest.raises(RuntimeError, match="not installed"):
    backend.contact_backend_source_paths()


def test_idempotence_portable_identity_and_archive_paths(isolated_backend, tmp_path):
  state = isolated_backend
  old = state.original.ccd
  identity = backend.install_contact_backend(state.cache)
  bound = state.convex.ccd
  assert bound is not old and state.original.ccd is old
  assert state.source.read_bytes() == state.content
  assert backend.install_contact_backend(tmp_path / "unused") == identity
  assert state.convex.ccd is bound and not (tmp_path / "unused").exists()
  identity["versions"]["mujoco"] = "caller mutation"
  actual = backend.active_contact_backend_identity()
  assert actual["versions"] == backend.PINNED_VERSIONS
  assert str(tmp_path) not in json.dumps(actual)
  paths = backend.contact_backend_source_paths()
  assert set(paths) == {
    "installed_collision_gjk",
    "generated_collision_gjk",
    "installer",
  }
  assert paths["installed_collision_gjk"] == state.source
  assert paths["generated_collision_gjk"].name.endswith(backend.PATCHED_SHA256 + ".py")
  assert paths["generated_collision_gjk"].read_bytes() == backend._patch_source(
    state.content
  )


def test_competing_binding_is_not_overwritten(isolated_backend):
  competing = object()
  isolated_backend.convex.ccd = competing
  with pytest.raises(RuntimeError, match="Competing"):
    backend.install_contact_backend(isolated_backend.cache)
  assert isolated_backend.convex.ccd is competing
  assert not isolated_backend.cache.exists()


@pytest.mark.parametrize("builder", ["ccd_kernel_builder", "ccd_hfield_kernel_builder"])
def test_preexisting_ccd_kernels_refuse_late_install(isolated_backend, builder):
  isolated_backend.utility._KERNEL_CACHE[(6, 7, hash(builder))] = object()
  with pytest.raises(RuntimeError, match="before any CCD kernel"):
    backend.install_contact_backend(isolated_backend.cache)
  assert isolated_backend.convex.ccd is isolated_backend.original.ccd


def test_unrelated_kernel_cache_does_not_block(isolated_backend):
  isolated_backend.utility._KERNEL_CACHE[(hash("other_builder"),)] = object()
  backend.install_contact_backend(isolated_backend.cache)


@pytest.mark.parametrize(
  "tamper", ["live_binding", "original_binding", "module_binding", "module_slot"]
)
def test_binding_replacement_detected(isolated_backend, tamper, monkeypatch):
  backend.install_contact_backend(isolated_backend.cache)
  if tamper == "live_binding":
    isolated_backend.convex.ccd = object()
  elif tamper == "original_binding":
    isolated_backend.original.ccd = object()
  elif tamper == "module_binding":
    backend._STATE.module.ccd = object()
  else:
    monkeypatch.setitem(sys.modules, backend._MODULE_NAME, ModuleType("competing"))
  with pytest.raises(RuntimeError, match="binding has been replaced"):
    backend.active_contact_backend_identity()
  with pytest.raises(RuntimeError, match="binding has been replaced"):
    backend.install_contact_backend(isolated_backend.cache)


def test_tampered_existing_cache_rejected_without_overwrite(isolated_backend):
  state = isolated_backend
  path = state.cache / (backend._MODULE_NAME + ".py")
  path.parent.mkdir()
  path.write_bytes(b"must not execute or overwrite")
  with pytest.raises(RuntimeError, match="cache content mismatch"):
    backend.install_contact_backend(state.cache)
  assert path.read_bytes() == b"must not execute or overwrite"
  assert state.convex.ccd is state.original.ccd


def test_valid_existing_cache_reused_and_symlinks_rejected(isolated_backend, tmp_path):
  state = isolated_backend
  path = state.cache / (backend._MODULE_NAME + ".py")
  path.parent.mkdir()
  content = backend._patch_source(state.content)
  other = tmp_path / "other.py"
  other.write_bytes(content)
  path.symlink_to(other)
  with pytest.raises(RuntimeError, match="regular file"):
    backend.install_contact_backend(state.cache)
  path.unlink()
  path.write_bytes(content)
  before = path.stat().st_ino
  backend.install_contact_backend(state.cache)
  assert path.stat().st_ino == before


def test_post_install_cache_tamper_is_detected(isolated_backend):
  backend.install_contact_backend(isolated_backend.cache)
  path = backend.contact_backend_source_paths()["generated_collision_gjk"]
  path.chmod(0o644)
  path.write_bytes(b"changed after installation")
  with pytest.raises(RuntimeError, match="cache content mismatch"):
    backend.active_contact_backend_identity()


def test_package_version_mismatch_prevents_mutation(isolated_backend, monkeypatch):
  monkeypatch.setattr(backend.importlib.metadata, "version", lambda package: "wrong")
  with pytest.raises(RuntimeError, match="pinned package versions"):
    backend.install_contact_backend(isolated_backend.cache)
  assert isolated_backend.convex.ccd is isolated_backend.original.ccd


def test_installed_package_cannot_be_cache_target(isolated_backend):
  with pytest.raises(RuntimeError, match="inside the installed package"):
    backend.install_contact_backend(isolated_backend.source.parent)


def test_verified_loader_ignores_stale_pyc(tmp_path, monkeypatch):
  monkeypatch.delitem(sys.modules, backend._MODULE_NAME, raising=False)
  path = tmp_path / "standalone.py"
  path.write_bytes(b"value = 99\n")
  py_compile.compile(str(path), doraise=True)
  original_stat = path.stat()
  expected = b"value = 42\n"
  path.write_bytes(expected)
  os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
  try:
    module = backend._load_generated_module(path, expected)
    assert module.value == 42
  finally:
    sys.modules.pop(backend._MODULE_NAME, None)


def test_verified_loader_cleans_failed_import(tmp_path, monkeypatch):
  monkeypatch.delitem(sys.modules, backend._MODULE_NAME, raising=False)
  path = tmp_path / "raises.py"
  content = b"raise RuntimeError('synthetic failure')\n"
  path.write_bytes(content)
  with pytest.raises(RuntimeError, match="synthetic failure"):
    backend._load_generated_module(path, content)
  assert backend._MODULE_NAME not in sys.modules


def test_real_cpu_installation_in_clean_process(tmp_path):
  code = """
import json, sys
from pathlib import Path
from mujoco_warp._src import collision_convex, collision_gjk
from pochi_rl.mjlab.warp_contact_backend import (
  install_contact_backend, active_contact_backend_identity,
  contact_backend_source_paths,
)
assert collision_convex.ccd is collision_gjk.ccd
identity = install_contact_backend(Path(sys.argv[1]))
assert collision_convex.ccd is not collision_gjk.ccd
assert install_contact_backend() == identity == active_contact_backend_identity()
assert all(path.is_file() for path in contact_backend_source_paths().values())
print(json.dumps(identity))
"""
  completed = subprocess.run(
    [sys.executable, "-c", code, str(tmp_path / "real")],
    check=True,
    capture_output=True,
    text=True,
    timeout=60,
  )
  identity = json.loads(completed.stdout.strip().splitlines()[-1])
  assert identity["patch_id"] == backend.PATCH_ID
  assert identity["patched_sha256"] == backend.PATCHED_SHA256


def test_competing_binding_during_module_load_is_not_overwritten(
  isolated_backend,
  monkeypatch,
):
  state = isolated_backend
  original_loader = backend._load_generated_module
  competing = object()

  def race(path, content):
    module = original_loader(path, content)
    state.convex.ccd = competing
    return module

  monkeypatch.setattr(backend, "_load_generated_module", race)
  with pytest.raises(RuntimeError, match="appeared during installation"):
    backend.install_contact_backend(state.cache)
  assert state.convex.ccd is competing
  assert backend._STATE is None
  assert backend._MODULE_NAME not in sys.modules
