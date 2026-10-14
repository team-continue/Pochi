"""Explicit, pinned float32 EPA contact backend for new Pochi experiments.

The installed mujoco-warp package is never edited. An isolated copy changes only
its discrete EPA stopping epsilon, following the verified numerical regression.
Install before constructing an environment or materializing any CCD kernels.
Existing tasks retain their original backend unless their entrypoint opts in.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import importlib.util
import os
import sys
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

PATCH_ID = "pochi-warp-discrete-epa-epsilon-1e-7-v1"
PINNED_VERSIONS = {
  "mujoco-warp": "3.8.0.2",
  "mujoco": "3.10.1.dev948945088",
  "warp-lang": "1.13.0",
}
SOURCE_SHA256 = "b91673337448db07a7e4ccda665d652a4eb84c2b9a770ed3b59d72c8818360b7"
PATCHED_SHA256 = "ae90d1a6a20cc478cead90b539f1b06e086f24484cd71f03ef3443857c6f2e06"
_OLD = b"epsilon = wp.where(is_discrete, 1e-15, tolerance)"
_NEW = b"epsilon = wp.where(is_discrete, 1e-7, tolerance)"
_MODULE_NAME = "pochi_warp_contact_epa_" + PATCHED_SHA256
_LOCK = threading.RLock()


@dataclass(frozen=True)
class _Installation:
  source: Path
  generated: Path
  module: ModuleType
  original_ccd: Any
  installed_ccd: Any


_STATE: _Installation | None = None


def _sha256(content: bytes) -> str:
  return hashlib.sha256(content).hexdigest()


def _package_versions() -> dict[str, str]:
  try:
    versions = {
      package: importlib.metadata.version(package) for package in PINNED_VERSIONS
    }
  except importlib.metadata.PackageNotFoundError as exc:
    raise RuntimeError("Pinned contact backend dependency is missing") from exc
  if versions != PINNED_VERSIONS:
    raise RuntimeError(
      f"Contact backend requires pinned package versions {PINNED_VERSIONS}; "
      f"found {versions}"
    )
  return versions


def _dependencies() -> tuple[ModuleType, ModuleType, ModuleType]:
  return (
    importlib.import_module("mujoco_warp._src.collision_gjk"),
    importlib.import_module("mujoco_warp._src.collision_convex"),
    importlib.import_module("mujoco_warp._src.warp_util"),
  )


def _patch_source(content: bytes) -> bytes:
  if _sha256(content) != SOURCE_SHA256:
    raise RuntimeError("Installed collision_gjk.py does not match pinned SHA256")
  if content.count(_OLD) != 1:
    raise RuntimeError("Contact backend requires exactly one EPA epsilon literal")
  patched = content.replace(_OLD, _NEW, 1)
  if _sha256(patched) != PATCHED_SHA256:
    raise RuntimeError("Generated contact backend does not match verified SHA256")
  return patched


def _verify_file(path: Path, expected: bytes) -> None:
  if path.is_symlink() or not path.is_file():
    raise RuntimeError(f"Contact backend cache must be a regular file: {path}")
  if path.read_bytes() != expected:
    raise RuntimeError(f"Contact backend cache content mismatch: {path}")


def _write_verified_cache(path: Path, content: bytes) -> None:
  """Publish a complete file without overwriting another process's cache."""
  path.parent.mkdir(parents=True, exist_ok=True)
  if not path.exists() and not path.is_symlink():
    temporary: Path | None = None
    try:
      with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
      temporary.chmod(0o444)
      try:
        os.link(temporary, path)
      except FileExistsError:
        pass
    finally:
      if temporary is not None:
        temporary.unlink(missing_ok=True)
  _verify_file(path, content)


def _load_generated_module(path: Path, content: bytes) -> ModuleType:
  if _MODULE_NAME in sys.modules:
    raise RuntimeError("Contact backend module name is already occupied")
  spec = importlib.util.spec_from_file_location(_MODULE_NAME, path)
  if spec is None or spec.loader is None:
    raise RuntimeError("Cannot construct isolated contact backend module")
  module = importlib.util.module_from_spec(spec)
  sys.modules[_MODULE_NAME] = module
  try:
    # Compile the verified bytes, never a possibly stale or unverified .pyc.
    # __file__/__spec__ remain available for Warp's source inspection.
    exec(compile(content, str(path), "exec"), module.__dict__)
  except BaseException:
    if sys.modules.get(_MODULE_NAME) is module:
      del sys.modules[_MODULE_NAME]
    raise
  return module


def _identity(versions: dict[str, str]) -> dict:
  return {
    "patch_id": PATCH_ID,
    "versions": dict(versions),
    "source_sha256": SOURCE_SHA256,
    "patched_sha256": PATCHED_SHA256,
    "discrete_epa_epsilon": 1.0e-7,
    "scope": "mujoco_warp._src.collision_convex.ccd (process-local explicit binding)",
  }


def install_contact_backend(cache_dir: Path | None = None) -> dict:
  """Install once, before environment creation; return a portable identity.

  An existing installation is validated and reused regardless of a later cache
  directory argument. The default cache is task-owned within TMPDIR or /dev/shm
  when available. Generated copies retain the installed source's Apache header.
  """
  global _STATE
  with _LOCK:
    if _STATE is not None:
      return active_contact_backend_identity()
    versions = _package_versions()
    original, convex, utility = _dependencies()
    if convex.ccd is not original.ccd:
      raise RuntimeError("Competing contact CCD binding exists; refusing replacement")
    builder_hashes = {hash("ccd_kernel_builder"), hash("ccd_hfield_kernel_builder")}
    if any(key[-1] in builder_hashes for key in utility._KERNEL_CACHE if key):
      raise RuntimeError("Install contact backend before any CCD kernel is built")
    original_ccd = original.ccd
    source = Path(original.__file__).resolve()
    patched = _patch_source(source.read_bytes())
    if cache_dir is None:
      root = Path(os.environ.get("TMPDIR", "/dev/shm"))
      if not root.is_dir():
        root = Path(tempfile.gettempdir())
      cache_dir = root / "pochi_lp_acrl_contact_backend"
    directory = Path(cache_dir).expanduser().resolve()
    if directory.is_relative_to(source.parent.parent):
      raise RuntimeError("Contact backend cache cannot be inside the installed package")
    generated = directory / (_MODULE_NAME + ".py")
    _write_verified_cache(generated, patched)
    module = _load_generated_module(generated, patched)
    if convex.ccd is not original_ccd or original.ccd is not original_ccd:
      if sys.modules.get(_MODULE_NAME) is module:
        del sys.modules[_MODULE_NAME]
      raise RuntimeError("Competing contact CCD binding appeared during installation")
    # No mutation of the installed package's files or original CCD function.
    convex.ccd = module.ccd
    _STATE = _Installation(source, generated, module, original_ccd, module.ccd)
    return _identity(versions)


def active_contact_backend_identity() -> dict:
  """Return identity only if the pinned sources and live binding remain intact."""
  with _LOCK:
    if _STATE is None:
      raise RuntimeError("Pochi contact backend is not installed in this process")
    versions = _package_versions()
    original, convex, _ = _dependencies()
    if (
      convex.ccd is not _STATE.installed_ccd
      or original.ccd is not _STATE.original_ccd
      or sys.modules.get(_MODULE_NAME) is not _STATE.module
      or _STATE.module.ccd is not _STATE.installed_ccd
    ):
      raise RuntimeError("Active contact backend binding has been replaced")
    if Path(original.__file__).resolve() != _STATE.source:
      raise RuntimeError("Installed contact backend source location changed")
    patched = _patch_source(_STATE.source.read_bytes())
    _verify_file(_STATE.generated, patched)
    return _identity(versions)


def contact_backend_source_paths() -> dict[str, Path]:
  """Validated source paths for archiving; paths are excluded from identity."""
  with _LOCK:
    active_contact_backend_identity()
    assert _STATE is not None
    return {
      "installed_collision_gjk": _STATE.source,
      "generated_collision_gjk": _STATE.generated,
      "installer": Path(__file__).resolve(),
    }
