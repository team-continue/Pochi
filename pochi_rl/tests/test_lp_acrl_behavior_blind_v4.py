"""Backend validation must precede the frozen evaluator's base actor load."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

_PATH = (
  Path(__file__).resolve().parents[1] / "scripts/eval_lp_acrl_behavior_blind_v4.py"
)
_SPEC = importlib.util.spec_from_file_location("blind_v4_eval_wrapper", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_WRAPPER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_WRAPPER)


@pytest.mark.parametrize(
  "saved",
  [{}, {"infos": {}}, {"infos": {"lp_acrl_contact_backend": {"epsilon": 1e-15}}}],
)
def test_missing_or_different_backend_rejected(saved):
  with pytest.raises(ValueError, match="exactly match"):
    _WRAPPER.validate_backend_identity(saved, {"epsilon": 1e-7})


def test_exact_backend_dictionary_required():
  identity = {"patch_id": "fixed", "versions": {"warp": "1"}, "epsilon": 1e-7}
  _WRAPPER.validate_backend_identity(
    {"infos": {"lp_acrl_contact_backend": identity}}, identity.copy()
  )
  with pytest.raises(ValueError, match="exactly match"):
    _WRAPPER.validate_backend_identity(
      {"infos": {"lp_acrl_contact_backend": {**identity, "extra": 1}}}, identity
    )


def _setup(monkeypatch, tmp_path, saved_identity):
  calls = []
  identity = {"patch_id": "fixed", "epsilon": 1e-7}
  checkpoint = tmp_path / "model.pt"
  checkpoint.write_bytes(b"immutable test checkpoint")
  output = tmp_path / "result.json"
  backend = ModuleType("pochi_rl.mjlab.warp_contact_backend")
  backend.install_contact_backend = lambda: calls.append("install") or identity.copy()
  backend.active_contact_backend_identity = lambda: identity.copy()
  task = ModuleType("pochi_rl.mjlab.lp_acrl_blind_v4")
  task.register_blind_v4 = lambda: calls.append("register")
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
  for module in (backend, task, plane):
    monkeypatch.setitem(sys.modules, module.__name__, module)
  monkeypatch.setattr(
    torch,
    "load",
    lambda *a, **kw: (
      calls.append("load") or {"infos": {"lp_acrl_contact_backend": saved_identity}}
    ),
  )
  monkeypatch.setattr(
    sys,
    "argv",
    [
      "wrapper",
      "--checkpoint",
      str(checkpoint),
      "--output",
      str(output),
      "--seed",
      "456",
    ],
  )
  return calls, output


def test_installs_and_checks_before_registration_or_environment(monkeypatch, tmp_path):
  calls, output = _setup(monkeypatch, tmp_path, {"patch_id": "fixed", "epsilon": 1e-7})
  _WRAPPER.main()
  assert calls == ["install", "load", "register", "evaluate"]
  report = json.loads(output.read_text())
  assert report["summary"] == {"stand": {"passed": 8}}
  assert report["contact_backend"]["epsilon"] == 1e-7
  assert report["contact_backend_validation"][
    "checkpoint_identity_checked_before_environment_creation"
  ]
  assert sys.argv[-2:] == ["--task", "Pochi-LP-ACRL-Blind-v4"]


def test_legacy_identity_stops_before_registration_or_rollout(monkeypatch, tmp_path):
  calls, output = _setup(monkeypatch, tmp_path, None)
  with pytest.raises(ValueError, match="exactly match"):
    _WRAPPER.main()
  assert calls == ["install", "load"]
  assert not output.exists()


def test_existing_output_rejected_before_installation(monkeypatch, tmp_path):
  calls, output = _setup(monkeypatch, tmp_path, None)
  output.write_text("prior result")
  with pytest.raises(SystemExit) as error:
    _WRAPPER.main()
  assert error.value.code == 2
  assert calls == [] and output.read_text() == "prior result"
