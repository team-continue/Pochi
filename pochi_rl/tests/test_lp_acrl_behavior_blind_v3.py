"""Local v3 registration preserves the frozen plane protocol and arguments."""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

_PATH = (
  Path(__file__).resolve().parents[1] / "scripts/eval_lp_acrl_behavior_blind_v3.py"
)
_SPEC = importlib.util.spec_from_file_location("blind_v3_eval_wrapper", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
_WRAPPER = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_WRAPPER)


@pytest.mark.parametrize("explicit_task", [[], ["--task=Pochi-LP-ACRL-Blind-v3"]])
def test_registration_before_frozen_evaluator_without_changing_other_args(
  monkeypatch, explicit_task
):
  calls = []
  registration = ModuleType("pochi_rl.mjlab.lp_acrl_blind_v3")
  registration.register_blind_v3 = lambda: calls.append("register")
  plane = ModuleType("eval_lp_acrl_behavior_plane")
  plane.main = lambda: calls.append(("evaluate", sys.argv.copy()))
  monkeypatch.setitem(sys.modules, registration.__name__, registration)
  monkeypatch.setitem(sys.modules, plane.__name__, plane)
  original = ["wrapper", "--checkpoint", "model.pt", "--output", "new.json"]
  monkeypatch.setattr(sys, "argv", original.copy() + explicit_task)
  _WRAPPER.main()
  assert calls[0] == "register"
  assert calls[1][0] == "evaluate"
  assert calls[1][1][: len(original)] == original
  assert "Pochi-LP-ACRL-Blind-v3" in " ".join(calls[1][1])


def test_wrong_task_fails_before_registration_or_evaluation(monkeypatch):
  monkeypatch.setattr(sys, "argv", ["wrapper", "--task", "Pochi-LP-ACRL-Blind-v2"])
  with pytest.raises(SystemExit) as error:
    _WRAPPER.main()
  assert error.value.code == 2
