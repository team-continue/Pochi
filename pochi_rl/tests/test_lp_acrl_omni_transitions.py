"""Check that in-motion diagnostics reject stopping and delayed responses."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

SCRIPT = Path(__file__).parents[1] / "scripts/eval_lp_acrl_omni_transitions.py"
spec = importlib.util.spec_from_file_location("omni_transitions", SCRIPT)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def response(vx=0.25):
  time = np.arange(1, 51) * 0.02
  velocity = np.zeros((50, 1, 3))
  velocity[:, 0, 0] = vx
  velocity[:, 0, 1] = 0.2 * np.minimum(time / 0.2, 1.0)
  return {
    "time_s": 6.0 + time,
    "velocity": velocity,
    "valid": np.ones((50, 1), dtype=bool),
    "contact": np.ones((50, 1, 4), dtype=bool),
    "nonfoot_contact": np.zeros((50, 1), dtype=bool),
    "tilt_deg": np.zeros((50, 1)),
    "slip": np.zeros((50, 1)),
  }


def score(samples, vx=0.25):
  return module.transition_metrics(
    samples, 0, 6.0, (vx, 0.0, 0.0), (vx, 0.2, 0.0), 0.02
  )


@pytest.mark.parametrize("vx", [0.25, -0.25])
def test_finite_lateral_response_retains_forward_or_backward_motion(vx):
  result = score(response(vx), vx)
  assert result["passed"]
  assert 0.5 <= result["first_qualified_500ms_window_end_s_after_switch"] <= 0.7


@pytest.mark.parametrize("fault", ["stopped", "wrong_side", "nonfoot", "invalid"])
def test_switch_failure_cannot_be_hidden_by_later_steady_response(fault):
  samples = response()
  if fault == "stopped":
    samples["velocity"][:, 0, 0] = 0.0
  elif fault == "wrong_side":
    samples["velocity"][:, 0, 1] = -0.2
  elif fault == "nonfoot":
    samples["nonfoot_contact"][0, 0] = True
  elif fault == "invalid":
    samples["valid"][0, 0] = False
  result = score(samples)
  assert not result["passed"]
  if fault == "stopped":
    assert not result["checks"]["retain_direction_at_least90pct"]
  elif fault == "wrong_side":
    assert result["first_qualified_500ms_window_end_s_after_switch"] is None


def test_transition_report_is_json_serializable():
  result = score(response())
  restored = json.loads(json.dumps(result))
  assert restored["passed"] is True
  assert restored["checks"] == result["checks"]
  assert restored["after"] == list(result["after"])
