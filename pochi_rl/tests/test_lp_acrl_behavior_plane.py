"""Plane geometry contract and post-forward contact-cache alignment."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/eval_lp_acrl_behavior_plane.py"
_SPEC = importlib.util.spec_from_file_location("plane_behavior", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_EVAL = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EVAL)


def scene():
  return mujoco.MjSpec.from_string("""<mujoco><worldbody>
  <body name="terrain"><geom name="tile0" type="box" pos="12 12 -.5" size="12 12 .5"/>
  <geom name="tile1" type="box" pos="36 12 -.5" size="12 12 .5"/></body>
  <body name="robot" pos="0 0 1"><geom name="foot" type="sphere" size=".02"/></body>
  </worldbody></mujoco>""")


def test_plane_override_preserves_robot_and_disables_all_other_ground():
  spec = scene()
  original = spec.compile()
  _EVAL.install_single_plane(spec)
  model = spec.compile()
  info = _EVAL.verify_single_plane(model)
  assert info["kind"] == "single_infinite_horizontal_plane"
  assert info["collision_disabled_other_terrain_geoms"] == 1
  assert not info["robot_geometry_override"]
  for field in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_friction"):
    assert np.array_equal(getattr(model, field)[2], getattr(original, field)[2])


@pytest.mark.parametrize("fault", ["box", "extra", "tilted", "height"])
def test_compiled_ground_verification_rejects_wrong_geometry(fault):
  spec = scene()
  _EVAL.install_single_plane(spec)
  if fault == "box":
    spec.geom("tile0").type = mujoco.mjtGeom.mjGEOM_BOX
    spec.geom("tile0").size = (12, 12, 0.5)
  elif fault == "extra":
    spec.geom("tile1").contype = 1
  elif fault == "tilted":
    spec.geom("tile0").quat = (0.70710678, 0.70710678, 0, 0)
  else:
    spec.geom("tile0").pos[2] = 0.1
  with pytest.raises(ValueError):
    _EVAL.verify_single_plane(spec.compile())


def test_contact_refresh_occurs_after_step_and_does_not_advance_physics():
  calls = []

  class Sensor:
    def _invalidate_cache(self):
      calls.append("invalidate")

  class Wrapped:
    def step(self, actions):
      calls.append("step")
      return actions

  env = SimpleNamespace(
    scene={
      "feet_ground_contact": Sensor(),
      "limb_ground_contact": Sensor(),
    }
  )
  wrapped = _EVAL.FreshContactStepWrapper(env, Wrapped())
  assert wrapped.step("result") == "result"
  assert calls == ["step", "invalidate", "invalidate"]


def test_plane_protocol_reuses_frozen_thresholds_and_profiles():
  assert _EVAL.STAND_LIMITS is _EVAL.frozen.STAND_LIMITS
  assert _EVAL.WALK_LIMITS is _EVAL.frozen.WALK_LIMITS
  assert _EVAL.PROFILES is _EVAL.frozen.PROFILES
  assert _EVAL.MIN_AIR_TIME_S == 0.04
