"""The URDF's dof_ mates must describe a symmetric robot.

The export buries the twelve real joints among ~1600 spurious ones (every
untagged Onshape mate becomes a joint), so these check that cutting the tree at
the dof_ mates recovers the machine the CAD actually describes.
"""

from __future__ import annotations

from pathlib import Path

import pytest

URDF = (
  Path(__file__).resolve().parents[2]
  / "pkg_4leg_assem2"
  / "urdf"
  / "pkg_4leg_assem2.urdf"
)


@pytest.fixture(scope="module")
def model():
  pytest.importorskip("trimesh")
  if not URDF.is_file():
    pytest.skip("onshape-to-robot URDF export not present")
  from pochi_rl.cad import urdf

  return urdf.load(URDF)[0]


def test_all_twelve_dof_mates_are_present(model) -> None:
  from pochi_rl.robot import LEGS

  assert set(model.legs) == set(LEGS)
  for leg in LEGS:
    assert set(model.legs[leg].assemblies) == {"hip", "thigh", "shank"}


def test_link_geometry_matches_the_shared_constants(model) -> None:
  """Guards the constants every backend builds on against a CAD change."""
  from pochi_rl.robot.pochi_constants import (
    FOOT_OFFSET_Y,
    HIP_OFFSET_X,
    HIP_OFFSET_Y,
    SHANK_LENGTH,
    THIGH_LENGTH,
  )

  for leg, lm in model.legs.items():
    assert abs(lm.hip_pos[0]) == pytest.approx(HIP_OFFSET_X, abs=1e-4), leg
    assert abs(lm.hip_pos[1]) == pytest.approx(HIP_OFFSET_Y, abs=1e-4), leg
    assert lm.hip_pos[2] == pytest.approx(0.0, abs=1e-4), leg
    assert lm.thigh_length == pytest.approx(THIGH_LENGTH, abs=2e-4), leg
    assert lm.shank_length == pytest.approx(SHANK_LENGTH, abs=2e-4), leg
    assert abs(lm.foot_pos[1]) == pytest.approx(FOOT_OFFSET_Y, abs=5e-4), leg


def test_the_four_legs_are_identical(model) -> None:
  """A left/right or front/rear difference here is a CAD mate defect.

  The foot tip is a plate edge whose two faces tie for farthest-from-the-knee,
  so this also pins the tie-breaking in ``convert._build_leg``: picking a single
  extreme vertex instead of centring the tip put the feet half a plate width
  apart, and not even consistently per leg.
  """
  legs = list(model.legs.values())
  for attr in ("thigh_length", "shank_length"):
    values = [getattr(lm, attr) for lm in legs]
    assert max(values) - min(values) < 5e-4, (attr, values)
  lateral = [abs(lm.foot_pos[1]) for lm in legs]
  assert max(lateral) - min(lateral) < 5e-4, lateral
  fore_aft = [abs(lm.hip_pos[0]) for lm in legs]
  assert max(fore_aft) - min(fore_aft) < 1e-4, fore_aft


def test_every_part_lands_in_exactly_one_link(model) -> None:
  """Cutting at the dof mates must partition the assembly, not drop parts."""
  total = len(model.base_assembly.parts) + sum(
    len(asm.parts) for lm in model.legs.values() for asm in lm.assemblies.values()
  )
  import xml.etree.ElementTree as ET

  root = ET.parse(URDF).getroot()
  with_mesh = sum(
    1 for link in root.findall("link") if link.find("visual/geometry/mesh") is not None
  )
  assert total == with_mesh
