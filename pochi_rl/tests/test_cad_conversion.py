"""Checks on the CAD -> MJCF conversion.

Skipped unless the CAD export is present; it is not committed to the repo.
"""

from __future__ import annotations

from pathlib import Path

import pytest

GLB = Path(__file__).resolve().parents[2] / "4leg_assem2.glb"

pytestmark = pytest.mark.skipif(not GLB.exists(), reason="CAD export not available")


@pytest.fixture(scope="module")
def cad():
  pytest.importorskip("trimesh")
  from pochi_rl.cad import convert
  from pochi_rl.cad.glb import Glb

  glb = Glb(GLB)
  meshes = convert.MeshCache(glb)
  return convert, meshes, convert.build_kinematics(glb, meshes)


def test_legs_are_symmetric(cad) -> None:
  _convert, _meshes, model = cad
  for leg, lm in model.legs.items():
    assert lm.thigh_length == pytest.approx(0.200, abs=1e-4)
    assert lm.shank_length == pytest.approx(0.225, abs=1e-4)
    assert abs(lm.hip_pos[0]) == pytest.approx(0.2545, abs=1e-4)
    assert abs(lm.hip_pos[1]) == pytest.approx(0.070, abs=1e-4)
    assert lm.hip_pos[0] > 0 if leg.startswith("F") else lm.hip_pos[0] < 0
    assert lm.hip_pos[1] > 0 if leg.endswith("L") else lm.hip_pos[1] < 0


def test_forward_kinematics_reproduces_the_cad_pose(cad) -> None:
  """Driving the generated model to the CAD joint angles must rebuild the CAD."""
  mujoco = pytest.importorskip("mujoco")
  import numpy as np

  convert, meshes, model = cad
  mj = mujoco.MjModel.from_xml_path(str(Path("assets") / "pochi" / "pochi.xml"))
  data = mujoco.MjData(mj)
  data.qpos[:3] = 0.0
  data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
  for leg, lm in model.legs.items():
    for kind, angle in lm.cad_angles.items():
      joint = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_{kind}")
      data.qpos[mj.jnt_qposadr[joint]] = angle
  mujoco.mj_forward(mj, data)

  for leg, lm in model.legs.items():
    for part in ("hip", "thigh", "shank"):
      body = mujoco.mj_name2id(mj, mujoco.mjtObj.mjOBJ_BODY, f"{leg}_{part}")
      points = convert._assembly_points(lm.assemblies[part], meshes, stride=97)
      posed = (
        convert._apply(lm.transforms[part], points) @ data.xmat[body].reshape(3, 3).T
        + data.xpos[body]
      )
      expected = convert._apply(model.base_transform, points)
      assert np.abs(posed - expected).max() < 1e-6, f"{leg}_{part}"


def test_urdf_mass_table_covers_every_part() -> None:
  """Onshape's masses must reach every part, not silently fall back.

  The density-by-name fallback is 0.8 kg heavier, so a lookup that quietly
  misses would inflate the robot without failing anything else.  ``Part 1``
  is the case that matters: one CAD name covering six distinct solids, which
  onshape-to-robot splits into Part_1 and Part_1_1 .. Part_1_5.
  """
  pytest.importorskip("trimesh")
  from pochi_rl.cad import convert
  from pochi_rl.cad.glb import Glb

  urdf = (
    Path(__file__).resolve().parents[2]
    / "pkg_4leg_assem2"
    / "urdf"
    / "pkg_4leg_assem2.urdf"
  )
  if not urdf.is_file():
    pytest.skip("onshape-to-robot URDF export not present")

  table = convert.load_urdf_part_masses(urdf)
  glb_path = Path(__file__).resolve().parents[2] / "4leg_assem2.glb"
  if not glb_path.is_file():
    pytest.skip("CAD GLB not present")
  meshes = convert.MeshCache(Glb(glb_path))

  fell_back = set()
  for asm in convert.read_assemblies(meshes.glb):
    for part in asm.parts:
      if part.is_motor:
        continue  # no material in CAD; the datasheet mass is used instead
      volume = abs(meshes.get(part.mesh_index).volume)
      guess = volume * convert.density_for(part.name)
      if convert.mass_for(part.name, volume, table) == guess:
        fell_back.add(part.name)
  assert not fell_back, f"no Onshape mass for {sorted(fell_back)}"
