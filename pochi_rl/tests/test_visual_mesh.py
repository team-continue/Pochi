"""Keep thin CAD plates and mechanical edges intact through visual export."""

from __future__ import annotations

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")


def _triangles(mesh):
  # Shading may reorder faces and split vertices, but must preserve the surface.
  return sorted(tuple(triangle.ravel()) for triangle in mesh.triangles)


def test_cad_visual_preserves_thin_parts_and_motor_geometry(tmp_path) -> None:
  from pochi_rl.cad.convert import Part, build_visual_mesh
  from pochi_rl.cad.urdf import UrdfMeshCache

  plate = trimesh.creation.box(extents=[0.101, 0.047, 0.002])
  plate.apply_translation([0.0007, 0.0004, 0.0002])
  motor = trimesh.creation.cylinder(radius=0.03925, height=0.0454, sections=64)
  for name, mesh in (("plate", plate), ("RS02", motor)):
    mesh.export(tmp_path / f"{name}.stl")
  cache = UrdfMeshCache(tmp_path)
  source_plate = cache.get("plate").copy()
  source_motor = cache.get("RS02").copy()
  placement = np.eye(4)
  placement[:3, 3] = [0.04, 0.03, 0.06]
  parts = [
    Part("plate", "plate", np.eye(4)),
    Part("RS02", "RS02", placement),
    Part("M3_fastener", "plate", placement),
  ]
  result = build_visual_mesh(parts, cache, np.eye(4))
  expected = trimesh.util.concatenate(
    [source_plate, source_motor.copy().apply_transform(placement)]
  )
  np.testing.assert_allclose(_triangles(result), _triangles(expected), atol=1e-14)
  np.testing.assert_array_equal(cache.get("plate").vertices, source_plate.vertices)
  np.testing.assert_array_equal(cache.get("RS02").vertices, source_motor.vertices)


def test_plate_shading_keeps_corners_sharp_in_mujoco(tmp_path) -> None:
  from pochi_rl.cad.convert import Part, build_visual_mesh
  from pochi_rl.cad.urdf import UrdfMeshCache

  mujoco = pytest.importorskip("mujoco")
  trimesh.creation.box(extents=[0.1, 0.04, 0.002]).export(tmp_path / "plate.stl")
  mesh = build_visual_mesh(
    [Part("plate", "plate", np.eye(4))], UrdfMeshCache(tmp_path), np.eye(4)
  )
  mesh.export(tmp_path / "plate.obj", include_normals=True)
  model = mujoco.MjModel.from_xml_string(
    '<mujoco><asset><mesh name="plate" file="plate.obj"/></asset>'
    '<worldbody><geom type="mesh" mesh="plate"/></worldbody></mujoco>',
    assets={"plate.obj": (tmp_path / "plate.obj").read_bytes()},
  )
  tri = model.mesh_vert[model.mesh_face]
  face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
  face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)
  normals = model.mesh_normal[model.mesh_facenormal]
  alignment = np.einsum("fvc,fc->fv", normals, face_normals)
  np.testing.assert_allclose(alignment, 1.0, atol=1e-6)


def test_cylinder_shading_is_smooth_with_flat_end_caps(tmp_path) -> None:
  from pochi_rl.cad.convert import Part, build_visual_mesh
  from pochi_rl.cad.urdf import UrdfMeshCache

  cylinder = trimesh.creation.cylinder(radius=0.04, height=0.05, sections=64)
  cylinder.export(tmp_path / "RS02.stl")
  mesh = build_visual_mesh(
    [Part("RS02", "RS02", np.eye(4))], UrdfMeshCache(tmp_path), np.eye(4)
  )
  normals = mesh.vertex_normals[mesh.faces]
  caps = np.abs(mesh.face_normals[:, 2]) > 0.99
  alignment = np.einsum("fvc,fc->fv", normals[caps], mesh.face_normals[caps])
  np.testing.assert_allclose(alignment, 1.0, atol=1e-6)
  radial = mesh.triangles[~caps].copy()
  radial[:, :, 2] = 0
  radial /= np.linalg.norm(radial, axis=2, keepdims=True)
  alignment = np.einsum("fvc,fvc->fv", normals[~caps], radial)
  assert alignment.min() > np.cos(np.radians(2))
