"""Temporary post-import fix: rotate the exported model to z-up, x-forward.

The Onshape assembly is not yet oriented to the MuJoCo convention
(x-forward, y-left, z-up), so the raw export spawns pitched 90 degrees
nose-up. Until the assembly is reoriented at the source, wrap all top-level
bodies in a corrective frame and lift the model so its lowest point touches
z=0. Runs automatically via `post_import_commands` in config.json. Delete
this script and that config entry once the CAD is fixed.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

# Rotation (unit quaternion, wxyz) mapping the CAD frame to z-up with the
# torso's long axis along x. Solved from the plane and long axis of the four
# hipunit bodies in the raw export, then rotated 180 deg about z so +x points
# to the head end (the one carrying Hiproll_1 <1>/<2> = FL/FR). Recompute if
# the CAD orientation changes.
QUAT = "0.507204 -0.455983 0.540908 -0.492181"
FRAME_NAME = "reorient_fix"
GROUND_CLEARANCE = 0.002

MODEL_PATH = Path(__file__).resolve().parent / "pochi_cad.xml"


def _wrap_bodies(quat: str, z_offset: float) -> None:
  tree = ET.parse(MODEL_PATH)
  worldbody = tree.getroot().find("worldbody")
  if worldbody is None:
    raise SystemExit(f"no <worldbody> in {MODEL_PATH}")

  # Idempotent: unwrap any previous run's frame before re-wrapping.
  for old in worldbody.findall("frame"):
    for child in list(old):
      old.remove(child)
      worldbody.append(child)
    worldbody.remove(old)

  bodies = [el for el in list(worldbody) if el.tag == "body"]
  frame = ET.SubElement(
    worldbody,
    "frame",
    {"name": FRAME_NAME, "quat": quat, "pos": f"0 0 {z_offset:.4f}"},
  )
  for el in bodies:
    worldbody.remove(el)
    frame.append(el)
  ET.indent(tree, space="  ")
  tree.write(MODEL_PATH)


def _lowest_point_z() -> float:
  model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)
  min_z = np.inf
  for g in range(model.ngeom):
    xpos = data.geom_xpos[g]
    xmat = data.geom_xmat[g].reshape(3, 3)
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
      mesh_id = model.geom_dataid[g]
      adr = model.mesh_vertadr[mesh_id]
      num = model.mesh_vertnum[mesh_id]
      verts = model.mesh_vert[adr : adr + num]
      min_z = min(min_z, float((verts @ xmat[2]).min() + xpos[2]))
    else:
      center = model.geom_aabb[g, :3]
      half = model.geom_aabb[g, 3:]
      corners = center + half * np.array(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
      )
      min_z = min(min_z, float(((xmat @ corners.T).T + xpos)[:, 2].min()))
  return min_z


def main() -> None:
  _wrap_bodies(QUAT, 0.0)
  z_offset = GROUND_CLEARANCE - _lowest_point_z()
  _wrap_bodies(QUAT, z_offset)
  print(
    f"reorient.py: quat='{QUAT}', lifted by {z_offset:+.4f} m "
    f"so the lowest point sits at z={GROUND_CLEARANCE}"
  )


if __name__ == "__main__":
  main()
