"""Post-import stand-in for the Onshape mate connectors.

The assembly has no `link_base_link` / `frame_*` mate connectors yet, so this
script adds their equivalents to the exported MJCF instead: it renames the
root body to `base_link`, adds an `imu` site at its origin, and adds a
`<LEG>_foot_site` at the bottom of each shank (centroid of the mesh vertices
near the lowest point of the body attached to the `<LEG>_knee` joint).

Runs via `post_import_commands` in config.json, after reorient.py (the foot
search assumes z-up). Site names match `pochi_constants.py`. If the mate
connectors get added in CAD later, delete this script and that config entry.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "pochi_cad.xml"
BASE_LINK = "base_link"
LEGS = ("FL", "FR", "RL", "RR")
FOOT_BAND = 0.006  # vertices within this height of the lowest point


def _rename_base(tree: ET.ElementTree) -> None:
  root = tree.getroot()
  parents = {child: parent for parent in root.iter() for child in parent}
  free = root.find(".//freejoint")
  if free is None:
    raise SystemExit("no <freejoint> found; is the export broken?")
  base = parents[free]
  old = base.get("name")
  base.set("name", BASE_LINK)
  for el in root.iter():
    for attr in ("body", "body1", "body2"):
      if el.get(attr) == old:
        el.set(attr, BASE_LINK)


def _body_element(root: ET.Element, name: str) -> ET.Element:
  el = root.find(f".//body[@name='{name}']")
  if el is None:
    raise SystemExit(f"body {name!r} not found in {MODEL_PATH}")
  return el


def _add_site(body_el: ET.Element, name: str, pos: np.ndarray, size: str) -> None:
  for old in body_el.findall(f"site[@name='{name}']"):
    body_el.remove(old)
  ET.SubElement(
    body_el,
    "site",
    {"name": name, "pos": " ".join(f"{v:.6f}" for v in pos), "size": size},
  )


def _foot_point_local(
  model: mujoco.MjModel, data: mujoco.MjData, body_id: int
) -> np.ndarray:
  """Centroid of the body's lowest mesh vertices, in body-local coordinates."""
  verts_world = []
  for g in range(model.ngeom):
    if model.geom_bodyid[g] != body_id:
      continue
    xpos = data.geom_xpos[g]
    xmat = data.geom_xmat[g].reshape(3, 3)
    if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
      mesh_id = model.geom_dataid[g]
      adr = model.mesh_vertadr[mesh_id]
      num = model.mesh_vertnum[mesh_id]
      verts_world.append(model.mesh_vert[adr : adr + num] @ xmat.T + xpos)
  if not verts_world:
    raise SystemExit(f"body id {body_id} has no mesh geoms")
  verts = np.vstack(verts_world)
  low = verts[verts[:, 2] < verts[:, 2].min() + FOOT_BAND]
  point_world = low.mean(axis=0)
  xmat = data.xmat[body_id].reshape(3, 3)
  return xmat.T @ (point_world - data.xpos[body_id])


def main() -> None:
  tree = ET.parse(MODEL_PATH)
  _rename_base(tree)
  root = tree.getroot()

  model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
  data = mujoco.MjData(model)
  mujoco.mj_forward(model, data)

  _add_site(_body_element(root, BASE_LINK), "imu", np.zeros(3), "0.01")

  for leg in LEGS:
    joint_id = mujoco.mj_name2id(
      model, mujoco.mjtObj.mjOBJ_JOINT, f"{leg}_knee"
    )
    if joint_id < 0:
      raise SystemExit(f"joint {leg}_knee not found; check the dof_ mate names")
    body_id = int(model.jnt_bodyid[joint_id])
    body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    pos = _foot_point_local(model, data, body_id)
    _add_site(_body_element(root, body_name), f"{leg}_foot_site", pos, "0.012")
    print(f"add_frames.py: {leg}_foot_site on {body_name!r} at {pos.round(4)}")

  ET.indent(tree, space="  ")
  tree.write(MODEL_PATH)
  print(f"add_frames.py: renamed root body to {BASE_LINK!r}, added imu site")


if __name__ == "__main__":
  main()
