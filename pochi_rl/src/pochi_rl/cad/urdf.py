"""Read the onshape-to-robot URDF export into the converter's CAD model.

This is the preferred input: the assembly's twelve ``dof_<LEG>_<KIND>`` mates
give the joint axes explicitly, so nothing has to be inferred from how the
motors happen to be placed.  It produces the same ``RobotModel`` that the GLB
reader does, so ``convert.build_mjcf`` consumes either one unchanged.

The export is a *flat* one: onshape-to-robot turns every untagged Onshape mate
into a joint, so the file carries ~1600 spurious prismatic/cylindrical joints
for bolts sitting in holes.  Those are all rigid in reality, so this module cuts
the tree only at the twelve ``dof_`` mates and welds everything else into the
thirteen rigid links.  The spurious joints are all at zero, which *is* the
assembled pose, so forward kinematics reproduces the CAD exactly -- verified in
``tests/test_urdf_kinematics.py``.
"""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh

DOF_RE = re.compile(r"^dof_(?P<leg>FL|FR|RL|RR)_(?P<kind>hip_roll|hip_pitch|knee)$")


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
  cr, cp, cy = np.cos([roll, pitch, yaw])
  sr, sp, sy = np.sin([roll, pitch, yaw])
  return np.array(
    [
      [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
      [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
      [-sp, cp * sr, cp * cr],
    ]
  )


def _origin(element: ET.Element | None) -> np.ndarray:
  """The 4x4 transform of a URDF ``<origin>`` (identity when absent)."""
  T = np.eye(4)
  if element is None:
    return T
  node = element.find("origin")
  if node is None:
    return T
  T[:3, :3] = _rpy(*(float(v) for v in node.get("rpy", "0 0 0").split()))
  T[:3, 3] = [float(v) for v in node.get("xyz", "0 0 0").split()]
  return T


class UrdfMeshCache:
  """Watertight meshes keyed by STL stem, mirroring ``convert.MeshCache``."""

  def __init__(self, mesh_dir: Path, provenance: str = "") -> None:
    self.mesh_dir = mesh_dir
    self.provenance = provenance
    self._cache: dict[str, trimesh.Trimesh] = {}
    self._decimated: dict[tuple[str, int], trimesh.Trimesh] = {}

  def get(self, key: str) -> trimesh.Trimesh:
    if key not in self._cache:
      mesh = trimesh.load_mesh(self.mesh_dir / f"{key}.stl", process=True)
      if mesh.is_volume and mesh.volume < 0:
        mesh.invert()
      self._cache[key] = mesh
    return self._cache[key]

  def decimated(self, key: str, ratio: float):
    from pochi_rl.cad.convert import MIN_PART_FACES, _decimate

    mesh = self.get(key)
    if ratio >= 1.0:
      return mesh
    target = max(MIN_PART_FACES, int(len(mesh.faces) * ratio))
    cache_key = (key, target)
    if cache_key not in self._decimated:
      self._decimated[cache_key] = _decimate(mesh, target)
    return self._decimated[cache_key]


@dataclass
class UrdfJoint:
  name: str
  parent: str
  child: str
  origin: np.ndarray  # 4x4, in the parent link frame
  axis: np.ndarray  # unit, in the joint frame


def _read(urdf_path: Path):
  root = ET.parse(urdf_path).getroot()
  links = {link.get("name"): link for link in root.findall("link")}
  joints: list[UrdfJoint] = []
  for j in root.findall("joint"):
    axis = j.find("axis")
    xyz = axis.get("xyz") if axis is not None else None
    vec = (
      np.array([float(v) for v in xyz.split()]) if xyz else np.array([0.0, 0.0, 1.0])
    )
    parent, child = j.find("parent"), j.find("child")
    if parent is None or child is None:
      raise ValueError(f"joint {j.get('name')!r} has no parent/child link")
    joints.append(
      UrdfJoint(
        name=str(j.get("name")),
        parent=str(parent.get("link")),
        child=str(child.get("link")),
        origin=_origin(j),
        axis=vec / np.linalg.norm(vec),
      )
    )
  return links, joints


def _forward_kinematics(
  joints: list[UrdfJoint], root_link: str
) -> dict[str, np.ndarray]:
  """World transform of every link with all joints at zero.

  Zero is the as-assembled pose for an Onshape export, so this reproduces the
  CAD placement -- for the spurious mates as much as for the real joints.
  """
  children = defaultdict(list)
  for j in joints:
    children[j.parent].append(j)
  world = {root_link: np.eye(4)}
  stack = [root_link]
  while stack:
    parent = stack.pop()
    for j in children[parent]:
      if j.child in world:
        continue  # loop closure: the tree branch already reached it
      world[j.child] = world[parent] @ j.origin
      stack.append(j.child)
  return world


def load(urdf_path: Path):
  """Build a ``convert.RobotModel`` from the URDF package.

  Returns ``(model, meshes)``; feed both straight to ``convert.build_mjcf``.
  """
  from pochi_rl.cad import convert as C

  urdf_path = Path(urdf_path)
  links, joints = _read(urdf_path)
  digest = hashlib.sha256(urdf_path.read_bytes()).hexdigest()
  meshes = UrdfMeshCache(
    urdf_path.resolve().parent.parent / "meshes",
    provenance=f"{urdf_path.name} sha256:{digest[:16]}",
  )

  by_child = {j.child: j for j in joints}
  root_link = next(name for name in links if name not in by_child)
  world = _forward_kinematics(joints, root_link)
  if len(world) != len(links):
    raise ValueError(f"{len(links) - len(world)} links unreachable from {root_link!r}")

  dof = {}
  for j in joints:
    m = DOF_RE.match(j.name)
    if m:
      dof[(m["leg"], m["kind"])] = j
  missing = {
    (leg, kind) for leg in C.LEGS for kind in ("hip_roll", "hip_pitch", "knee")
  } - set(dof)
  if missing:
    raise ValueError(
      f"{urdf_path} is missing {len(missing)} dof_ mates: {sorted(missing)}. "
      "Name the twelve revolute mates dof_<LEG>_<KIND> in Onshape."
    )

  # Cut the tree at the twelve dof mates; every other joint is rigid, so each
  # cut subtree is one link of the robot.
  cut = {j.child for j in dof.values()}
  children = defaultdict(list)
  for j in joints:
    children[j.parent].append(j)

  def subtree(start: str) -> list[str]:
    out, stack, seen = [], [start], {start}
    while stack:
      name = stack.pop()
      out.append(name)
      for j in children[name]:
        if j.child in cut or j.child in seen:
          continue
        seen.add(j.child)
        stack.append(j.child)
    return out

  def assembly(name: str, start: str) -> C.Assembly:
    parts = []
    for link_name in subtree(start):
      mesh = links[link_name].find("visual/geometry/mesh")
      if mesh is None:
        continue  # mate connectors and loop-closure stubs carry no geometry
      stem = Path(str(mesh.get("filename"))).name.removesuffix(".stl")
      # The visual has its own origin inside the link frame.
      placement = world[link_name] @ _origin(links[link_name].find("visual"))
      parts.append(C.Part(stem, stem, placement))
    return C.Assembly(name=name, parts=parts)

  body = assembly("body", root_link)

  # Base frame: centred between the four hip joints (where each leg's roll and
  # pitch axes cross), oriented to MuJoCo's x-forward, y-left, z-up.
  def axis_line(j: UrdfJoint) -> tuple[np.ndarray, np.ndarray]:
    W = world[j.parent] @ j.origin
    d = W[:3, :3] @ j.axis
    return W[:3, 3], d / np.linalg.norm(d)

  hips = {}
  for leg in C.LEGS:
    p1, d1 = axis_line(dof[(leg, "hip_roll")])
    p2, d2 = axis_line(dof[(leg, "hip_pitch")])
    w = p1 - p2
    a, b, c = d1 @ d1, d1 @ d2, d2 @ d2
    dd, e = d1 @ w, d2 @ w
    den = a * c - b * b
    if abs(den) < 1e-9:
      raise ValueError(f"{leg}: hip roll and pitch axes are parallel")
    s, t = (b * e - c * dd) / den, (a * e - b * dd) / den
    q1, q2 = p1 + s * d1, p2 + t * d2
    if np.linalg.norm(q1 - q2) > 1e-4:
      raise ValueError(f"{leg}: hip roll and pitch axes do not intersect")
    hips[leg] = (q1 + q2) / 2.0

  centre = np.mean(list(hips.values()), axis=0)
  G = C._translate(-(C.CAD_TO_ROBOT[:3, :3] @ centre)) @ C.CAD_TO_ROBOT

  legs = {}
  for leg in C.LEGS:
    roll_joint = dof[(leg, "hip_roll")]
    # This leg's roll motor is simply the dof mate's parent link.
    roll_motor = C.Part(
      C.MOTOR_PART,
      C.MOTOR_PART,
      world[roll_joint.parent] @ _origin(links[roll_joint.parent].find("visual")),
    )
    legs[leg] = C._build_leg(
      leg,
      G,
      roll_motor,
      assembly(f"{leg}_hip", roll_joint.child),
      assembly(f"{leg}_thigh", dof[(leg, "hip_pitch")].child),
      assembly(f"{leg}_shank", dof[(leg, "knee")].child),
      meshes,
    )

  return C.RobotModel(body, G, legs), meshes
