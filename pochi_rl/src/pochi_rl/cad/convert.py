"""Convert the Pochi CAD export (``4leg_assem2.glb``) into an MJCF robot model.

The GLB carries no kinematics, only a flat list of placed parts.  The joint
frames are recovered from the twelve RobStride ``RS02`` actuators: each motor's
local ``+z`` is its output axis, so the motor placements give exact joint axes
and origins.

  * ``Body_2``   holds 4 motors -> hip roll joints  (axis: fore/aft)
  * ``Hiproll_1`` holds 1 motor -> hip pitch joint  (axis: lateral)
  * ``Hippitch_1`` holds 1 motor -> knee joint      (axis: lateral)
  * ``Knee_1``   holds no motor -> shank + foot

The CAD assembly is posed with the legs folded, and not perfectly symmetric, so
every link is rotated back about its own joint axis into a canonical zero pose:
all joints at zero puts every leg straight down.

Masses come from the onshape-to-robot URDF export, which carries the mass
Onshape computed from each part's assigned material.  Without it the converter
falls back to mesh volume times a density guessed from the part name.  The
motors use the RS02 datasheet mass either way: they are the one part with no
material assigned in CAD, so the URDF reports them as 0.18 g.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np
import trimesh

from pochi_rl.cad.glb import Glb
from pochi_rl.robot.pochi_constants import (
  LEGS,
  PAYLOAD_DENSITY,
  PAYLOAD_MASS_KG,
  PAYLOAD_POS,
  RS02_PEAK_TORQUE_NM,
  RS02_REFLECTED_INERTIA,
)
from pochi_rl.robot.pochi_constants import RS02_MASS_KG as _RS02_MASS_KG_DEFAULT

# --- Physical assumptions -----------------------------------------------------

# The RS02 datasheet numbers live in ``pochi_rl.robot.pochi_constants`` so that
# every backend sees the same actuator.  ``RS02_MASS_KG`` stays a module global
# because ``scripts/glb_to_mjcf.py --rs02-mass`` rebinds it to sweep the mass
# budget without editing the source.
RS02_MASS_KG = _RS02_MASS_KG_DEFAULT
# Envelope of the physical module, used to stand in for its 47k-triangle mesh.
RS02_RADIUS_M = 0.03925
RS02_LENGTH_M = 0.0454
RS02_ARMATURE = RS02_REFLECTED_INERTIA

# Density by part-name prefix [kg/m^3].
DENSITIES: tuple[tuple[str, float], ...] = (
  ("M3_", 7850.0),  # steel fasteners
  ("M4_", 7850.0),
  ("carbon", 1600.0),  # CFRP plates
  ("plate_hip_3_2_spacerAl", 2700.0),  # aluminium spacers
  ("Body_pipe", 2700.0),  # aluminium tube
  # Fallback only.  Onshape says these are ~1050 kg/m^3 (printed plastic) for
  # the large ones and steel for the small; see load_urdf_part_masses.
  ("Part 1", 2700.0),
)
# Fasteners are counted for mass but dropped from the visual meshes.
VISUAL_SKIP = ("M3_", "M4_")

MOTOR_PART = "RS02"

# Target triangle budget per exported visual mesh.  Decimation happens per part
# (each part is a clean closed solid); decimating the merged link soup barely
# reduces anything because almost every edge borders a different component.
VISUAL_FACE_BUDGET = {"base": 8000, "hip": 3000, "thigh": 3500, "shank": 3000}
MIN_PART_FACES = 48
# Vertex-clustering pitches tried when quadric decimation stalls on a part that
# is riddled with bolt holes [m].
CLUSTER_PITCHES = (0.0015, 0.003, 0.006)

# Joint travel is not encoded in the CAD; these are conservative software
# limits, tighten once the hardware end stops are measured.
JOINT_RANGES = {
  "hip_roll": (-0.7, 0.7),
  "hip_pitch": (-1.6, 1.6),
  "knee": (-2.4, 2.4),
}

# Base height that puts the feet on the ground at the default crouch pose.
STAND_HEIGHT = 0.32

# Vertices this close to the farthest-from-the-knee point count as the foot tip.
# Wide enough to span both faces of the tip plate under the CAD's pose tilt.
FOOT_TIE_TOL = 3.0e-4

_IDENTITY = np.eye(4)


# --- Small rigid-transform helpers -------------------------------------------


def _rot(axis: np.ndarray, angle: float) -> np.ndarray:
  """Rotation matrix about a unit ``axis`` through the origin."""
  axis = axis / np.linalg.norm(axis)
  kx, ky, kz = axis
  K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
  return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)


def _rot_about_line(axis: np.ndarray, point: np.ndarray, angle: float) -> np.ndarray:
  """4x4 rotation about the line through ``point`` along ``axis``."""
  R = _rot(axis, angle)
  T = np.eye(4)
  T[:3, :3] = R
  T[:3, 3] = point - R @ point
  return T


def _translate(offset: np.ndarray) -> np.ndarray:
  T = np.eye(4)
  T[:3, 3] = offset
  return T


def _apply(T: np.ndarray, points: np.ndarray) -> np.ndarray:
  points = np.atleast_2d(points)
  return points @ T[:3, :3].T + T[:3, 3]


def _apply_point(T: np.ndarray, point: np.ndarray) -> np.ndarray:
  return T[:3, :3] @ point + T[:3, 3]


# CAD axes are x = lateral, y = fore/aft, z = up.  MuJoCo wants x forward,
# y left, z up.
CAD_TO_ROBOT = np.array(
  [
    [0.0, 1.0, 0.0, 0.0],
    [-1.0, 0.0, 0.0, 0.0],
    [0.0, 0.0, 1.0, 0.0],
    [0.0, 0.0, 0.0, 1.0],
  ]
)

X_AXIS = np.array([1.0, 0.0, 0.0])
Y_AXIS = np.array([0.0, 1.0, 0.0])


# --- CAD scene model ----------------------------------------------------------


class MeshSource(Protocol):
  """Where link meshes come from: GLB mesh indices or URDF STL stems."""

  @property
  def provenance(self) -> str:
    """Which CAD export this model was built from."""

  def get(self, key) -> trimesh.Trimesh: ...

  def decimated(self, key, ratio: float) -> trimesh.Trimesh: ...


@dataclass
class Part:
  """One placed CAD part."""

  name: str
  mesh_index: int | str  # glTF mesh index, or the STL stem for a URDF export
  world: np.ndarray  # 4x4, CAD frame

  @property
  def is_motor(self) -> bool:
    return self.name == MOTOR_PART

  @property
  def is_visual(self) -> bool:
    return not self.name.startswith(VISUAL_SKIP)


@dataclass
class Assembly:
  """One top-level CAD subassembly, i.e. one rigid link."""

  name: str
  parts: list[Part] = field(default_factory=list)

  @property
  def family(self) -> str:
    return re.sub(r"\s*<\d+>$", "", self.name)


def read_assemblies(glb: Glb) -> list[Assembly]:
  root = glb.nodes[glb.gltf["scenes"][glb.gltf.get("scene", 0)]["nodes"][0]]
  out = []
  for top_index in root["children"]:
    top = glb.nodes[top_index]
    top_world = Glb.local_transform(top)
    asm = Assembly(name=top["name"])
    for child in top.get("children", []):
      part_name = glb.nodes[child]["name"].removeprefix("occurrence of ")
      for _idx, node, world in glb.walk(child, top_world):
        if "mesh" in node:
          asm.parts.append(Part(part_name, node["mesh"], world))
    out.append(asm)
  return out


class MeshCache:
  """Watertight, correctly wound meshes keyed by glTF mesh index."""

  def __init__(self, glb: Glb) -> None:
    self.glb = glb
    self._cache: dict[int, trimesh.Trimesh] = {}
    self._decimated: dict[tuple[int, int], trimesh.Trimesh] = {}

  @property
  def provenance(self) -> str:
    """Which CAD export this model was built from."""
    return f"{self.glb.path.name} sha256:{self.glb.digest[:16]}"

  def get(self, index: int) -> trimesh.Trimesh:
    if index not in self._cache:
      verts, faces = self.glb.mesh_geometry(index)
      mesh = trimesh.Trimesh(verts, faces, process=True)
      if mesh.is_volume and mesh.volume < 0:
        mesh.invert()
      self._cache[index] = mesh
    return self._cache[index]

  def decimated(self, index: int, ratio: float) -> trimesh.Trimesh:
    mesh = self.get(index)
    if ratio >= 1.0:
      return mesh
    target = max(MIN_PART_FACES, int(len(mesh.faces) * ratio))
    key = (index, target)
    if key not in self._decimated:
      self._decimated[key] = _decimate(mesh, target)
    return self._decimated[key]


def _cluster(mesh: trimesh.Trimesh, pitch: float) -> trimesh.Trimesh:
  snapped = trimesh.Trimesh(
    np.round(mesh.vertices / pitch) * pitch, mesh.faces, process=True
  )
  snapped.update_faces(snapped.nondegenerate_faces())
  snapped.update_faces(snapped.unique_faces())
  snapped.remove_unreferenced_vertices()
  return snapped


def _decimate(mesh: trimesh.Trimesh, target: int) -> trimesh.Trimesh:
  if len(mesh.faces) <= target:
    return mesh
  out = mesh.simplify_quadric_decimation(face_count=target)
  for pitch in CLUSTER_PITCHES:
    if len(out.faces) <= target * 1.5:
      break
    clustered = _cluster(mesh, pitch)
    out = (
      clustered.simplify_quadric_decimation(face_count=target)
      if len(clustered.faces) > target
      else clustered
    )
  return out


def motor_visual_mesh() -> trimesh.Trimesh:
  """The RS02 as a plain cylinder; the CAD mesh is 47k triangles of detail."""
  mesh = trimesh.creation.cylinder(
    radius=RS02_RADIUS_M, height=RS02_LENGTH_M, sections=32
  )
  mesh.apply_translation([0.0, 0.0, 0.004 - RS02_LENGTH_M / 2.0])
  return mesh


def density_for(part_name: str) -> float:
  for prefix, rho in DENSITIES:
    if part_name.startswith(prefix):
      return rho
  raise KeyError(f"no density for CAD part {part_name!r}")


# --- Onshape mass properties (from the onshape-to-robot URDF export) ----------


@dataclass(frozen=True)
class UrdfPart:
  """One part as the URDF export describes it."""

  volume: float
  mass: float


def _norm_part_name(name: str) -> str:
  """onshape-to-robot sanitises part names when it writes mesh filenames."""
  return re.sub(r"[^0-9A-Za-z]", "_", name)


def load_urdf_part_masses(urdf_path: Path) -> dict[str, list[UrdfPart]]:
  """Onshape's own mass for every part, keyed by sanitised part name.

  The GLB carries no materials, so without this the converter has to guess a
  density from the part name.  Onshape computed these from the materials
  actually assigned in CAD, so they win wherever the two disagree.

  A single GLB part name can cover several distinct solids -- ``Part 1`` is six
  of them -- so each entry also carries the volume of the exported STL, letting
  ``mass_for`` tell them apart.  The URDF's own kinematics are unusable (every
  untagged Onshape mate became a joint), so only the mass table is read.
  """
  import xml.etree.ElementTree as ET

  mesh_dir = urdf_path.resolve().parent.parent / "meshes"
  table: dict[str, list[UrdfPart]] = defaultdict(list)
  seen: set[str] = set()
  for link in ET.parse(urdf_path).getroot().findall("link"):
    mesh = link.find("visual/geometry/mesh")
    mass = link.find("inertial/mass")
    if mesh is None or mass is None:
      continue
    stem = Path(str(mesh.get("filename"))).name.removesuffix(".stl")
    if stem in seen:  # instances of one part all carry the same mass
      continue
    seen.add(stem)
    stl = mesh_dir / f"{stem}.stl"
    if not stl.is_file():
      raise FileNotFoundError(f"{urdf_path} references a missing mesh: {stl}")
    volume = abs(trimesh.load_mesh(stl, process=True).volume)
    entry = UrdfPart(volume, float(str(mass.get("value"))))
    # onshape-to-robot appends _<n> to tell apart distinct solids sharing one
    # CAD name, so "Part 1" arrives as Part_1, Part_1_1 ... Part_1_5.  Index
    # every such prefix -- including the bare stem, which may itself end in a
    # number -- so a lookup by the GLB's part name finds all the variants.
    # Over-collecting is harmless: mass_for then picks by volume.
    key = stem
    while True:
      table[key].append(entry)
      trimmed = re.sub(r"_\d+$", "", key)
      if trimmed == key:
        break
      key = trimmed
  return dict(table)


def mass_for(
  part_name: str, volume: float, urdf_masses: dict[str, list[UrdfPart]] | None
) -> float:
  """Mass of one part: Onshape's value if we have it, else volume x density."""
  candidates = (urdf_masses or {}).get(_norm_part_name(part_name))
  if candidates:
    # Same name, several solids: the one whose STL volume matches is the part.
    best = min(candidates, key=lambda c: abs(volume - c.volume))
    if abs(best.volume - volume) <= 0.02 * max(volume, best.volume):
      return best.mass
  return volume * density_for(part_name)


# --- Mass properties ----------------------------------------------------------


@dataclass
class MassProps:
  mass: float
  com: np.ndarray  # in the frame the properties were accumulated in
  inertia: np.ndarray  # 3x3 about ``com``

  @staticmethod
  def zero() -> MassProps:
    return MassProps(0.0, np.zeros(3), np.zeros((3, 3)))


def _inertia_about_origin(mass: float, com: np.ndarray, inertia: np.ndarray):
  d = com
  return inertia + mass * (np.dot(d, d) * np.eye(3) - np.outer(d, d))


def _motor_mass_props(world: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
  """RS02 modelled as a uniform cylinder about its own +z output axis."""
  m = RS02_MASS_KG
  r, h = RS02_RADIUS_M, RS02_LENGTH_M
  local_com = np.array([0.0, 0.0, (0.004 + (0.004 - h)) / 2.0])
  ixx = iyy = m * (3 * r * r + h * h) / 12.0
  izz = m * r * r / 2.0
  inertia_local = np.diag([ixx, iyy, izz])
  R = world[:3, :3]
  return m, _apply_point(world, local_com), R @ inertia_local @ R.T


def add_payload(
  props: MassProps,
  mass: float = PAYLOAD_MASS_KG,
  position=PAYLOAD_POS,
  density: float = PAYLOAD_DENSITY,
) -> MassProps:
  """Bolt a ballast block onto a link's mass properties.

  Sized as a solid cube of ``density`` so it carries a real rotational inertia;
  a point mass would add weight without any resistance to rotation.
  """
  if mass <= 0.0:
    return props
  side = (mass / density) ** (1.0 / 3.0)
  block_com = np.asarray(position, dtype=float)
  block_inertia = np.eye(3) * (mass * side * side / 6.0)

  total = props.mass + mass
  com = (props.mass * props.com + mass * block_com) / total
  about_origin = _inertia_about_origin(
    props.mass, props.com, props.inertia
  ) + _inertia_about_origin(mass, block_com, block_inertia)
  inertia = about_origin - total * (np.dot(com, com) * np.eye(3) - np.outer(com, com))
  return MassProps(total, com, inertia)


def mass_properties(
  parts: list[Part],
  meshes: MeshSource,
  transform: np.ndarray,
  urdf_masses: dict[str, list[UrdfPart]] | None = None,
):
  """Accumulate mass properties of ``parts`` in the frame given by ``transform``."""
  total_m = 0.0
  first_moment = np.zeros(3)
  inertia_origin = np.zeros((3, 3))
  for part in parts:
    world = transform @ part.world
    if part.is_motor:
      m, com, inertia_com = _motor_mass_props(world)
    else:
      mesh = meshes.get(part.mesh_index)
      m = mass_for(part.name, abs(mesh.volume), urdf_masses)
      R = world[:3, :3]
      com = _apply_point(world, mesh.center_mass)
      unit = mesh.moment_inertia / max(abs(mesh.volume), 1e-15)
      inertia_com = R @ (unit * m) @ R.T
    total_m += m
    first_moment += m * com
    inertia_origin += _inertia_about_origin(m, com, inertia_com)
  com = first_moment / total_m
  inertia_com = inertia_origin - total_m * (
    np.dot(com, com) * np.eye(3) - np.outer(com, com)
  )
  return MassProps(total_m, com, inertia_com)


# --- Visual meshes ------------------------------------------------------------


def build_visual_mesh(
  parts: list[Part], meshes: MeshSource, transform: np.ndarray, face_budget: int
) -> trimesh.Trimesh:
  visual = [p for p in parts if p.is_visual]
  total = sum(len(meshes.get(p.mesh_index).faces) for p in visual if not p.is_motor)
  ratio = min(1.0, face_budget / max(total, 1))
  pieces = []
  for part in visual:
    mesh = (
      motor_visual_mesh()
      if part.is_motor
      else meshes.decimated(part.mesh_index, ratio).copy()
    )
    mesh.apply_transform(transform @ part.world)
    pieces.append(mesh)
  merged = trimesh.util.concatenate(pieces)
  merged.merge_vertices()
  merged.remove_unreferenced_vertices()
  return merged


# --- Kinematics ---------------------------------------------------------------


def _assembly_points(asm: Assembly, meshes: MeshSource, stride: int = 7) -> np.ndarray:
  chunks = []
  for part in asm.parts:
    v = meshes.get(part.mesh_index).vertices[::stride]
    chunks.append(_apply(part.world, v))
  return np.vstack(chunks)


def _motor(asm: Assembly) -> Part:
  motors = [p for p in asm.parts if p.is_motor]
  if len(motors) != 1:
    raise ValueError(f"{asm.name} has {len(motors)} motors, expected 1")
  return motors[0]


def _nearest(point: np.ndarray, clouds: dict[str, np.ndarray]) -> str:
  return min(clouds, key=lambda k: np.linalg.norm(clouds[k] - point, axis=1).min())


@dataclass
class LegModel:
  leg: str
  hip_pos: np.ndarray
  thigh_length: float
  shank_length: float
  foot_pos: np.ndarray
  transforms: dict[str, np.ndarray]
  assemblies: dict[str, Assembly]
  cad_angles: dict[str, float]


@dataclass
class RobotModel:
  base_assembly: Assembly
  base_transform: np.ndarray
  legs: dict[str, LegModel]


def build_kinematics(glb: Glb, meshes: MeshCache) -> RobotModel:
  assemblies = read_assemblies(glb)
  by_family: dict[str, list[Assembly]] = defaultdict(list)
  for asm in assemblies:
    by_family[asm.family].append(asm)

  (body,) = by_family["Body_2"]
  hip_asms = by_family["Hiproll_1"]
  thigh_asms = by_family["Hippitch_1"]
  shank_asms = by_family["Knee_1"] + by_family["Knee_1 mirror"]
  if not (len(hip_asms) == len(thigh_asms) == len(shank_asms) == 4):
    raise ValueError("expected four instances of each leg subassembly")

  # Hip roll motors live on the body; their placement names the four legs.
  roll_motors = [p for p in body.parts if p.is_motor]
  if len(roll_motors) != 4:
    raise ValueError(f"body has {len(roll_motors)} motors, expected 4")
  mid_y = float(np.mean([m.world[1, 3] for m in roll_motors]))
  roll_by_leg = {}
  for motor in roll_motors:
    x, y = motor.world[0, 3], motor.world[1, 3]
    roll_by_leg[("F" if y > mid_y else "R") + ("L" if x < 0 else "R")] = motor
  if set(roll_by_leg) != set(LEGS):
    raise ValueError(f"could not label the four legs, got {sorted(roll_by_leg)}")

  # Walk down the chain: each child subassembly wraps its parent's motor.
  hip_by_leg = _match(
    hip_asms, {k: v.world[:3, 3] for k, v in roll_by_leg.items()}, meshes
  )
  pitch_origin = {k: _motor(v).world[:3, 3] for k, v in hip_by_leg.items()}
  thigh_by_leg = _match(thigh_asms, pitch_origin, meshes)
  knee_origin = {k: _motor(v).world[:3, 3] for k, v in thigh_by_leg.items()}
  shank_by_leg = _match(shank_asms, knee_origin, meshes)

  # Base frame: centred between the four hip joints, at hip height.  The roll
  # axis runs fore/aft so it fixes only x and z; the fore/aft position of a hip
  # comes from where its pitch axis crosses it.
  origin_cad = np.array(
    [
      float(np.mean([roll_by_leg[leg].world[0, 3] for leg in LEGS])),
      float(np.mean([pitch_origin[leg][1] for leg in LEGS])),
      float(np.mean([roll_by_leg[leg].world[2, 3] for leg in LEGS])),
    ]
  )
  G = _translate(-(CAD_TO_ROBOT[:3, :3] @ origin_cad)) @ CAD_TO_ROBOT

  legs = {}
  for leg in LEGS:
    legs[leg] = _build_leg(
      leg,
      G,
      roll_by_leg[leg],
      hip_by_leg[leg],
      thigh_by_leg[leg],
      shank_by_leg[leg],
      meshes,
    )
  return RobotModel(body, G, legs)


def _match(
  candidates: list[Assembly], origins: dict[str, np.ndarray], meshes: MeshSource
) -> dict[str, Assembly]:
  clouds = {asm.name: _assembly_points(asm, meshes) for asm in candidates}
  by_name = {asm.name: asm for asm in candidates}
  out: dict[str, Assembly] = {}
  for leg, origin in origins.items():
    name = _nearest(origin, clouds)
    out[leg] = by_name[name]
    clouds.pop(name)
  if len(out) != len(candidates):
    raise ValueError("subassembly-to-leg matching was not one-to-one")
  return out


def _signed_axis(world: np.ndarray, G: np.ndarray, reference: np.ndarray) -> np.ndarray:
  axis = G[:3, :3] @ world[:3, 2]
  return axis if float(axis @ reference) >= 0 else -axis


def _angle_about_y(vec: np.ndarray) -> float:
  """Rotation about +y taking (0,0,-1) onto the xz part of ``vec``."""
  return float(np.arctan2(-vec[0], -vec[2]))


def _build_leg(
  leg: str,
  G: np.ndarray,
  roll_motor: Part,
  hip_asm: Assembly,
  thigh_asm: Assembly,
  shank_asm: Assembly,
  meshes: MeshSource,
) -> LegModel:
  pitch_motor = _motor(hip_asm)
  knee_motor = _motor(thigh_asm)

  a_roll = _apply_point(G, roll_motor.world[:3, 3])
  a_pitch = _apply_point(G, pitch_motor.world[:3, 3])
  a_knee = _apply_point(G, knee_motor.world[:3, 3])
  pitch_axis = _signed_axis(pitch_motor.world, G, Y_AXIS)
  knee_axis = _signed_axis(knee_motor.world, G, Y_AXIS)

  # Hip roll: the CAD pose tilts the pitch axis out of the horizontal plane.
  theta_roll = float(np.arctan2(pitch_axis[2], pitch_axis[1]))
  R1 = _rot_about_line(X_AXIS, a_roll, -theta_roll)
  a_pitch = _apply_point(R1, a_pitch)
  a_knee = _apply_point(R1, a_knee)

  hip_pos = np.array([a_pitch[0], a_roll[1], a_roll[2]])
  if abs(a_pitch[2] - a_roll[2]) > 1e-4:
    raise ValueError(f"{leg}: roll and pitch axes do not intersect")

  # Hip pitch: zero puts the thigh straight down.
  thigh_vec = a_knee - a_pitch
  theta_pitch = _angle_about_y(thigh_vec)
  thigh_length = float(np.hypot(thigh_vec[0], thigh_vec[2]))
  R2 = _rot_about_line(Y_AXIS, a_pitch, -theta_pitch)
  a_knee = _apply_point(R2, a_knee)
  knee_pos = np.array([a_knee[0], hip_pos[1], a_knee[2]])

  # Knee: zero puts the shank in line with the thigh.  The foot is the point
  # of the shank assembly farthest from the knee axis.
  shank_pts = _apply(R2 @ R1 @ G, _assembly_points(shank_asm, meshes, stride=1))
  radial = shank_pts - a_knee
  radial -= np.outer(radial @ knee_axis, knee_axis)
  reach = np.linalg.norm(radial, axis=1)
  # The tip is a plate edge, so a dozen vertices tie for farthest-from-the-knee,
  # spread across the plate's thickness.  Picking one of them puts the foot on
  # whichever face the mesh happens to list first -- half a plate width off, and
  # not even the same face on every leg.  Keep the radial reach of the extreme
  # vertex but centre the tip across the tie.
  tied = shank_pts[reach > reach.max() - FOOT_TIE_TOL]
  # Take the midpoint of the tip's extent along the knee axis, not the mean:
  # the knee axis is tilted a fraction of a degree by the CAD pose, so the tie
  # catches an uneven number of vertices from the plate's two faces and a mean
  # would lean toward whichever face contributed more.
  along = tied @ knee_axis
  centre = 0.5 * (float(along.min()) + float(along.max()))
  tip = shank_pts[int(reach.argmax())]
  foot_cad = tip + (centre - tip @ knee_axis) * knee_axis
  shank_vec = foot_cad - a_knee
  theta_knee = _angle_about_y(shank_vec)
  shank_length = float(np.hypot(shank_vec[0], shank_vec[2]))
  R3 = _rot_about_line(Y_AXIS, a_knee, -theta_knee)
  foot = _apply_point(R3, foot_cad)

  return LegModel(
    leg=leg,
    hip_pos=hip_pos,
    thigh_length=thigh_length,
    shank_length=shank_length,
    foot_pos=np.array([0.0, foot[1] - hip_pos[1], -shank_length]),
    transforms={
      "hip": _translate(-hip_pos) @ R1 @ G,
      "thigh": _translate(-hip_pos) @ R2 @ R1 @ G,
      "shank": _translate(-knee_pos) @ R3 @ R2 @ R1 @ G,
    },
    assemblies={"hip": hip_asm, "thigh": thigh_asm, "shank": shank_asm},
    cad_angles={
      "hip_roll": theta_roll,
      "hip_pitch": theta_pitch,
      "knee": theta_knee,
    },
  )


# --- Collision primitives -----------------------------------------------------

# Limb capsules are fitted to the link geometry: centred on the middle of the
# link's lateral extent, with a radius covering this fraction of its vertices.
CAPSULE_COVERAGE = 90.0


def _box_from_points(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  lo, hi = points.min(0), points.max(0)
  return (lo + hi) / 2.0, (hi - lo) / 2.0


def _capsule_from_points(points: np.ndarray, length: float) -> tuple[float, float]:
  """Fit a z-aligned capsule to a limb, returning ``(y_centre, radius)``."""
  span = points[(points[:, 2] < 0.0) & (points[:, 2] > -length)]
  y = float((span[:, 1].min() + span[:, 1].max()) / 2.0)
  radial = np.hypot(span[:, 0], span[:, 1] - y)
  return y, float(np.percentile(radial, CAPSULE_COVERAGE))


def foot_pad_part(lm: LegModel, meshes: MeshSource) -> Part:
  """The shank part that actually reaches the ground.

  Identified by geometry rather than by name: it is the part owning the vertex
  farthest from the knee axis, i.e. the one that defines ``foot_pos``.
  """
  best, best_reach = None, -1.0
  for part in lm.assemblies["shank"].parts:
    pts = _apply(
      lm.transforms["shank"] @ part.world, meshes.get(part.mesh_index).vertices
    )
    reach = float(np.hypot(pts[:, 0], pts[:, 2]).max())
    if reach > best_reach:
      best, best_reach = part, reach
  assert best is not None
  return best


def build_foot_pad_mesh(lm: LegModel, meshes: MeshSource) -> trimesh.Trimesh:
  """Convex hull of the foot pad, in the foot body's frame.

  MuJoCo collides meshes by their convex hull anyway, so taking it here keeps
  the exported file small and makes the model state plainly what is simulated.
  """
  part = foot_pad_part(lm, meshes)
  mesh = meshes.get(part.mesh_index).copy()
  mesh.apply_transform(lm.transforms["shank"] @ part.world)
  mesh.apply_translation(-lm.foot_pos)
  return mesh.convex_hull


# --- MJCF emission ------------------------------------------------------------


def _fmt(values, digits: int = 6) -> str:
  """Format numbers for MJCF, snapping floating-point noise to zero."""
  cleaned = (0.0 if abs(float(v)) < 1e-9 else float(v) for v in np.atleast_1d(values))
  return " ".join(f"{v:.{digits}g}" for v in cleaned)


def _fmt_pos(values) -> str:
  """Format a position or size, rounded to micrometres."""
  return _fmt(np.round(np.atleast_1d(values).astype(float), 6))


def _inertial(props: MassProps, indent: str) -> str:
  eig = np.linalg.eigvalsh(props.inertia)
  if eig.min() <= 0:
    raise ValueError(f"non positive-definite inertia: {eig}")
  a, b, c = sorted(eig)
  if a + b < c * (1 - 1e-9):
    raise ValueError(f"inertia violates the triangle inequality: {eig}")
  i = props.inertia
  full = (i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2])
  return (
    f'{indent}<inertial pos="{_fmt(props.com)}" mass="{props.mass:.6g}"\n'
    f'{indent}  fullinertia="{_fmt(full)}"/>'
  )


def build_mjcf(
  model: RobotModel,
  meshes: MeshSource,
  mesh_dir: Path,
  urdf_masses: dict[str, list[UrdfPart]] | None = None,
) -> tuple[str, float, dict[str, int]]:
  mesh_dir.mkdir(parents=True, exist_ok=True)

  def export(name: str, asm: Assembly, transform: np.ndarray, budget: int):
    mesh = build_visual_mesh(asm.parts, meshes, transform, budget)
    mesh.export(mesh_dir / f"{name}.obj", file_type="obj", include_normals=False)
    return len(mesh.faces)

  faces_written = {}
  base_props = add_payload(
    mass_properties(
      model.base_assembly.parts, meshes, model.base_transform, urdf_masses
    )
  )
  faces_written["base_link"] = export(
    "base_link",
    model.base_assembly,
    model.base_transform,
    VISUAL_FACE_BUDGET["base"],
  )
  base_pts = _apply(
    model.base_transform, _assembly_points(model.base_assembly, meshes, stride=3)
  )
  base_box_pos, base_box_size = _box_from_points(base_pts)

  bodies: list[str] = []
  actuators: list[str] = []
  assets: list[str] = ['    <mesh name="base_link" file="base_link.obj"/>']
  total_mass = base_props.mass

  for leg in LEGS:
    lm = model.legs[leg]
    props = {}
    for part in ("hip", "thigh", "shank"):
      asm = lm.assemblies[part]
      transform = lm.transforms[part]
      props[part] = mass_properties(asm.parts, meshes, transform, urdf_masses)
      total_mass += props[part].mass
      name = f"{leg}_{part}"
      faces_written[name] = export(name, asm, transform, VISUAL_FACE_BUDGET[part])
      assets.append(f'    <mesh name="{name}" file="{name}.obj"/>')

    pad = build_foot_pad_mesh(lm, meshes)
    pad_name = f"{leg}_foot_pad"
    pad.export(mesh_dir / f"{pad_name}.obj", file_type="obj", include_normals=False)
    faces_written[pad_name] = len(pad.faces)
    assets.append(f'    <mesh name="{pad_name}" file="{pad_name}.obj"/>')

    hip_pts = _apply(
      lm.transforms["hip"],
      _assembly_points(lm.assemblies["hip"], meshes, stride=3),
    )
    hip_box_pos, hip_box_size = _box_from_points(hip_pts)
    thigh_y, thigh_r = _capsule_from_points(
      _apply(
        lm.transforms["thigh"],
        _assembly_points(lm.assemblies["thigh"], meshes, stride=3),
      ),
      lm.thigh_length,
    )
    shank_y, shank_r = _capsule_from_points(
      _apply(
        lm.transforms["shank"],
        _assembly_points(lm.assemblies["shank"], meshes, stride=3),
      ),
      lm.shank_length,
    )
    # End the shank capsule where the foot pad begins, so the pad is the only
    # geom that can touch the ground.  ``pad`` sits in the foot frame, whose
    # origin is the tip, so its top is that far above the tip.
    pad_top_in_shank = -(lm.shank_length - float(pad.vertices[:, 2].max()))
    shank_end = f"{pad_top_in_shank + shank_r:.6g}"
    roll_range = _fmt(JOINT_RANGES["hip_roll"])
    pitch_range = _fmt(JOINT_RANGES["hip_pitch"])
    knee_range = _fmt(JOINT_RANGES["knee"])
    bodies.append(f"""
      <body name="{leg}_hip" pos="{_fmt_pos(lm.hip_pos)}">
        <joint name="{leg}_hip_roll" axis="1 0 0" range="{roll_range}"/>
{_inertial(props["hip"], " " * 8)}
        <geom class="visual" mesh="{leg}_hip"/>
        <geom name="{leg}_hip_collision" class="collision" type="box"
          pos="{_fmt_pos(hip_box_pos)}" size="{_fmt_pos(hip_box_size)}"/>
        <body name="{leg}_thigh" pos="0 0 0">
          <joint name="{leg}_hip_pitch" axis="0 1 0" range="{pitch_range}"/>
{_inertial(props["thigh"], " " * 10)}
          <geom class="visual" mesh="{leg}_thigh"/>
          <geom name="{leg}_thigh_collision" class="collision" type="capsule"
            fromto="0 {thigh_y:.6g} 0 0 {thigh_y:.6g} {-lm.thigh_length:.6g}"
            size="{thigh_r:.4g}"/>
          <body name="{leg}_shank" pos="0 0 {-lm.thigh_length:.6g}">
            <joint name="{leg}_knee" axis="0 1 0" range="{knee_range}"/>
{_inertial(props["shank"], " " * 12)}
            <geom class="visual" mesh="{leg}_shank"/>
            <geom name="{leg}_shank_collision" class="collision" type="capsule"
              fromto="0 {shank_y:.6g} 0 0 {shank_y:.6g} {shank_end}"
              size="{shank_r:.4g}"/>
            <body name="{leg}_foot" pos="{_fmt_pos(lm.foot_pos)}">
              <inertial pos="0 0 0" mass="1e-6" diaginertia="1e-9 1e-9 1e-9"/>
              <geom name="{leg}_foot_collision" class="foot" type="mesh"
                mesh="{leg}_foot_pad"/>
              <!-- Anchors the <touch> sensor below.  Drawn transparent: the
                   viewer renders sites as solid spheres, which put a ball on
                   the toe that reads as geometry the robot does not have. -->
              <site name="{leg}_foot_site" size="0.012" rgba="1 1 1 0"/>
            </body>
          </body>
        </body>
      </body>""")

    for kind in ("hip_roll", "hip_pitch", "knee"):
      lo, hi = JOINT_RANGES[kind]
      actuators.append(
        f'    <position class="pochi" name="{leg}_{kind}_act" '
        f'joint="{leg}_{kind}" ctrlrange="{lo:g} {hi:g}"/>'
      )

  torque = f"{-RS02_PEAK_TORQUE_NM:g} {RS02_PEAK_TORQUE_NM:g}"
  # The thigh capsule over-approximates a narrow bracket and clips the torso
  # box near the hip; the real parts cannot touch, so filter the pair.
  excludes = "\n".join(
    f'    <exclude body1="base_link" body2="{leg}_thigh"/>' for leg in LEGS
  )
  sensors = "\n".join(
    f'    <touch name="{leg}_foot_touch" site="{leg}_foot_site"/>' for leg in LEGS
  )
  xml = f"""<mujoco model="pochi">
  <!-- Generated by scripts/glb_to_mjcf.py from the Onshape CAD export.
       Do not edit by hand; change the converter and regenerate.
       Source: {meshes.provenance}
       Total mass {total_mass:.3f} kg (RS02 taken as {RS02_MASS_KG} kg each,
       plus a {PAYLOAD_MASS_KG} kg ballast block on the base link).
       The <position> actuators below describe the hardware and drive the
       standalone model; mjlab deletes them in pochi_rl.mjlab.entity_cfg and
       substitutes <motor> actuators with the RS02 torque-speed curve. -->
  <compiler angle="radian" autolimits="true" meshdir="meshes"/>
  <option timestep="0.005" iterations="50" solver="Newton" cone="elliptic"/>

  <visual>
    <global offwidth="1920" offheight="1080"/>
    <quality shadowsize="4096"/>
  </visual>

  <default>
    <default class="pochi">
      <joint armature="{RS02_ARMATURE}" damping="0.2"/>
      <geom condim="3" friction="0.8 0.02 0.001"/>
      <position kp="60" kv="1.5" forcerange="{torque}"/>
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2" rgba="0.35 0.37 0.4 1"/>
      </default>
      <default class="collision">
        <geom group="3" rgba="0.9 0.5 0.1 0.3"/>
      </default>
      <default class="foot">
        <geom group="3" priority="1" rgba="0.08 0.08 0.08 1"/>
      </default>
    </default>
  </default>

  <asset>
{chr(10).join(assets)}
  </asset>

  <worldbody>
    <body name="base_link" pos="0 0 {STAND_HEIGHT:.4g}" childclass="pochi">
      <freejoint name="floating_base_joint"/>
{_inertial(base_props, " " * 6)}
      <geom class="visual" mesh="base_link"/>
      <geom name="base_collision" class="collision" type="box"
        pos="{_fmt_pos(base_box_pos)}" size="{_fmt_pos(base_box_size)}"/>
      <site name="imu" pos="0 0 0" size="0.01" rgba="0 0.4 1 1"/>
{"".join(bodies)}
    </body>
  </worldbody>

  <contact>
{excludes}
  </contact>

  <actuator>
{chr(10).join(actuators)}
  </actuator>

  <sensor>
    <framequat name="imu_quat" objtype="site" objname="imu"/>
    <gyro name="imu_gyro" site="imu"/>
    <accelerometer name="imu_accel" site="imu"/>
    <velocimeter name="imu_vel" site="imu"/>
{sensors}
  </sensor>
</mujoco>
"""
  return xml, total_mass, faces_written
