"""Minimal glTF-binary (.glb) reader.

Only what the CAD conversion needs: the node hierarchy with world transforms
and per-node triangle geometry. No external glTF dependency.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Iterator
from pathlib import Path

import numpy as np

_COMPONENT_TYPE = {
  5120: np.int8,
  5121: np.uint8,
  5122: np.int16,
  5123: np.uint16,
  5125: np.uint32,
  5126: np.float32,
}
_NUM_COMPONENTS = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

_CHUNK_JSON = 0x4E4F534A
_CHUNK_BIN = 0x004E4942
_MAGIC = 0x46546C67


class Glb:
  """A parsed .glb file."""

  def __init__(self, path: str | Path) -> None:
    self.path = Path(path)
    # Recorded so the generated MJCF can name the CAD export it came from; the
    # GLB is a binary that gets overwritten in place on every Onshape re-export.
    self.digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
    with open(path, "rb") as fh:
      magic, _version, total = struct.unpack("<III", fh.read(12))
      if magic != _MAGIC:
        raise ValueError(f"{path} is not a glTF binary file")
      gltf: dict | None = None
      binary = b""
      while fh.tell() < total:
        length, kind = struct.unpack("<II", fh.read(8))
        payload = fh.read(length)
        if kind == _CHUNK_JSON:
          gltf = json.loads(payload)
        elif kind == _CHUNK_BIN:
          binary = payload
    if gltf is None:
      raise ValueError(f"{path} has no JSON chunk")
    self.gltf = gltf
    self.binary = binary
    self.nodes: list[dict] = gltf["nodes"]

  def accessor(self, index: int) -> np.ndarray:
    acc = self.gltf["accessors"][index]
    view = self.gltf["bufferViews"][acc["bufferView"]]
    dtype = np.dtype(_COMPONENT_TYPE[acc["componentType"]]).newbyteorder("<")
    ncomp = _NUM_COMPONENTS[acc["type"]]
    offset = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
    stride = view.get("byteStride")
    if stride and stride != ncomp * dtype.itemsize:
      raw = np.frombuffer(
        self.binary, np.uint8, count=acc["count"] * stride, offset=offset
      )
      raw = raw.reshape(acc["count"], stride)[:, : ncomp * dtype.itemsize].copy()
      return raw.view(dtype).reshape(acc["count"], ncomp)
    flat = np.frombuffer(self.binary, dtype, count=acc["count"] * ncomp, offset=offset)
    return flat.reshape(acc["count"], ncomp)

  @staticmethod
  def local_transform(node: dict) -> np.ndarray:
    if "matrix" in node:
      return np.array(node["matrix"], np.float64).reshape(4, 4).T
    mat = np.eye(4)
    if "rotation" in node:
      x, y, z, w = node["rotation"]
      mat[:3, :3] = np.array(
        [
          [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
          [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
          [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
      )
    if "scale" in node:
      mat[:3, :3] = mat[:3, :3] @ np.diag(node["scale"])
    if "translation" in node:
      mat[:3, 3] = node["translation"]
    return mat

  def walk(
    self, index: int, parent: np.ndarray | None = None
  ) -> Iterator[tuple[int, dict, np.ndarray]]:
    """Yield ``(node_index, node, world_transform)`` over a subtree."""
    node = self.nodes[index]
    world = self.local_transform(node)
    if parent is not None:
      world = parent @ world
    yield index, node, world
    for child in node.get("children", []):
      yield from self.walk(child, world)

  def mesh_geometry(self, mesh_index: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(vertices, faces)`` of a mesh in its own local frame."""
    verts: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    offset = 0
    for prim in self.gltf["meshes"][mesh_index]["primitives"]:
      v = self.accessor(prim["attributes"]["POSITION"]).astype(np.float64)
      f = self.accessor(prim["indices"]).astype(np.int64).reshape(-1, 3)
      verts.append(v)
      faces.append(f + offset)
      offset += len(v)
    return np.vstack(verts), np.vstack(faces)
