"""Regenerate assets/pochi/pochi.xml and its meshes from the CAD GLB export."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pochi_rl.cad import convert
from pochi_rl.cad.glb import Glb

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GLB = REPO_ROOT / "4leg_assem2.glb"
# onshape-to-robot URDF export of the same assembly.  Its kinematics are
# unusable (every untagged Onshape mate became a joint), but it carries the
# per-part masses Onshape computed from the real materials, which the GLB has
# no way to express.  Optional: without it the converter falls back to guessing
# a density from each part name.
DEFAULT_URDF = REPO_ROOT / "pkg_4leg_assem2" / "urdf" / "pkg_4leg_assem2.urdf"
DEFAULT_OUT = Path(__file__).resolve().parents[1] / "assets" / "pochi"


def main() -> None:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--glb", type=Path, default=DEFAULT_GLB)
  parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
  parser.add_argument("--rs02-mass", type=float, default=convert.RS02_MASS_KG)
  parser.add_argument(
    "--urdf",
    type=Path,
    default=DEFAULT_URDF,
    help="onshape-to-robot URDF to take part masses from",
  )
  parser.add_argument(
    "--no-urdf-masses",
    action="store_true",
    help="ignore the URDF and guess densities from part names",
  )
  parser.add_argument(
    "--source",
    choices=("urdf", "glb"),
    default="urdf",
    help="where the kinematics come from (default: the URDF's dof_ mates)",
  )
  parser.add_argument("--report", action="store_true", help="print CAD diagnostics")
  args = parser.parse_args()

  convert.RS02_MASS_KG = args.rs02_mass

  urdf_masses = None
  if not args.no_urdf_masses and args.urdf.is_file():
    urdf_masses = convert.load_urdf_part_masses(args.urdf)
    print(f"part masses from {args.urdf} ({len(urdf_masses)} names)")
  else:
    print("no URDF: falling back to density-by-part-name")

  if args.source == "urdf":
    if not args.urdf.is_file():
      parser.error(f"--source urdf needs {args.urdf}")
    from pochi_rl.cad import urdf as urdf_reader

    model, meshes = urdf_reader.load(args.urdf)
    print(f"kinematics from {args.urdf} (dof_ mates)")
  else:
    glb = Glb(args.glb)
    meshes = convert.MeshCache(glb)
    model = convert.build_kinematics(glb, meshes)
    print(f"kinematics from {args.glb} (inferred from motor placement)")

  xml, total_mass, faces = convert.build_mjcf(
    model, meshes, args.out / "meshes", urdf_masses
  )
  (args.out / "pochi.xml").write_text(xml)

  print(f"wrote {args.out / 'pochi.xml'}  ({total_mass:.3f} kg total)")
  print(f"wrote {len(faces)} meshes, {sum(faces.values())} triangles")
  if args.report:
    for leg in model.legs.values():
      angles = {k: round(float(np.degrees(v)), 2) for k, v in leg.cad_angles.items()}
      print(
        f"  {leg.leg}: hip={np.round(leg.hip_pos, 4)} "
        f"thigh={leg.thigh_length:.4f} shank={leg.shank_length:.4f} "
        f"foot={np.round(leg.foot_pos, 4)} cad_pose_deg={angles}"
      )


if __name__ == "__main__":
  main()
