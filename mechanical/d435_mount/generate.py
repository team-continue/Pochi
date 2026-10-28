"""D435 portrait bracket. Millimetres; +X forward, +Y left, +Z up.

Run with pochi_rl/.venv/bin/python mechanical/d435_mount/generate.py.
Uses the existing trimesh/manifold3d environment; never modifies robot CAD.
"""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import zipfile

import manifold3d as mf
import numpy as np
import trimesh

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MESHES = ROOT / "pochi_rl/pkg_4leg_assem2/meshes"
BOARD = MESHES / "carbon_Body_plate3_x__3mm_x2_.stl"
ANGLES = (25, 30, 35, 40, 45)
BOARD_HOLES = [(y, z) for y in (-15.0, 15.0) for z in (-24.0, 24.0)]
CAMERA_HOLES = (-22.5, 22.5)
REVISION = 4
WIDTH = 50.0
INSERT = {
    "model": "ruthex RX-M3x5.7", "thread": "M3",
    "outer_diameter_mm": 4.6, "length_mm": 5.7,
    "printed_bore_diameter_mm": 4.0, "bore_depth_mm": 6.7,
    "minimum_radial_wall_mm": 1.6,
    "source": "https://www.igo3d.com/mediafiles/Sonstiges/Ruthex/ruthex_Datenblatt_RX-Serie.pdf",
}
CAMERA_DATASHEET = (
    "https://www.realsenseai.com/wp-content/uploads/2020/06/"
    "Intel-RealSense-D400-Series-Datasheet-June-2020.pdf#page=113"
)


def box(size, centre):
    return mf.Manifold.cube(size, center=True).translate(centre)


def cylinder_x(radius, length, centre, segments=96):
    return mf.Manifold.cylinder(
        length, radius, circular_segments=segments, center=True
    ).rotate((0, 90, 0)).translate(centre)


def rounded_plate(depth, width, height, radius=3):
    """Rounded YZ rectangle extruded along X, centred at the origin."""
    solids = [
        box((depth, width - 2 * radius, height), (0, 0, 0)),
        box((depth, width, height - 2 * radius), (0, 0, 0)),
    ]
    for y in (-width / 2 + radius, width / 2 - radius):
        for z in (-height / 2 + radius, height / 2 - radius):
            solids.append(cylinder_x(radius, depth, (0, y, z)))
    return mf.Manifold.batch_boolean(solids, mf.OpType.Add)


def camera_frame(angle):
    a = np.deg2rad(angle)
    normal = np.array([np.cos(a), 0.0, -np.sin(a)])
    vertical = np.array([np.sin(a), 0.0, np.cos(a)])
    # Maintain clearance even for the 90 mm camera at 45 degrees.
    centre = np.array([12.0 + 45.0 * np.sin(a), 0.0, 2.0])
    return normal, vertical, centre


def camera_transform(solid, angle):
    return solid.rotate((0, angle, 0)).translate(camera_frame(angle)[2])


def teardrop_x(radius, length, centre):
    """Horizontal bore with a 60 degree roof, printed with original +Y up."""
    tangent_y = radius / 2
    tangent_z = radius * np.sqrt(3) / 2
    points = []
    for x in (-length / 2, length / 2):
        points.extend([[x, tangent_y, -tangent_z], [x, tangent_y, tangent_z], [x, radius * 2, 0]])
    roof = mf.Manifold.hull_points(points).translate(centre)
    return cylinder_x(radius, length, centre) + roof


def make_bracket(angle):
    normal, vertical, centre = camera_frame(angle)
    # A continuous wedge rather than two thin webs and a suspended carrier.
    # Its full XZ profile is on the bed; there is no roof over a hollow wedge.
    points = []
    for y in (-WIDTH / 2, WIDTH / 2):
        for x in (0, 8):
            for z in (-32, 32):
                points.append([x, y, z])
        for s in (-37, 37):
            for d in (-8, 0):
                p = centre + vertical * s + normal * d
                points.append([p[0], y, p[2]])
    bracket = mf.Manifold.hull_points(points)
    cuts = []
    for y, z in BOARD_HOLES:
        depth = INSERT["bore_depth_mm"]
        # Full round insert seats are printed directly; no pilot/drill step.
        cuts.append(cylinder_x(INSERT["printed_bore_diameter_mm"] / 2,
                               depth + 0.02, (depth / 2 - 0.01, y, z)))
    for z in CAMERA_HOLES:
        # An M3x8 screw bears on a full 6 mm pad and enters the camera 2 mm.
        cuts.append(camera_transform(teardrop_x(1.7, 10, (-3, 0, z)), angle))
        # Rear access for an Allen key/head. Camera is fixed before the board.
        cuts.append(camera_transform(teardrop_x(3.5, 144, (-78, 0, z)), angle))
    return mf.Manifold.batch_boolean([bracket, *cuts], mf.OpType.Subtract)


def print_mesh(mesh):
    printed = mesh.copy()
    # +90, not -90: original +Y becomes printer +Z. The teardrop roof points up.
    printed.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 2, (1, 0, 0)))
    printed.apply_translation(-printed.bounds[0])
    return printed


def printability(printed):
    from shapely.geometry import MultiPoint
    downward = (printed.face_normals[:, 2] < -np.cos(np.pi / 4) - 1e-4)
    above_bed = printed.triangles_center[:, 2] > 0.02
    ids = np.flatnonzero(downward & above_bed & (printed.area_faces > 1e-5))
    edges = printed.face_adjacency
    edges = edges[np.isin(edges[:, 0], ids) & np.isin(edges[:, 1], ids)]
    spans = []
    if len(ids):
        groups = trimesh.graph.connected_components(edges, nodes=ids, min_len=1)
        for group in groups:
            vertices = printed.triangles[group].reshape(-1, 3)[:, :2]
            rectangle = MultiPoint(vertices).minimum_rotated_rectangle
            if rectangle.geom_type != "Polygon":
                spans.append(0.0)
                continue
            points = np.array(rectangle.exterior.coords)[:4]
            spans.append(float(np.min(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1))))
    # The circular phi4 bores have short roofs; keep their span bounded.
    assert max(spans, default=0) <= 3.0, ("Unexpected unsupported roof", spans)
    section = printed.section(plane_origin=(0, 0, 0.1), plane_normal=(0, 0, 1))
    planar, _ = section.to_2D()
    assert len(planar.polygons_full) == 1, "First layer contains a disconnected island"
    assert planar.area > 1200
    return {
        "first_layer_area_mm2": round(float(planar.area), 3),
        "first_layer_connected_components": 1,
        "maximum_short_bridge_span_mm": round(max(spans, default=0), 3),
        "overhang_roof_angle_deg": 60,
        "round_insert_hole_roofs_present": True,
        "slicer_verified": False,
    }


def mesh_of(solid):
    data = solid.to_mesh()
    return trimesh.Trimesh(
        vertices=np.asarray(data.vert_properties)[:, :3],
        faces=np.asarray(data.tri_verts), process=True,
    )


def board_mesh():
    mesh = trimesh.load_mesh(BOARD)
    mesh.apply_scale(1000)
    # CAD: lateral X, rearward -Y. Viewer/part: forward X, left Y.
    mesh.vertices = np.column_stack((mesh.vertices[:, 1], -mesh.vertices[:, 0], mesh.vertices[:, 2]))
    mesh.invert() if mesh.volume < 0 else None
    return mesh


def extract_holes(mesh):
    section = mesh.section(plane_origin=(-1.5, 0, 0), plane_normal=(1, 0, 0))
    holes = []
    for path in section.discrete:
        p = path[:, 1:3]
        fit = np.column_stack((2 * p[:, 0], 2 * p[:, 1], np.ones(len(p))))
        cy, cz, c = np.linalg.lstsq(fit, (p * p).sum(axis=1), rcond=None)[0]
        r = np.sqrt(c + cy**2 + cz**2)
        err = np.max(np.abs(np.linalg.norm(p - [cy, cz], axis=1) - r))
        if err < 0.03 and r < 8:
            holes.append({"y": float(cy), "z": float(cz), "diameter": float(2 * r), "fit_error": float(err)})
    return sorted(holes, key=lambda h: (round(h["z"], 1), h["y"]))


def validate(solid, mesh, holes, angle):
    assert solid.status() == mf.Error.NoError
    assert mesh.is_watertight and mesh.is_winding_consistent and mesh.is_volume
    assert len(mesh.split()) == 1, "Bracket must be one connected printable solid"
    errors = []
    for y, z in BOARD_HOLES:
        error = min(np.linalg.norm([h["y"] - y, h["z"] - z]) for h in holes if 3.2 < h["diameter"] < 3.4)
        assert error < 0.02, "A mounting hole does not match the source board"
        errors.append(float(error))
        assert (solid ^ cylinder_x(1.99, 6.6, (3.3, y, z))).volume() < 0.001
        for yy in (y - 4.5, y + 4.5):
            shoulder = box((7.8, 1, 2), (4, yy, z))
            assert (shoulder - solid).volume() < 0.01
    normal, vertical, centre = camera_frame(angle)
    for z in CAMERA_HOLES:
        bore = camera_transform(cylinder_x(1.5, 9, (-3, 0, z)), angle)
        assert (solid ^ bore).volume() < 0.001
        sleeve = cylinder_x(3.3, 5.8, (-3, 0, z)) - teardrop_x(1.9, 6, (-3, 0, z))
        sleeve = camera_transform(sleeve, angle)
        assert (sleeve - solid).volume() < 0.01
    assert mesh.bounds[0, 0] > -0.0001, "Printed bracket must remain forward of the board"
    # Front roll motors occupy |Y| >= 30.75 mm in the source assembly.
    clearance = 30.75 - np.max(np.abs(mesh.vertices[:, 1]))
    assert clearance > 5.7
    assert np.allclose(normal @ vertical, 0)
    assert np.isclose(np.linalg.norm(vertical * 45), 45)
    for y, z in BOARD_HOLES:
        radius = INSERT["outer_diameter_mm"] / 2
        wall = cylinder_x(radius + INSERT["minimum_radial_wall_mm"], 5.6, (2.85, y, z))
        wall -= cylinder_x(radius + 0.01, 5.8, (2.85, y, z))
        assert (wall - solid).volume() < 0.01, "Insufficient material around insert"
        assert (solid ^ cylinder_x(1.5, 4.5, (2.25, y, z))).volume() < 0.001
        bottom = box((1.0, 2, 2), (7.3, y, z))
        assert (bottom - solid).volume() < 0.01
    camera_roi = camera_transform(box((6, 20, 74), (-3, 0, 0)), angle)
    camera_delta = None
    v2_archive = HERE / "archive_v2/pochi_d435_mount_v2.zip"
    if v2_archive.exists():
        with zipfile.ZipFile(v2_archive) as z:
            previous = trimesh.load_mesh(io.BytesIO(z.read(f"exports/d435_portrait_down{angle}_v2_assembly_mm.stl")), file_type="stl")
        previous_solid = mf.Manifold(mf.Mesh(np.asarray(previous.vertices, np.float32), np.asarray(previous.faces, np.uint32)))
        difference = (solid - previous_solid) + (previous_solid - solid)
        camera_delta = float((difference ^ camera_roi).volume())
        assert camera_delta < 0.05, "RealSense fixing changed from v2"
    camera = camera_transform(
        rounded_plate(25.05, 25, 90, 12.49).translate((12.525, 0, 0)), angle
    )
    overlap = float((solid ^ camera).volume())
    assert overlap < 0.001, "Printed bracket intrudes into the camera envelope"
    section = mesh.section(plane_origin=(8, 0, 0), plane_normal=(1, 0, 0))
    planar, _ = section.to_2D()
    old_area = None
    archive = HERE / "archive_v1/pochi_d435_mount_v1.zip"
    if archive.exists():
        with zipfile.ZipFile(archive) as z:
            old = trimesh.load_mesh(io.BytesIO(z.read(f"exports/d435_portrait_down{angle}_assembly_mm.stl")), file_type="stl")
            old_planar, _ = old.section(plane_origin=(8, 0, 0), plane_normal=(1, 0, 0)).to_2D()
            old_area = float(old_planar.area)
    return {
        "angle_deg": angle, "watertight": True, "connected_components": 1,
        "bounds_mm": mesh.bounds.round(4).tolist(), "volume_cm3": round(mesh.volume / 1000, 3),
        "board_hole_centre_max_error_mm": max(errors),
        "motor_lateral_clearance_mm": round(float(clearance), 3),
        "camera_back_centre_mm": centre.round(5).tolist(),
        "optical_direction": normal.round(6).tolist(),
        "camera_envelope_overlap_mm3": overlap,
        "camera_side_unchanged_from_revision": 2,
        "camera_side_difference_volume_mm3": camera_delta,
        "insert_bore_printed_at_full_size": True,
        "root_section_at_x8_area_mm2": round(float(planar.area), 3),
        "old_root_section_at_x8_area_mm2": round(old_area, 3) if old_area else None,
        "root_section_area_ratio": round(float(planar.area) / old_area, 3) if old_area else None,
        "physical_fit_verified": False,
    }


def main():
    out = HERE / "exports"
    out.mkdir(exist_ok=True)
    board = board_mesh()
    holes = extract_holes(board)
    (out / "board_holes.json").write_text(json.dumps(holes, indent=2) + "\n")
    board.export(out / "reference_front_board_mm.stl")
    reports = []
    for angle in ANGLES:
        solid = make_bracket(angle)
        mesh = mesh_of(solid)
        report = validate(solid, mesh, holes, angle)
        name = f"d435_portrait_down{angle}_v{REVISION}"
        mesh.export(out / f"{name}_assembly_mm.stl")
        printed = print_mesh(mesh)
        assert printed.is_volume and np.min(printed.vertices[:, 2]) >= -1e-5
        printed.export(out / f"{name}_print_mm.stl")
        report["print_dimensions_mm"] = printed.extents.round(3).tolist()
        report["printability"] = printability(printed)
        checks_path = out / "print_checks.json"
        if checks_path.exists():
            checks = json.loads(checks_path.read_text())
            record = next((r for r in checks.get("variants", []) if r["angle_deg"] == angle), None)
            digest = hashlib.sha256((out / f"{name}_print_mm.stl").read_bytes()).hexdigest()
            if record and record["stl_sha256"] == digest:
                report["printability"]["slicer_verified"] = True
                report["printability"]["slicer"] = checks["slicer"]
                report["printability"]["layer_count"] = record["layers"]
                report["printability"]["generated_support_toolpaths"] = record["support_toolpaths"]
        reports.append(report)
    manifest = {
        "units": "mm", "revision": REVISION, "default_angle_deg": 30,
        "construction": "continuous wedge; full-size round heat-set insert bores printed directly",
        "board_source": str(BOARD.relative_to(ROOT)),
        "board_sha256": hashlib.sha256(BOARD.read_bytes()).hexdigest(),
        "camera_source": CAMERA_DATASHEET,
        "board_fixing_centres_yz_mm": BOARD_HOLES,
        "camera_mount_pitch_mm": 45, "camera_mount_max_insertion_mm": 3,
        "base_thickness_mm": 8, "camera_pad_thickness_mm": 6,
        "board_screws": "4 x M3x8 inserted from board rear; 4 x M3 heat-set inserts",
        "board_insert": INSERT,
        "board_screw_engagement_mm": 4.5,
        "insert_drilling_required": False,
        "camera_screws": "2 x M3x8 (2 mm nominal insertion without washer)",
        "variants": reports,
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    archive = out / "pochi_d435_mount.zip"
    temporary_archive = out / "pochi_d435_mount.zip.tmp"
    with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for p in [HERE / "generate.py", HERE / "viewer.py", HERE / "verify_print.py", HERE / "mount.scad", HERE / "README.md", *out.glob(f"*_v{REVISION}_*.stl"), out / "reference_front_board_mm.stl", *out.glob("*.json")]:
            if p.is_file():
                z.write(p, p.relative_to(HERE))
    temporary_archive.replace(archive)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
