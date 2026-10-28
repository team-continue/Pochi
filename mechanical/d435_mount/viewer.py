"""Local Viser GUI for the generated D435 bracket; no robot control access."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import threading

import numpy as np
import trimesh
import viser

from generate import ANGLES, BOARD_HOLES, HERE, ROOT, REVISION, INSERT, camera_frame, camera_transform, cylinder_x, mesh_of, rounded_plate


def load_context():
    from pochi_rl.cad import urdf
    model, source = urdf.load(ROOT / "pochi_rl/pkg_4leg_assem2/urdf/pkg_4leg_assem2.urdf")
    items = []
    for part in model.base_assembly.parts:
        if not (part.name.startswith("carbon_Body_plate3") or part.is_motor):
            continue
        transform = model.base_transform @ part.world
        mesh = source.get(part.mesh_index).copy()
        mesh.apply_transform(transform)
        if mesh.centroid[0] < 0:
            continue
        mesh.apply_scale(1000)
        mesh.apply_translation((-153, 0, 0))
        items.append((part.name, mesh))
    return items


def bolt(radius, length, centre):
    return mesh_of(cylinder_x(radius, length, centre))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8790)
    args = parser.parse_args()
    manifest = json.loads((HERE / "exports/manifest.json").read_text())
    reports = {row["angle_deg"]: row for row in manifest["variants"]}
    meshes = {angle: trimesh.load_mesh(HERE / f"exports/d435_portrait_down{angle}_v{REVISION}_assembly_mm.stl") for angle in ANGLES}
    contexts = load_context()
    server = viser.ViserServer(host=args.host, port=args.port, label="Pochi · D435 縦置きマウント")
    server.scene.set_up_direction("+z")
    server.gui.configure_theme(titlebar_content=None, control_layout="collapsible", dark_mode=True)
    server.gui.add_markdown(f"# D435 マウント v{REVISION}\nボード側はM3インサート式。RealSense固定はv2と同じです。")
    angle = server.gui.add_slider("下向き角度（°）", min=25, max=45, step=5, initial_value=30)
    mode = server.gui.add_dropdown("表示", ("取り付け状態", "パーツ単体", "印刷姿勢"), initial_value="取り付け状態")
    show_camera = server.gui.add_checkbox("D435を表示", True)
    show_board = server.gui.add_checkbox("板とモーターを表示", True)
    show_bolts = server.gui.add_checkbox("ねじを表示", True)
    show_labels = server.gui.add_checkbox("固定穴・光軸の説明", False)
    dimensions = server.gui.add_markdown("")
    download = server.gui.add_button("印刷用STLをダウンロード", color="cyan", icon=viser.Icon.DOWNLOAD)
    archive = server.gui.add_button("全STL・編集用CAD・説明書 ZIP", icon=viser.Icon.DOWNLOAD)
    with server.gui.add_folder("視点"):
        iso = server.gui.add_button("斜めから")
        front = server.gui.add_button("正面から")
        side = server.gui.add_button("横から（傾きを確認）")
        rear = server.gui.add_button("ボード接触面（インサート）")
    server.gui.add_markdown(
        "**ボード固定**\n\nM3熱圧入インサート4個：外径4.6 × 長さ5.7 mmを基準。"
        "下穴は印刷時からφ4.0 × 深さ6.7 mm。バリを除去してボード接触面から熱圧入します。"
        "板の裏からM3×8ねじ4本と座金で固定。\n\n"
        "カメラはM3×8 2本。厚さ6 mmのねじ受け、挿入量2 mm。"
        "カメラを先に固定し、最後にボードへ取り付けます。\n\n"
        "**印刷**\n\n0.4 mmノズル、0.2 mm積層、外周6本、上下各6層、充填50%を初期条件に。"
        "印刷姿勢のまま、サポートOFFでスライス確認。インサート穴はφ4の丸穴です。実印刷は未検証。\n\n"
        "**現物合わせ未確認**：実物の空き4穴・内部配線・可動部との隙間を確認してください。"
        "カメラは外形の模式図です。\n\n"
        "[カメラ寸法の出典：メーカー図面 p.113](" + manifest["camera_source"] + ")"
    )
    lock = threading.Lock()

    def add_mesh(name, mesh, color, opacity=1.0):
        return server.scene.add_mesh_simple(
            name, mesh.vertices.astype(np.float32) / 1000, mesh.faces,
            color=color, opacity=opacity, flat_shading=True, side="double",
        )

    def set_view(client, view="iso"):
        target = (0.027, 0.0, 0.003)
        positions = {"iso": (0.25, -0.25, 0.15), "front": (0.36, 0, 0.005), "side": (0.07, -0.36, 0.005)}
        if mode.value == "パーツ単体":
            positions = {"iso": (0.20, -0.20, 0.13), "front": (0.24, 0, 0.01), "side": (0.04, -0.25, 0.01)}
        if mode.value == "印刷姿勢":
            target = (0.027, 0.034, 0.023)
            positions = {"iso": (0.20, -0.17, 0.18), "front": (0.24, 0.034, 0.023), "side": (0.027, -0.23, 0.023)}
        client.camera.up_direction = (0, 0, 1)
        if view == "rear":
            positions[view] = (-0.22, target[1], target[2])
        client.camera.position = positions[view]
        client.camera.look_at = target

    @server.on_client_connect
    def on_connect(client):
        set_view(client)

    @iso.on_click
    def on_iso(event):
        if event.client is not None:
            set_view(event.client, "iso")

    @front.on_click
    def on_front(event):
        if event.client is not None:
            set_view(event.client, "front")

    @side.on_click
    def on_side(event):
        if event.client is not None:
            set_view(event.client, "side")

    @rear.on_click
    def on_rear(event):
        show_board.value = False
        if event.client is not None:
            set_view(event.client, "rear")

    @download.on_click
    def download_stl(event):
        if event.client is not None:
            a = min(ANGLES, key=lambda candidate: abs(candidate - angle.value))
            path = HERE / f"exports/d435_portrait_down{a}_v{REVISION}_print_mm.stl"
            event.client.send_file_download(path.name, path.read_bytes())

    @archive.on_click
    def download_archive(event):
        if event.client is not None:
            path = HERE / "exports/pochi_d435_mount.zip"
            event.client.send_file_download(path.name, path.read_bytes())

    def rebuild(_event=None):
        a = min(ANGLES, key=lambda candidate: abs(candidate - angle.value))
        if angle.value != a:
            angle.value = a
        with lock, server.atomic():
            server.scene.reset()
            assembled = mode.value == "取り付け状態"
            part = meshes[a].copy()
            normal, vertical, centre = camera_frame(a)
            if mode.value == "印刷姿勢":
                part = trimesh.load_mesh(HERE / f"exports/d435_portrait_down{a}_v{REVISION}_print_mm.stl")
            add_mesh("/bracket", part, (20, 185, 193))
            if assembled and show_board.value:
                for i, (name, mesh) in enumerate(contexts):
                    add_mesh(f"/context/{i}", mesh, (45, 48, 59) if name.startswith("carbon") else (95, 100, 112))
            if assembled and show_camera.value:
                # A conservative capsule envelope, not an exact lens/USB CAD.
                body = rounded_plate(25.05, 25, 90, radius=12.49).translate((12.525, 0, 0))
                add_mesh("/camera/body", mesh_of(camera_transform(body, a)), (188, 191, 199))
                face = rounded_plate(0.6, 22, 87, radius=10.9).translate((25.35, 0, 0))
                add_mesh("/camera/face", mesh_of(camera_transform(face, a)), (30, 33, 41))
                for i, z in enumerate((-25, 0, 25)):
                    lens = cylinder_x(4.3 if z else 3.5, 0.7, (25.9, 0, z))
                    add_mesh(f"/camera/lens/{i}", mesh_of(camera_transform(lens, a)), (35, 93, 144))
                if show_labels.value:
                    server.scene.add_label("/camera/label", "D435：長辺90 mmを縦置き", position=(centre + vertical * 52) / 1000)
            if assembled and show_bolts.value:
                for i, (y, z) in enumerate(BOARD_HOLES):
                    if show_board.value:
                        add_mesh(f"/fixings/board_shank/{i}", bolt(1.5, 8, (0.5, y, z)), (156, 162, 176))
                        add_mesh(f"/fixings/board_head/{i}", bolt(2.75, 3, (-5, y, z)), (185, 190, 202))
                        add_mesh(f"/fixings/board_washer/{i}", bolt(3.5, 0.5, (-3.25, y, z)), (180, 185, 197))
                    insert = cylinder_x(INSERT["outer_diameter_mm"] / 2, INSERT["length_mm"], (INSERT["length_mm"] / 2, y, z))
                    insert -= cylinder_x(1.5, INSERT["length_mm"] + 0.1, (INSERT["length_mm"] / 2, y, z))
                    add_mesh(f"/fixings/board_insert/{i}", mesh_of(insert), (211, 159, 65))
                for i, z in enumerate((-22.5, 22.5)):
                    shank = camera_transform(cylinder_x(1.5, 8, (-2, 0, z)), a)
                    head = camera_transform(cylinder_x(2.75, 3, (-7.5, 0, z)), a)
                    add_mesh(f"/fixings/camera_shank/{i}", mesh_of(shank), (156, 162, 176))
                    add_mesh(f"/fixings/camera_head/{i}", mesh_of(head), (185, 190, 202))
            if mode.value != "印刷姿勢" and show_labels.value:
                if assembled:
                    for i, (y, z) in enumerate(BOARD_HOLES):
                        server.scene.add_label(f"/labels/board/{i}", f"M3 ({y:+.0f}, {z:+.0f}) mm", position=(np.array([7, y, z]) / 1000))
                start = centre + normal * (26 if assembled else 2)
                end = start + normal * 70
                server.scene.add_line_segments("/axis", np.array([[start, end]], dtype=np.float32) / 1000, colors=(251, 168, 54), line_width=3)
                server.scene.add_label("/axis/label", f"光軸：水平から{a}°下向き", position=end / 1000)
            floor_z = 0 if mode.value == "印刷姿勢" else -57
            if mode.value == "印刷姿勢":
                section = part.section(plane_origin=(0, 0, 0.1), plane_normal=(0, 0, 1))
                for i, path in enumerate(section.discrete):
                    segments = np.stack((path[:-1], path[1:]), axis=1).astype(np.float32) / 1000
                    server.scene.add_line_segments(f"/bed-contact/{i}", segments, colors=(252, 191, 65), line_width=4)
                arrow = np.array([[[-10, 20, 0], [-10, 20, 50]]], dtype=np.float32) / 1000
                server.scene.add_line_segments("/build-direction", arrow, colors=(252, 191, 65), line_width=3)
                server.scene.add_label("/build-label", "積層方向 +Z", position=(-0.01, 0.02, 0.055))
                server.scene.add_label("/bed-label", "黄色：第一層の接地面", position=(0, 0.07, 0))
            server.scene.add_grid("/grid", width=0.32, height=0.24, cell_size=0.01, section_size=0.05,
                                  plane="xy", position=(0, 0, floor_z / 1000), plane_color=(24, 28, 39), plane_opacity=0.7)
            row = reports[a]
            dims = row["print_dimensions_mm"]
            dimensions.content = (
                f"**v{REVISION}・{a}°版** · 印刷時 {dims[0]:.1f} × {dims[1]:.1f} × {dims[2]:.1f} mm\n\n"
                f"外形体積 {row['volume_cm3']:.1f} cm³ · モーター横隙間 {row['motor_lateral_clearance_mm']:.2f} mm（CAD）\n\n"
                f"付け根の幾何学的断面積：旧版の{row['root_section_area_ratio']:.1f}倍。耐荷重の実測値ではありません。\n\n"
                "角度の変更は別のSTLを選ぶ操作です。印刷後の角度は固定です。"
                "インサート下穴は印刷時からφ4.0 mmです。"
            )

    for control in (angle, mode, show_camera, show_board, show_bolts, show_labels):
        control.on_update(rebuild)

    @mode.on_update
    def reset_view(_event):
        for client in server.get_clients().values():
            set_view(client)

    rebuild()
    print(f"D435 mount GUI: http://{args.host}:{server.get_port()}", flush=True)
    threading.Event().wait()


if __name__ == "__main__":
    main()
