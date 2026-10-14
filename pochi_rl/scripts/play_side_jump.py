"""Watch an IK-controlled lateral jump in a browser, without training.

uv run --extra mpc python scripts/play_side_jump.py --direction left --port 8093
"""

from __future__ import annotations

import argparse

import mjviser
import numpy as np
import viser

from pochi_rl.control.side_jump import JUMP_SIZES, SideJumpConfig, SideJumpSim


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--host", default="0.0.0.0")
  parser.add_argument("--port", type=int, default=8093)
  parser.add_argument("--direction", choices=("left", "right"), default="left")
  parser.add_argument("--jump-size", choices=JUMP_SIZES, default="large")
  parser.add_argument(
    "--lateral-speed",
    type=float,
    default=None,
    help="override the selected preset takeoff speed (m/s)",
  )
  args = parser.parse_args()
  try:
    cfg = SideJumpConfig.for_size(
      args.jump_size,
      direction=1 if args.direction == "left" else -1,
      lateral_speed=args.lateral_speed,
    )
  except ValueError as exc:
    parser.error(str(exc))
  sim = SideJumpSim(cfg)
  sim.model.stat.extent = 0.65
  server = viser.ViserServer(
    host=args.host, port=args.port, label="Pochi · IK side jump"
  )
  server.gui.add_markdown(
    "### Pochi · IK 横ジャンプ\n"
    "沈み込み → 踏み切り → 空中 → 着地 → 足位置を戻す → 標準姿勢\n\n"
    "**オレンジ:** 足先の目標位置  \n"
    "左はロボットの +Y、右は −Y 方向です。\n\n"
    "大きさと方向を選び、**Controls → Reset** で再実行。"
    " **Slower** でスロー再生できます。"
  )
  direction = server.gui.add_dropdown(
    "ジャンプ方向",
    ("左 (+Y)", "右 (−Y)"),
    initial_value="左 (+Y)" if cfg.direction == 1 else "右 (−Y)",
  )
  size_labels = {
    "small": "小（約13 cm）",
    "medium": "中（約33 cm）",
    "large": "大（約52 cm）",
  }
  size = server.gui.add_dropdown(
    "ジャンプの大きさ",
    tuple(size_labels.values()),
    initial_value=size_labels[args.jump_size],
  )
  if args.lateral_speed is not None:
    server.gui.add_markdown(
      f"横速度目標を {args.lateral_speed:g} m/s に上書き中です。"
      "大きさの表示距離とは異なる場合があります。"
    )
  status = server.gui.add_html("")
  phases = {
    "settle": "準備",
    "crouch": "沈み込み",
    "thrust": "踏み切り",
    "flight": "空中",
    "land": "着地・吸収",
    "recover": "標準姿勢へ復帰（一脚ずつ踏み直し）",
    "hold": "標準姿勢",
  }

  def reset(model=None, data=None) -> None:
    selected = next(key for key, label in size_labels.items() if label == size.value)
    sim.cfg = SideJumpConfig.for_size(
      selected,
      direction=1 if direction.value == "左 (+Y)" else -1,
      lateral_speed=args.lateral_speed,
    )
    sim.reset()

  def render(scene) -> None:
    scene.update_from_mjdata(sim.data)
    offset = (
      -sim.data.xpos[sim.base_id] if scene.camera_tracking_enabled else np.zeros(3)
    )
    server.scene.add_point_cloud(
      "/jump/targets",
      points=(sim.foot_targets + offset).astype(np.float32),
      colors=(255, 140, 40),
      point_size=0.015,
    )
    result = "停止（Reset で再実行）" if sim.failed else phases[sim.phase]
    if sim.success:
      result = "標準姿勢で静止・完了"
    metrics = sim.metrics()
    airborne = metrics["airborne_lateral_distance_m"]
    flight_distance = "—" if airborne is None else f"{airborne * 100:+.1f} cm"
    status.content = (
      '<div style="padding:0.5em 1em;line-height:1.7">'
      f"<b>{result}</b><br>"
      f"時刻: {sim.data.time:.2f} s<br>"
      f"横移動（踏み切り・着地を含む）: {sim.data.qpos[1] * 100:+.1f} cm<br>"
      f"空中での横移動: {flight_distance}<br>"
      f"全脚の離地時間: {sim.airtime:.3f} s<br>"
      f"胴体の高さ: {sim.base_height * 100:.1f} cm<br>"
      f"接地: {sum(sim.contacts)} / 4 脚<br>"
      f"胴体の傾き: {metrics['final_tilt_deg']:.2f}°<br>"
      f"標準の関節角との差（最大）: {metrics['standard_joint_error_deg']:.2f}°</div>"
    )

  viewer = mjviser.Viewer(
    sim.model,
    sim.data,
    server=server,
    step_fn=sim.step,
    reset_fn=reset,
    render_fn=render,
  )
  print(f"Pochi IK jump viewer: http://localhost:{server.get_port()}", flush=True)
  viewer.run()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
