"""Show live MPC walking and its planned trajectories in viser.

uv run --extra mpc python scripts/play_mpc_walk.py --port 8092
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import replace

import mjviser
import numpy as np
import viser

from pochi_rl.control.mpc_walk import MPCWalkSim, WalkConfig
from pochi_rl.robot import LEGS


def main() -> int:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--host", default="0.0.0.0")
  parser.add_argument("--port", type=int, default=8092)
  parser.add_argument("--speed", type=float, default=0.06, help="forward speed, m/s")
  parser.add_argument(
    "--loop-duration",
    type=float,
    default=30.0,
    help="restart after this many simulated seconds; 0 disables",
  )
  args = parser.parse_args()
  if not np.isfinite(args.loop_duration) or args.loop_duration < 0:
    parser.error("--loop-duration must be finite and nonnegative")
  sim = MPCWalkSim(WalkConfig(speed=args.speed))
  # mjviser derives its initial camera distance from this visualization statistic.
  sim.model.stat.extent = 0.65
  server = viser.ViserServer(host=args.host, port=args.port, label="Pochi · MPC walk")
  server.gui.add_markdown(
    "### Pochi · MPC walking\n"
    "Live MuJoCo physics · one foot at a time\n\n"
    "**Cyan:** MPC COM prediction (0.8 s)  \n"
    "**Yellow:** desired COM path  \n"
    "**Purple:** measured COM trail  \n"
    "**Orange:** foot targets · **Green:** planned ground forces\n\n"
    "Use **Controls** to pause, reset, or slow playback. "
    "Change speed below, then press **Reset** to apply."
  )
  speed = server.gui.add_slider(
    "Walking speed (m/s)",
    min=-0.09,
    max=0.09,
    step=0.01,
    initial_value=float(np.clip(args.speed, -0.09, 0.09)),
  )
  overlays = server.gui.add_checkbox("Show MPC plan", initial_value=True)
  status = server.gui.add_html("")
  frame = server.scene.add_frame("/mpc", show_axes=False)
  trail: deque[np.ndarray] = deque(maxlen=400)
  last_trail_step = -1

  def reset(model=None, data=None) -> None:
    nonlocal last_trail_step
    sim.cfg = replace(sim.cfg, speed=speed.value)
    sim.reset()
    trail.clear()
    last_trail_step = -1

  def step(model, data) -> None:
    if args.loop_duration and sim.data.time >= args.loop_duration:
      reset()
    sim.step()

  def line(name: str, points: np.ndarray, color: tuple[int, int, int]) -> None:
    if len(points) > 1:
      server.scene.add_line_segments(
        f"/mpc/{name}",
        points=np.stack((points[:-1], points[1:]), axis=1),
        colors=color,
        line_width=3,
      )

  def render(scene) -> None:
    nonlocal last_trail_step
    scene.update_from_mjdata(sim.data)
    frame.visible = overlays.value
    frame.position = (
      -sim.data.xpos[sim.base_id] if scene.camera_tracking_enabled else np.zeros(3)
    )
    # Network and GUI updates at 20 Hz; physical control still runs at 200 Hz.
    if (
      last_trail_step >= 0
      and not sim.failed
      and (sim.steps == last_trail_step or sim.steps - last_trail_step < 10)
    ):
      return
    last_trail_step = sim.steps
    com = sim.data.subtree_com[sim.base_id].copy()
    trail.append(com)
    line("measured", np.array(trail), (186, 112, 255))
    line("prediction", np.vstack((com, sim.planner.prediction[:, :3])), (35, 215, 240))
    reference = np.array(
      [
        sim.origin + [sim.command_at(sim.data.time + k * sim.cfg.mpc.dt)[0], 0, 0]
        for k in range(sim.cfg.mpc.horizon + 1)
      ]
    )
    line("reference", reference, (255, 214, 70))
    server.scene.add_point_cloud(
      "/mpc/feet",
      points=sim.foot_targets.astype(np.float32),
      colors=(255, 140, 40),
      point_size=0.018,
    )
    server.scene.add_line_segments(
      "/mpc/forces",
      points=np.stack((sim.feet, sim.feet + 0.003 * sim.force), axis=1),
      colors=(65, 220, 135),
      line_width=3,
    )
    supports = " · ".join(
      leg for leg, contact in zip(LEGS, sim.contacts, strict=True) if contact
    )
    state = sim.state()
    status.content = (
      '<div style="padding:0.5em 1em;font-size:0.9em;line-height:1.6">'
      f"<b>{'STOPPED — reset to retry' if sim.failed else 'MPC crawl'}</b><br>"
      f"Time: {sim.data.time:.1f} s · Distance: {sim.data.qpos[0]:.2f} m<br>"
      f"Speed: {state[6]:.3f} / {sim.cfg.speed:.2f} m/s<br>"
      f"Base height: {sim.base_height:.3f} m<br>"
      f"Support: {supports}<br>"
      f"QP: {sim.planner.status} · {sim.planner.solve_ms:.1f} ms · "
      f"failures: {sim.planner.failures}</div>"
    )

  viewer = mjviser.Viewer(
    sim.model, sim.data, server=server, step_fn=step, reset_fn=reset, render_fn=render
  )
  print(f"Pochi MPC viewer: http://localhost:{server.get_port()}", flush=True)
  viewer.run()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
