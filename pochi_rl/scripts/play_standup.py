"""Watch Pochi stand up, in a browser, over viser.

Serves ``assets/pochi/scene_flat.xml`` driven by the scripted stand-up policy
in :mod:`pochi_rl.control.standup` -- no checkpoint and no training run needed.
Reset cuts the motors and lets the robot collapse onto its belly, then stands
it back up on its feet.  Press *Reset* in the viewer's Controls tab to run the
whole thing again, or *Slower* to watch it in slow motion.

  uv run python scripts/play_standup.py [--port 8080] [--rise-duration 6]
"""

from __future__ import annotations

import argparse

import mjviser
import viser

from pochi_rl.control.mujoco_driver import StandUpSim
from pochi_rl.control.standup import StandUpConfig

# How often to refresh the read-out, in physics steps.  At 200 Hz sim this is
# about 10 Hz, which is plenty for numbers a human is reading.
STATUS_EVERY = 20


def parse_args() -> argparse.Namespace:
  cfg = StandUpConfig()
  p = argparse.ArgumentParser(description=__doc__)
  p.add_argument("--port", type=int, default=8080, help="viser HTTP port")
  p.add_argument("--host", default="0.0.0.0", help="viser bind address")
  p.add_argument("--settle-duration", type=float, default=cfg.settle_duration_s)
  p.add_argument("--approach-duration", type=float, default=cfg.approach_duration_s)
  p.add_argument("--rise-duration", type=float, default=cfg.rise_duration_s)
  p.add_argument("--crouch-height", type=float, default=cfg.crouch_height)
  return p.parse_args()


def main() -> int:
  args = parse_args()
  sim = StandUpSim(
    StandUpConfig(
      settle_duration_s=args.settle_duration,
      approach_duration_s=args.approach_duration,
      rise_duration_s=args.rise_duration,
      crouch_height=args.crouch_height,
    )
  )

  server = viser.ViserServer(host=args.host, port=args.port)

  # A live read-out of what the policy is doing, above mjviser's own controls:
  # phase, the height it is commanding, and the height the robot managed.
  status = server.gui.add_html("")

  def refresh() -> None:
    status.content = (
      '<div style="font-size:0.85em;line-height:1.3;padding:0.5em 1em;">'
      f"<strong>Phase:</strong> {sim.phase}<br/>"
      f"<strong>Commanded height:</strong> "
      f"{sim.controller.target_base_height:.3f} m<br/>"
      f"<strong>Actual height:</strong> {sim.base_height:.3f} m</div>"
    )

  ticks = 0

  def step(model, data) -> None:
    nonlocal ticks
    sim.step()
    ticks += 1
    if ticks % STATUS_EVERY == 0:
      refresh()

  def reset(model, data) -> None:
    sim.reset()
    refresh()

  refresh()
  viewer = mjviser.Viewer(
    sim.model, sim.data, step_fn=step, reset_fn=reset, server=server
  )

  # viser silently walks to the next free port if --port is taken, so report
  # the one it actually bound rather than the one that was asked for.
  print(f"Serving the Pochi stand-up demo at http://localhost:{server.get_port()}")
  viewer.run()
  return 0


if __name__ == "__main__":
  raise SystemExit(main())
