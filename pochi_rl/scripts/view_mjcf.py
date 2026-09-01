"""Open the Pochi flat-scene MJCF in MuJoCo's viewer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> int:
  scene = Path(__file__).resolve().parents[1] / "assets" / "pochi" / "scene_flat.xml"
  return subprocess.call([sys.executable, "-m", "mujoco.viewer", "--mjcf", str(scene)])


if __name__ == "__main__":
  raise SystemExit(main())
