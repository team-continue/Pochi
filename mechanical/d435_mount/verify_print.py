"""Verify current STLs with PrusaSlicer. Generated G-code is for QA only.

Checks the intended support-disabled print setup and records actual toolpaths.
It does not claim a physical print/load test.
"""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import subprocess

from generate import ANGLES, HERE, REVISION


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--slicer", required=True, help="PrusaSlicer CLI executable")
    parser.add_argument("--output-dir", type=Path, required=True, help="QA-only G-code directory")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    version = subprocess.check_output([args.slicer, "--help"], text=True).splitlines()[0]

    def check(a):
        stl = HERE / f"exports/d435_portrait_down{a}_v{REVISION}_print_mm.stl"
        output = args.output_dir.resolve() / f"down{a}_QA_ONLY.gcode"
        cmd = [args.slicer, "--export-gcode", "--layer-height", "0.2",
               "--first-layer-height", "0.2", "--nozzle-diameter", "0.4",
               "--perimeters", "6", "--fill-density", "50%", "--fill-pattern", "gyroid",
               "--top-solid-layers", "6", "--bottom-solid-layers", "6",
               "--no-support-material", "--threads", "4", "--output", str(output), str(stl)]
        with (args.output_dir / f"slicer_down{a}.log").open("w") as log:
            subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
        types = Counter()
        layers = 0
        for line in output.open():
            if line.startswith(";TYPE:"):
                types[line.strip().split(":", 1)[1]] += 1
            if line.startswith(";LAYER_CHANGE"):
                layers += 1
        supports = sum(n for kind, n in types.items() if "support" in kind.lower())
        assert layers == 250, (a, layers)
        assert supports == 0, ("Unexpected support toolpaths", a, dict(types))
        print(f"{a} degrees: {layers} layers, {supports} support toolpaths", flush=True)
        return {"angle_deg": a, "stl_sha256": hashlib.sha256(stl.read_bytes()).hexdigest(),
                "layers": layers, "support_toolpaths": supports, "types": dict(types)}

    with ThreadPoolExecutor(max_workers=2) as pool:
        records = list(pool.map(check, ANGLES))
    result = {"slicer": version, "profile": {
        "nozzle_mm": 0.4, "layer_height_mm": 0.2, "perimeters": 6,
        "fill_density_percent": 50, "fill_pattern": "gyroid",
        "top_solid_layers": 6, "bottom_solid_layers": 6,
        "supports": False,
    }, "variants": records, "physical_print_verified": False, "load_test_verified": False}
    (HERE / "exports/print_checks.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
