#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
import select
import struct
import termios
import time
from pathlib import Path


SCAN_RANGES = [
    ["0x0000", "0x000F"],
    ["0x1000", "0x101F"],
    ["0x2000", "0x20FF"],
    ["0x3000", "0x30FF"],
    ["0x7000", "0x70FF"],
]


def configure_serial(fd: int) -> None:
    attrs = termios.tcgetattr(fd)
    attrs[0] = 0
    attrs[1] = 0
    attrs[2] = termios.CS8 | termios.CREAD | termios.CLOCAL
    attrs[3] = 0
    attrs[4] = termios.B115200
    attrs[5] = termios.B115200
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 1
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    termios.tcflush(fd, termios.TCIOFLUSH)


def decode_raw(raw_hex: str) -> dict[str, object]:
    if raw_hex == "--------":
        return {"raw_hex": None, "u32_le": None, "i32_le": None, "float_le": None}
    raw = bytes.fromhex(raw_hex)
    value = struct.unpack("<f", raw)[0]
    return {
        "raw_hex": raw_hex,
        "u32_le": int.from_bytes(raw, "little", signed=False),
        "i32_le": int.from_bytes(raw, "little", signed=True),
        "float_le": value if math.isfinite(value) else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="/dev/ttyACM0")
    parser.add_argument("--output", required=True)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    raw_log = output.with_suffix(".log")
    fd = os.open(args.device, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    configure_serial(fd)

    motors: dict[str, list[dict[str, object]]] = {str(i): [] for i in range(12)}
    lines: list[str] = []
    buffer = b""
    started = False
    ended = False
    summary = ""
    deadline = time.monotonic() + args.timeout
    try:
        while time.monotonic() < deadline and not ended:
            ready, _, _ = select.select([fd], [], [], 0.25)
            if not ready:
                continue
            try:
                chunk = os.read(fd, 65536)
            except BlockingIOError:
                continue
            if not chunk:
                continue
            buffer += chunk
            while b"\n" in buffer:
                raw_line, buffer = buffer.split(b"\n", 1)
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r")
                lines.append(line)
                if line.startswith("DUMP_BEGIN\t"):
                    started = True
                elif line.startswith("P\t"):
                    fields = line.split("\t")
                    if len(fields) != 6:
                        continue
                    _, motor_id, index, result, status, raw_hex = fields
                    record = {
                        "index": index,
                        "result": result,
                        "status": int(status),
                        **decode_raw(raw_hex),
                    }
                    motors.setdefault(motor_id, []).append(record)
                elif line.startswith("DUMP_END\t"):
                    summary = line
                    ended = True
    finally:
        os.close(fd)

    raw_log.write_text("\n".join(lines) + "\n", encoding="utf-8")
    document = {
        "format": "pochi.robostride.parameter-dump.v1",
        "captured_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "device": args.device,
        "motor_model": "RS02",
        "motor_ids": list(range(12)),
        "master_id": 0xFD,
        "can_bitrate": 1_000_000,
        "read_only": True,
        "scan_ranges": SCAN_RANGES,
        "complete": started and ended,
        "summary": summary,
        "motors": motors,
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, output)

    if not started or not ended:
        raise SystemExit(f"incomplete dump; raw log: {raw_log}")
    print(output)
    print(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
