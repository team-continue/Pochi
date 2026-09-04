#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

import serial

from pochi_client.protocol import CobsStreamDecoder, ProtocolError, decode_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Read Pochi telemetry directly over USB")
    parser.add_argument("--serial", default="/dev/ttyACM0")
    args = parser.parse_args()

    decoder = CobsStreamDecoder()
    last_print = 0.0
    with serial.Serial(args.serial, 115200, timeout=0.1) as teensy:
        while True:
            for frame in decoder.feed(teensy.read(teensy.in_waiting or 1)):
                try:
                    state = decode_state(frame)
                except ProtocolError:
                    continue
                now = time.monotonic()
                if now - last_print < 1.0:
                    continue
                last_print = now
                connected = sum(bool(motor.flags & (1 << 1)) for motor in state.motors)
                q = state.imu
                print(
                    f"seq={state.header.sequence} motors={connected}/12 "
                    f"imu=({q.quaternion_w:.4f}, {q.quaternion_x:.4f}, "
                    f"{q.quaternion_y:.4f}, {q.quaternion_z:.4f})"
                )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
