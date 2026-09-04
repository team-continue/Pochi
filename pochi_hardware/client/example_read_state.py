#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time

from pochi_client import PochiClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Read Pochi state through the client API")
    parser.add_argument("--duration", type=float, default=3.0)
    args = parser.parse_args()

    with PochiClient() as robot:
        deadline = time.monotonic() + args.duration
        state = None
        while time.monotonic() < deadline:
            state = robot.wait_for_state(0.1)
        if state is None:
            raise SystemExit("FAIL: no state packet received")
        stats = robot.stats()
        connected = sum(bool(motor.flags & (1 << 1)) for motor in state.motors)
        print(
            f"PASS: rx={stats.received_packets} rate={stats.state_hz:.1f}Hz "
            f"rtt={stats.rtt_ms:.2f}ms lost={stats.dropped_packets} "
            f"motors={connected}/12 imu_samples={state.imu.sample_counter}"
        )


if __name__ == "__main__":
    main()
