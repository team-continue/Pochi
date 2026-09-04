#!/usr/bin/env python3
from __future__ import annotations

import argparse
import socket
import time

from pochi_client.protocol import (
    STATE_COMMAND_ALIVE,
    ProtocolError,
    decode_state,
    disabled_commands,
    encode_command,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Safe UDP loop test using disabled motor commands")
    parser.add_argument("--bridge", default="127.0.0.1", help="bridge address")
    parser.add_argument("--duration", type=float, default=3.0)
    args = parser.parse_args()

    command_target = (args.bridge, 15000)
    state = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    state.bind(("0.0.0.0", 15001))
    state.setblocking(False)
    command = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    sequence = 0
    next_send = time.monotonic()
    deadline = next_send + args.duration
    packets = 0
    alive_packets = 0
    last_state = None
    try:
        while time.monotonic() < deadline:
            now = time.monotonic()
            if now >= next_send:
                command.sendto(encode_command(disabled_commands(), sequence), command_target)
                sequence = (sequence + 1) & 0xFFFFFFFF
                next_send += 0.01
            while True:
                try:
                    packet, _ = state.recvfrom(2048)
                except BlockingIOError:
                    break
                try:
                    last_state = decode_state(packet)
                except ProtocolError:
                    continue
                packets += 1
                alive_packets += bool(last_state.header.flags & STATE_COMMAND_ALIVE)
            time.sleep(0.0005)
    finally:
        command.close()
        state.close()

    if last_state is None or packets == 0 or alive_packets == 0:
        raise SystemExit("FAIL: no valid command-alive telemetry received")
    connected = sum(bool(motor.flags & (1 << 1)) for motor in last_state.motors)
    print(
        f"PASS: state_packets={packets} command_alive={alive_packets} "
        f"motors={connected}/12 imu_samples={last_state.imu.sample_counter}"
    )


if __name__ == "__main__":
    main()
