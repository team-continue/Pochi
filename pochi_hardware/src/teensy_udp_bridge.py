#!/usr/bin/env python3
from __future__ import annotations

import argparse
import socket
import time

import serial

from pochi_client.protocol import (
    CobsStreamDecoder,
    ProtocolError,
    decode_command,
    decode_state,
    encode_usb_frame,
)


def endpoint(value: str) -> tuple[str, int]:
    host, separator, port = value.rpartition(":")
    if not separator:
        raise argparse.ArgumentTypeError("endpoint must be HOST:PORT")
    return host, int(port)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pochi Teensy USB/UDP bridge")
    parser.add_argument("--serial", default="/dev/ttyACM0", help="Teensy USB serial device")
    parser.add_argument("--baud", type=int, default=115200, help="nominal USB CDC baud rate")
    parser.add_argument("--command-bind", type=endpoint, default=("0.0.0.0", 15000))
    parser.add_argument("--state-dest", type=endpoint, default=("127.0.0.1", 15001))
    args = parser.parse_args()

    udp = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    udp.bind(args.command_bind)
    udp.setblocking(False)
    decoder = CobsStreamDecoder()

    with serial.Serial(args.serial, args.baud, timeout=0, write_timeout=0) as teensy:
        while True:
            did_work = False
            waiting = teensy.in_waiting
            if waiting:
                frames = decoder.feed(teensy.read(waiting))
            else:
                frames = []
            for frame in frames:
                try:
                    decode_state(frame)
                except ProtocolError:
                    continue
                udp.sendto(frame, args.state_dest)
                did_work = True

            while True:
                try:
                    command, _source = udp.recvfrom(2048)
                except BlockingIOError:
                    break
                try:
                    decode_command(command)
                except ProtocolError:
                    continue
                teensy.write(encode_usb_frame(command))
                did_work = True

            if not did_work:
                time.sleep(0.0005)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
