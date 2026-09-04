from __future__ import annotations

import dataclasses
import math
import socket
import threading
import time
from collections.abc import Iterable

from .joint_layout import JOINT_BY_NAME
from .models import ClientStats
from .protocol import (
    COMMAND_ENABLE,
    CONTROL_DISABLED,
    CONTROL_MIT,
    MotorCommand,
    ProtocolError,
    StatePacket,
    decode_state,
    disabled_commands,
    encode_command,
)


class PochiClient:
    """Threaded UDP client for one Pochi Teensy bridge.

    The transmitter always sends one atomic 12-motor snapshot. Starting the
    client is safe: IDs 0..11 remain disabled until ``set_mit`` is explicitly
    called with ``enable=True``.
    """

    def __init__(
        self,
        command_address: tuple[str, int] = ("127.0.0.1", 15000),
        state_bind: tuple[str, int] = ("0.0.0.0", 15001),
        *,
        command_hz: float = 200.0,
        state_timeout: float = 0.25,
    ) -> None:
        if command_hz <= 0.0:
            raise ValueError("command_hz must be positive")
        self.command_address = command_address
        self.state_bind = state_bind
        self.command_hz = command_hz
        self.state_timeout = state_timeout

        self._commands = disabled_commands()
        self._command_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._state_condition = threading.Condition(self._state_lock)
        self._latest_state: StatePacket | None = None
        self._stats = ClientStats()
        self._sequence = 0
        self._emergency_stop = False
        self._running = threading.Event()
        self._command_socket: socket.socket | None = None
        self._state_socket: socket.socket | None = None
        self._tx_thread: threading.Thread | None = None
        self._rx_thread: threading.Thread | None = None
        self._send_times: dict[int, float] = {}
        self._previous_state_sequence: int | None = None
        self._previous_state_arrival = 0.0

    def start(self) -> PochiClient:
        if self._running.is_set():
            return self
        self._command_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        state_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        state_socket.bind(self.state_bind)
        state_socket.settimeout(0.05)
        self._state_socket = state_socket
        self._running.set()
        self._tx_thread = threading.Thread(
            target=self._transmit_loop,
            name="pochi-command",
            daemon=True,
        )
        self._rx_thread = threading.Thread(
            target=self._receive_loop,
            name="pochi-state",
            daemon=True,
        )
        self._tx_thread.start()
        self._rx_thread.start()
        return self

    def close(self) -> None:
        if not self._running.is_set() and self._command_socket is None:
            return
        self._running.clear()
        if self._tx_thread is not None:
            self._tx_thread.join(timeout=0.5)
        if self._rx_thread is not None:
            self._rx_thread.join(timeout=0.5)

        self.emergency_stop()
        for _ in range(5):
            self._send_once()
            time.sleep(0.002)

        if self._command_socket is not None:
            self._command_socket.close()
        if self._state_socket is not None:
            self._state_socket.close()
        self._command_socket = None
        self._state_socket = None
        self._tx_thread = None
        self._rx_thread = None

    def __enter__(self) -> PochiClient:
        return self.start()

    def __exit__(self, _exc_type: object, _exc: object, _traceback: object) -> None:
        self.close()

    def latest_state(self) -> StatePacket | None:
        with self._state_lock:
            return self._latest_state

    def wait_for_state(self, timeout: float = 1.0) -> StatePacket | None:
        deadline = time.monotonic() + timeout
        with self._state_condition:
            initial = self._latest_state
            while self._latest_state is initial:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    break
                self._state_condition.wait(remaining)
            return self._latest_state

    def stats(self) -> ClientStats:
        with self._stats_lock:
            return dataclasses.replace(self._stats)

    def state_age(self) -> float:
        with self._stats_lock:
            last = self._stats.last_state_monotonic
        return math.inf if last == 0.0 else time.monotonic() - last

    @property
    def connected(self) -> bool:
        return self.state_age() <= self.state_timeout

    def set_mit(
        self,
        motor: int | str,
        *,
        position_rad: float,
        velocity_rad_s: float,
        kp: float,
        kd: float,
        torque_nm: float,
        enable: bool = True,
    ) -> None:
        motor_id = self._resolve_motor_id(motor)
        command = MotorCommand(
            motor_id=motor_id,
            control_mode=CONTROL_MIT if enable else CONTROL_DISABLED,
            flags=COMMAND_ENABLE if enable else 0,
            position_rad=position_rad,
            velocity_rad_s=velocity_rad_s,
            kp=kp,
            kd=kd,
            torque_nm=torque_nm,
        )
        with self._command_lock:
            self._commands[motor_id] = command

    def set_all_mit(self, commands: Iterable[MotorCommand]) -> None:
        commands = list(commands)
        encode_command(commands, 0)
        commands.sort(key=lambda command: command.motor_id)
        with self._command_lock:
            self._commands = commands

    def disable(self, motor: int | str) -> None:
        motor_id = self._resolve_motor_id(motor)
        with self._command_lock:
            self._commands[motor_id] = MotorCommand(motor_id=motor_id)

    def disable_all(self) -> None:
        with self._command_lock:
            self._commands = disabled_commands()

    def emergency_stop(self) -> None:
        with self._command_lock:
            self._commands = disabled_commands()
            self._emergency_stop = True

    def clear_emergency_stop(self) -> None:
        with self._command_lock:
            self._emergency_stop = False

    def _resolve_motor_id(self, motor: int | str) -> int:
        if isinstance(motor, str):
            try:
                return JOINT_BY_NAME[motor].motor_id
            except KeyError as exc:
                raise ValueError(f"unknown joint {motor!r}") from exc
        if not 0 <= motor < 12:
            raise ValueError("motor ID must be in the range 0..11")
        return motor

    def _send_once(self) -> None:
        if self._command_socket is None:
            return
        with self._command_lock:
            commands = list(self._commands)
            emergency_stop = self._emergency_stop
            sequence = self._sequence
            self._sequence = (self._sequence + 1) & 0xFFFFFFFF
        try:
            packet = encode_command(commands, sequence, emergency_stop=emergency_stop)
            self._command_socket.sendto(packet, self.command_address)
            sent_at = time.monotonic()
            self._send_times[sequence] = sent_at
            if len(self._send_times) > 1024:
                oldest = next(iter(self._send_times))
                self._send_times.pop(oldest, None)
            with self._stats_lock:
                self._stats.sent_packets += 1
        except (OSError, ProtocolError) as exc:
            with self._stats_lock:
                self._stats.last_error = str(exc)

    def _transmit_loop(self) -> None:
        period = 1.0 / self.command_hz
        next_send = time.monotonic()
        while self._running.is_set():
            now = time.monotonic()
            if now >= next_send:
                self._send_once()
                next_send += period
                if next_send < now - period:
                    next_send = now + period
                continue
            time.sleep(min(next_send - now, 0.001))

    def _receive_loop(self) -> None:
        assert self._state_socket is not None
        while self._running.is_set():
            try:
                packet, _source = self._state_socket.recvfrom(2048)
            except socket.timeout:
                continue
            except OSError as exc:
                if self._running.is_set():
                    with self._stats_lock:
                        self._stats.last_error = str(exc)
                return
            try:
                state = decode_state(packet)
            except ProtocolError as exc:
                with self._stats_lock:
                    self._stats.invalid_packets += 1
                    self._stats.last_error = str(exc)
                continue
            self._record_state(state)

    def _record_state(self, state: StatePacket) -> None:
        arrived = time.monotonic()
        with self._state_condition:
            self._latest_state = state
            self._state_condition.notify_all()

        dropped = 0
        if self._previous_state_sequence is not None:
            delta = (state.header.sequence - self._previous_state_sequence) & 0xFFFFFFFF
            if 1 < delta < 0x80000000:
                dropped = delta - 1
        self._previous_state_sequence = state.header.sequence

        instantaneous_hz = 0.0
        if self._previous_state_arrival > 0.0 and arrived > self._previous_state_arrival:
            instantaneous_hz = 1.0 / (arrived - self._previous_state_arrival)
        self._previous_state_arrival = arrived

        accepted_sequence = state.motors[0].command_sequence if state.motors else None
        rtt_ms = None
        if accepted_sequence is not None:
            sent_at = self._send_times.pop(accepted_sequence, None)
            if sent_at is not None:
                rtt_ms = (arrived - sent_at) * 1000.0

        with self._stats_lock:
            self._stats.received_packets += 1
            self._stats.dropped_packets += dropped
            self._stats.last_state_monotonic = arrived
            if instantaneous_hz > 0.0:
                self._stats.state_hz = (
                    instantaneous_hz
                    if self._stats.state_hz == 0.0
                    else self._stats.state_hz * 0.9 + instantaneous_hz * 0.1
                )
            if rtt_ms is not None:
                self._stats.rtt_ms = rtt_ms
