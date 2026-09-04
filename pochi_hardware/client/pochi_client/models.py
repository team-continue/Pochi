from __future__ import annotations

import dataclasses


@dataclasses.dataclass(slots=True)
class ClientStats:
    sent_packets: int = 0
    received_packets: int = 0
    invalid_packets: int = 0
    dropped_packets: int = 0
    state_hz: float = 0.0
    rtt_ms: float = 0.0
    last_state_monotonic: float = 0.0
    last_error: str = ""

    @property
    def connected(self) -> bool:
        return self.last_state_monotonic > 0.0
