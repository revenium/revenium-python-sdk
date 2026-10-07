"""Stops sending usage records to a metering endpoint that has stopped answering.

While the endpoint answers, every record is sent as it comes. Once
``FAILURES_TO_OPEN`` deliveries in a row fail with a retryable error, the
circuit opens: the delivery workers buffer new records without sending them,
so an outage no longer holds a worker through every attempt's timeout for
each record. One record every ``PROBE_INTERVAL_SECONDS`` is still sent, and
the first one that succeeds closes the circuit again.
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Callable

logger = logging.getLogger("revenium_middleware")

FAILURES_TO_OPEN = 3
PROBE_INTERVAL_SECONDS = 5.0


class DeliveryCircuit:
    """Counts consecutive retryable delivery failures and rations sends while the endpoint is down.

    Its lock is reentrant because delivery outcomes are recorded from metering
    code that a garbage-collector finalizer can run on any thread.
    """

    def __init__(
        self,
        failures_to_open: int = FAILURES_TO_OPEN,
        probe_interval: float = PROBE_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._failures_to_open = failures_to_open
        self._probe_interval = probe_interval
        self._clock = clock
        self._lock = threading.RLock()
        self._consecutive_failures = 0
        self._open = False
        self._next_probe_at = 0.0

    def is_open(self) -> bool:
        with self._lock:
            return self._open

    def allows_send(self) -> bool:
        """Whether the next record should be sent; while open, True once per probe interval."""
        with self._lock:
            if not self._open:
                return True
            now = self._clock()
            if now < self._next_probe_at:
                return False
            self._next_probe_at = now + self._probe_interval
            return True

    def record_success(self) -> bool:
        """Note a delivered record; True when it closed the circuit."""
        with self._lock:
            self._consecutive_failures = 0
            was_open, self._open = self._open, False
        if was_open:
            logger.warning("Metering endpoint is answering again; sending usage records and replaying buffered ones")
        return was_open

    def record_failure(self) -> None:
        """Note a record that failed with a retryable error after the client's own retries."""
        with self._lock:
            self._consecutive_failures += 1
            self._next_probe_at = self._clock() + self._probe_interval
            if self._open or self._consecutive_failures < self._failures_to_open:
                return
            self._open = True
        logger.warning(
            "Metering endpoint failed %d deliveries in a row; buffering new usage records without sending them "
            "and trying one every %.0fs until it answers",
            self._failures_to_open, self._probe_interval,
        )


_circuit = DeliveryCircuit()


def get_circuit() -> DeliveryCircuit:
    return _circuit


def reset() -> None:
    """Start over with a closed circuit."""
    global _circuit
    _circuit = DeliveryCircuit()
