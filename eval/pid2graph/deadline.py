"""Outer wall-clock guard for local, synchronous evaluation requests."""
from __future__ import annotations

import signal
from contextlib import contextmanager


@contextmanager
def wall_clock_limit(seconds):
    # The evaluation CLI runs on the main thread on POSIX. SSE keepalive bytes
    # can prevent a socket timeout without yielding a complete SDK event.
    if seconds <= 0:
        raise ValueError("Wall-clock limit must be positive")
    previous = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    if previous_timer[0]:
        raise RuntimeError("Refusing to replace an active outer deadline")

    def expired(signum, frame):
        raise TimeoutError("Outer evaluation wall-clock deadline exceeded")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
