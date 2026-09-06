"""Wall-clock abstraction so tests can fake sleep, lid-close and restarts.

The deferred reminder compares an absolute deadline stored in SQLite against
``clock.now()``; it never relies on a sleeping thread, so jumping the fake clock
forward is exactly what a real laptop waking from sleep looks like to the code.
"""

from __future__ import annotations

import time


class Clock:
    """Real wall clock (epoch seconds)."""

    def now(self) -> float:
        return time.time()

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class FakeClock(Clock):
    """Deterministic clock for tests. ``advance`` simulates time passing, including a sleeping laptop."""

    def __init__(self, start: float = 1_800_000_000.0):
        self._now = float(start)

    def now(self) -> float:
        return self._now

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)

    def advance(self, seconds: float) -> None:
        self._now += float(seconds)

    def set(self, epoch: float) -> None:
        self._now = float(epoch)
