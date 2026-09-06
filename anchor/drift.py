"""Component 5 — Drift accumulator.

A leaky counter of off-task seconds with asymmetric decay: an off-task tick adds
``dt``, an on-task tick removes ``2 * dt`` (floor 0). It forgives fast and
remembers slowly. Low-confidence verdicts hold the counter where it is, and an
expected surface (search, AI chat) is tolerated for ``policy.tolerance_seconds``
before it starts counting as drift.
"""

from __future__ import annotations

from anchor.models import Policy, Verdict

LOW_CONFIDENCE = 0.6      # verdicts below this neither advance nor decay the timer
FORGIVENESS_RATE = 2.0    # on-task seconds are worth this many off-task seconds


class DriftAccumulator:
    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self.seconds: float = 0.0
        self.tolerated_run: float = 0.0

    def tick(self, verdict: Verdict, dt: float = 1.0) -> None:
        """Advance the accumulator by one sample of ``dt`` seconds."""
        if verdict.confidence < LOW_CONFIDENCE:
            return  # hold: an unsure judge must not move the timer either way

        if verdict.tolerated:
            self.tolerated_run += dt
            if self.tolerated_run > self.policy.tolerance_seconds:
                # An expected surface stops being "part of the task" after the tolerance.
                self.seconds += dt
            return

        if verdict.off_task:
            self.seconds += dt
        else:
            self.seconds = max(0.0, self.seconds - FORGIVENESS_RATE * dt)
        self.tolerated_run = 0.0

    def hold(self, dt: float = 1.0) -> None:
        """Explicit no-op tick, used when no verdict is available for this sample."""
        return

    @property
    def fired(self) -> bool:
        return self.seconds >= self.policy.patience_seconds

    @property
    def hard_cap(self) -> bool:
        return self.seconds >= 2 * self.policy.patience_seconds

    def reset(self) -> None:
        self.seconds = 0.0
        self.tolerated_run = 0.0

    def drain(self, amount: float) -> None:
        """Give time back after a wrong call, without forgetting everything."""
        self.seconds = max(0.0, self.seconds - amount)
