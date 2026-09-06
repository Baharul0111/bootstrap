"""Component 6 — Pause gate.

Decides whether *now* is a natural moment to speak. Ready when any one of:

* input has been idle for ``policy.pause_idle_s`` (disabled when ``None``, as for
  watching tasks where idle means "paying attention", not "available"),
* the fingerprint and pixel hash have both been stable for ``policy.pause_stable_s``,
* drift has reached the hard cap of ``2 * policy.patience_seconds`` — this is what
  catches endless scrolling where input never idles and the screen never settles.

``may_speak`` / ``blocker_reason`` cover the hard "never speak" conditions: a call
in progress, Do Not Disturb, a locked screen, or the user's own Mute.
"""

from __future__ import annotations

from anchor.gate import GateDecision
from anchor.models import Blockers, ContextFrame, Policy


class PauseGate:
    def __init__(self, policy: Policy) -> None:
        self.policy = policy
        self.stable_run: float = 0.0   # seconds both fingerprint and pixels have been unchanged

    def update(
        self,
        frame: ContextFrame,
        decision: GateDecision,
        drift_seconds: float,
        dt: float = 1.0,
    ) -> bool:
        """Fold in one sample and report whether this is a natural pause."""
        if decision.changed:
            self.stable_run = 0.0
        else:
            self.stable_run += dt

        policy = self.policy
        idle_ready = policy.pause_idle_s is not None and frame.idle_s >= policy.pause_idle_s
        stable_ready = self.stable_run >= policy.pause_stable_s
        hard_cap = drift_seconds >= 2 * policy.patience_seconds
        return idle_ready or stable_ready or hard_cap

    def reset(self) -> None:
        self.stable_run = 0.0


def may_speak(blockers: Blockers) -> bool:
    return not blockers.any


def blocker_reason(blockers: Blockers) -> str:
    """Human-readable reason for staying quiet, or ``""`` when nothing blocks."""
    if blockers.muted:
        return "muted"
    if blockers.mic_in_use:
        return "call in progress"
    if blockers.do_not_disturb:
        return "do not disturb"
    if blockers.screen_locked:
        return "screen locked"
    return ""
