"""Component 6 — Pause gate."""

from anchor.gate import GateDecision
from anchor.models import Blockers, ContextFrame, Policy
from anchor.pause import PauseGate, blocker_reason, may_speak


def frame(idle_s=0.0):
    return ContextFrame(ts=0.0, app="Google Chrome", idle_s=idle_s)


def decision(changed):
    return GateDecision(
        changed=changed,
        fingerprint="google chrome|x|",
        fingerprint_changed=changed,
        pixels_diverged=False,
        force_vision=False,
        distance=0,
    )

CHANGED = decision(True)
STABLE = decision(False)


def gate(**kw):
    return PauseGate(Policy(**kw))


def test_starts_not_ready():
    g = gate(pause_idle_s=4, pause_stable_s=10, patience_seconds=180)
    assert g.update(frame(idle_s=0.0), CHANGED, drift_seconds=0.0) is False
    assert g.stable_run == 0.0


def test_idle_triggers_at_threshold():
    g = gate(pause_idle_s=4)
    assert g.update(frame(idle_s=3.9), CHANGED, drift_seconds=0.0) is False
    assert g.update(frame(idle_s=4.0), CHANGED, drift_seconds=0.0) is True
    assert g.update(frame(idle_s=30.0), CHANGED, drift_seconds=0.0) is True


def test_stable_screen_triggers_after_pause_stable_s():
    g = gate(pause_idle_s=4, pause_stable_s=3)
    assert g.update(frame(), STABLE, 0.0) is False
    assert g.stable_run == 1.0
    assert g.update(frame(), STABLE, 0.0) is False
    assert g.stable_run == 2.0
    assert g.update(frame(), STABLE, 0.0) is True
    assert g.stable_run == 3.0


def test_stable_run_resets_on_change():
    g = gate(pause_stable_s=3)
    g.update(frame(), STABLE, 0.0)
    g.update(frame(), STABLE, 0.0)
    assert g.stable_run == 2.0
    g.update(frame(), CHANGED, 0.0)
    assert g.stable_run == 0.0
    assert g.update(frame(), STABLE, 0.0) is False
    assert g.update(frame(), STABLE, 0.0) is False
    assert g.update(frame(), STABLE, 0.0) is True


def test_hard_cap_triggers_with_never_idle_and_always_changing():
    g = gate(pause_idle_s=4, pause_stable_s=10, patience_seconds=10)
    for _ in range(5):
        assert g.update(frame(idle_s=0.0), CHANGED, drift_seconds=19.0) is False
    assert g.update(frame(idle_s=0.0), CHANGED, drift_seconds=20.0) is True
    assert g.stable_run == 0.0


def test_watching_task_idle_never_triggers_but_stability_does():
    g = gate(pause_idle_s=None, pause_stable_s=2)
    assert g.update(frame(idle_s=600.0), CHANGED, 0.0) is False
    assert g.update(frame(idle_s=600.0), CHANGED, 0.0) is False
    assert g.update(frame(idle_s=0.0), STABLE, 0.0) is False
    assert g.update(frame(idle_s=0.0), STABLE, 0.0) is True


def test_watching_task_hard_cap_still_triggers():
    g = gate(pause_idle_s=None, pause_stable_s=10, patience_seconds=5)
    assert g.update(frame(idle_s=0.0), CHANGED, drift_seconds=10.0) is True


def test_dt_scales_stable_run():
    g = gate(pause_stable_s=1)
    assert g.update(frame(), STABLE, 0.0, dt=0.5) is False
    assert g.update(frame(), STABLE, 0.0, dt=0.5) is True


def test_reset_clears_stable_run():
    g = gate(pause_stable_s=2)
    g.update(frame(), STABLE, 0.0)
    g.reset()
    assert g.stable_run == 0.0
    assert g.update(frame(), STABLE, 0.0) is False


# --- blockers ----------------------------------------------------------------

def test_may_speak_when_nothing_blocks():
    assert may_speak(Blockers()) is True
    assert blocker_reason(Blockers()) == ""


def test_muted_blocks():
    b = Blockers(muted=True)
    assert may_speak(b) is False
    assert blocker_reason(b) == "muted"


def test_mic_in_use_blocks():
    b = Blockers(mic_in_use=True)
    assert may_speak(b) is False
    assert blocker_reason(b) == "call in progress"


def test_do_not_disturb_blocks():
    b = Blockers(do_not_disturb=True)
    assert may_speak(b) is False
    assert blocker_reason(b) == "do not disturb"


def test_screen_locked_blocks():
    b = Blockers(screen_locked=True)
    assert may_speak(b) is False
    assert blocker_reason(b) == "screen locked"


def test_multiple_blockers_report_one_reason():
    b = Blockers(mic_in_use=True, do_not_disturb=True, screen_locked=True, muted=True)
    assert may_speak(b) is False
    assert blocker_reason(b) == "muted"
