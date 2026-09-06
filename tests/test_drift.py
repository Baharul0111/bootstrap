"""Component 5 — Drift accumulator."""

from anchor.drift import DriftAccumulator
from anchor.models import Policy, Verdict

OFF = Verdict(drift=1.0, confidence=0.9)
ON = Verdict(drift=0.0, confidence=0.9)
BORDERLINE_OFF = Verdict(drift=0.5, confidence=0.9)
UNSURE_OFF = Verdict(drift=1.0, confidence=0.59)
UNSURE_ON = Verdict(drift=0.0, confidence=0.3)
TOLERATED = Verdict(drift=0.8, confidence=0.9, tolerated=True)


def acc(**kw):
    return DriftAccumulator(Policy(**kw))


def test_starts_at_zero():
    a = acc()
    assert a.seconds == 0.0
    assert a.tolerated_run == 0.0
    assert a.fired is False
    assert a.hard_cap is False


def test_off_task_adds_one_per_tick():
    a = acc()
    for _ in range(3):
        a.tick(OFF)
    assert a.seconds == 3.0


def test_drift_of_exactly_half_counts_as_off_task():
    a = acc()
    a.tick(BORDERLINE_OFF)
    assert a.seconds == 1.0


def test_on_task_removes_two_per_tick_with_floor_zero():
    a = acc()
    for _ in range(3):
        a.tick(OFF)
    a.tick(ON)
    assert a.seconds == 1.0
    a.tick(ON)
    assert a.seconds == 0.0
    a.tick(ON)
    assert a.seconds == 0.0


def test_fires_at_exactly_patience_seconds():
    a = acc(patience_seconds=5)
    for _ in range(4):
        a.tick(OFF)
    assert a.fired is False
    a.tick(OFF)
    assert a.seconds == 5.0
    assert a.fired is True


def test_single_off_task_sample_never_fires():
    a = acc(patience_seconds=2)
    a.tick(OFF)
    assert a.fired is False


def test_low_confidence_holds_in_both_directions():
    a = acc()
    a.tick(UNSURE_OFF)
    assert a.seconds == 0.0
    a.tick(OFF)
    a.tick(OFF)
    a.tick(UNSURE_ON)
    assert a.seconds == 2.0
    a.tick(UNSURE_OFF)
    assert a.seconds == 2.0


def test_hold_is_a_no_op():
    a = acc()
    a.tick(OFF)
    a.hold()
    a.hold(dt=5.0)
    assert a.seconds == 1.0


def test_tolerated_holds_for_tolerance_seconds_then_counts():
    a = acc(tolerance_seconds=3)
    for _ in range(3):
        a.tick(TOLERATED)
    assert a.seconds == 0.0
    assert a.tolerated_run == 3.0
    a.tick(TOLERATED)
    assert a.seconds == 1.0
    a.tick(TOLERATED)
    assert a.seconds == 2.0


def test_tolerated_run_resets_on_a_plain_verdict():
    a = acc(tolerance_seconds=3)
    a.tick(TOLERATED)
    a.tick(TOLERATED)
    assert a.tolerated_run == 2.0
    a.tick(ON)
    assert a.tolerated_run == 0.0
    a.tick(TOLERATED)
    a.tick(OFF)
    assert a.tolerated_run == 0.0


def test_hard_cap_at_twice_patience():
    a = acc(patience_seconds=3)
    for _ in range(5):
        a.tick(OFF)
    assert a.fired is True
    assert a.hard_cap is False
    a.tick(OFF)
    assert a.hard_cap is True


def test_drain_subtracts_with_floor_zero():
    a = acc()
    for _ in range(5):
        a.tick(OFF)
    a.drain(3)
    assert a.seconds == 2.0
    a.drain(10)
    assert a.seconds == 0.0


def test_reset_clears_everything():
    a = acc(tolerance_seconds=100)
    a.tick(OFF)
    a.tick(TOLERATED)
    a.reset()
    assert a.seconds == 0.0
    assert a.tolerated_run == 0.0


def test_forgives_twice_as_fast():
    a = acc(patience_seconds=180)
    for _ in range(60):
        a.tick(OFF)
    assert a.seconds == 60.0
    for _ in range(29):
        a.tick(ON)
    assert a.seconds == 2.0
    a.tick(ON)
    assert a.seconds == 0.0


def test_dt_scales_both_directions():
    a = acc()
    a.tick(OFF, dt=0.5)
    a.tick(OFF, dt=0.5)
    assert a.seconds == 1.0
    a.tick(ON, dt=0.25)
    assert a.seconds == 0.5
