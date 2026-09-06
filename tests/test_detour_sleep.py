"""The deferred reminder survives sleep, lid-close and a restart, because the deadline lives in SQLite."""

from __future__ import annotations

import pytest

from anchor.db import Store
from anchor.engine import Engine
from anchor.llm import FakeLLM
from anchor.models import Intent, Observed, State
from anchor.sensors import FakeSensor
from anchor.statemachine import Conversation
from anchor.voice import FakeListener, FakeSpeaker
from anchor import roast as R
from tests.conftest import frame


def build(settings, store, clock, replies):
    sensor = FakeSensor([frame()])
    speaker, listener = FakeSpeaker(), FakeListener(list(replies))
    eng = Engine(settings, store, sensor, FakeLLM(), speaker, listener, clock)
    return eng, sensor, speaker, listener


def start_detour(eng, minutes_reply="twenty minutes"):
    eng.startup()
    conv = eng.conv
    out = conv.confront(Observed(activity="scrolling reddit", app="Google Chrome", title="r/all", url_domain="reddit.com", minutes_off_task=9))
    assert out.intent == Intent.DEFER and out.minutes == 20
    return conv


def test_detour_pauses_everything(settings, store, clock):
    eng, sensor, speaker, _ = build(settings, store, clock, ["finish the assignment", "twenty minutes"])
    start_detour(eng)
    judge_calls_before = eng.judge_calls
    sample_calls = 0
    orig = sensor.sample

    def counting():
        nonlocal sample_calls
        sample_calls += 1
        return orig()

    sensor.sample = counting
    for _ in range(300):
        clock.advance(1)
        eng.tick()
    assert sample_calls == 0                    # no sensing during a sanctioned detour
    assert eng.judge_calls == judge_calls_before
    assert eng.conv.state == State.DETOUR


def test_reminder_fires_on_time_and_is_plain(settings, store, clock):
    eng, _, speaker, _ = build(settings, store, clock, ["finish the assignment", "twenty minutes", "yes"])
    conv = start_detour(eng)
    clock.advance(20 * 60 - 1)
    eng.tick()
    assert conv.state == State.DETOUR
    clock.advance(2)
    eng.tick()
    assert conv.state == State.WATCHING
    text, lang, tone = [c for c in speaker.calls if "You said" in c[0]][-1]
    assert text == R.detour_reminder(conv.anchor.verbatim, None, "en")
    assert conv.anchor.verbatim in text
    assert tone == "warm" and "Bold pivot" not in text
    assert speaker.calls[-1][0] == R.accept_line("en")          # "yes" → a plain acknowledgement, then watching
    assert store.active_anchor().detour_until is None
    # fires once only
    n = len(speaker.calls)
    for _ in range(5):
        clock.advance(1)
        eng.tick()
    assert len(speaker.calls) == n


def test_laptop_sleeps_through_the_deadline_and_reminder_says_it_is_late(settings, store, clock):
    eng, _, speaker, _ = build(settings, store, clock, ["finish the assignment", "twenty minutes", ""])
    conv = start_detour(eng)
    clock.advance(3 * 3600)                     # lid closed: no ticks happened for three hours
    eng.tick()
    text = speaker.calls[-1][0]
    assert text == R.detour_reminder(conv.anchor.verbatim, 160, "en")
    assert "late" in text.lower()
    assert conv.state == State.WATCHING


def test_quit_and_relaunch_during_detour_fires_on_next_launch(settings, store, clock):
    eng, _, speaker, _ = build(settings, store, clock, ["finish the assignment", "twenty minutes"])
    conv = start_detour(eng)
    anchor_id = conv.anchor.id
    store.close()
    clock.advance(45 * 60)                      # app was quit; 45 minutes pass
    store2 = Store(settings.db_path).open()
    eng2, _, speaker2, _ = build(settings, store2, clock, [""])
    eng2.startup()
    assert eng2.conv.anchor.id == anchor_id
    assert eng2.conv.state == State.DETOUR
    eng2.tick()
    text = speaker2.calls[-1][0]
    assert text == R.detour_reminder(eng2.conv.anchor.verbatim, 25, "en")
    assert eng2.conv.state == State.WATCHING
    assert store2.active_anchor().detour_until is None


def test_relaunch_before_deadline_keeps_waiting(settings, store, clock):
    eng, _, _, _ = build(settings, store, clock, ["finish the assignment", "twenty minutes"])
    start_detour(eng)
    store.close()
    clock.advance(5 * 60)
    store2 = Store(settings.db_path).open()
    eng2, sensor2, speaker2, _ = build(settings, store2, clock, [""])
    eng2.startup()
    eng2.tick()
    assert eng2.conv.state == State.DETOUR
    assert not any("Time" in t for t, _, _ in speaker2.calls)
    clock.advance(15 * 60 + 1)
    eng2.tick()
    assert eng2.conv.state == State.WATCHING


def test_deadline_is_absolute_in_sqlite(settings, store, clock):
    eng, _, _, _ = build(settings, store, clock, ["finish the assignment", "half an hour"])
    eng.startup()
    eng.conv.confront(Observed(activity="scrolling reddit", minutes_off_task=5))
    row = store.conn.execute("SELECT detour_until, detour_minutes FROM anchors WHERE status='active'").fetchone()
    assert row[0] == pytest.approx(clock.now() + 30 * 60)
    assert row[1] == 30


def test_reminder_reply_can_extend_or_finish(settings, store, clock):
    eng, _, speaker, _ = build(settings, store, clock, ["finish the assignment", "twenty minutes", "ten more minutes"])
    conv = start_detour(eng)
    clock.advance(20 * 60 + 1)
    eng.tick()
    assert conv.state == State.DETOUR                      # asked for more time at the reminder
    assert conv.anchor.detour_until == pytest.approx(clock.now() + 10 * 60)


def test_hindi_reminder_is_native(settings, store, clock):
    eng, _, speaker, _ = build(settings, store, clock, ["मुझे आज रात असाइनमेंट खत्म करना है", "बीस मिनट", ""])
    conv = start_detour(eng)
    clock.advance(20 * 60 + 1)
    eng.tick()
    text, lang, tone = speaker.calls[-1]
    assert lang == "hi" and any("ऀ" <= ch <= "ॿ" for ch in text)
    assert conv.anchor.verbatim in text
