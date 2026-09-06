"""The 1 Hz loop: free when nothing changed, cached when seen before, vision only when unsure, never nagging."""

from __future__ import annotations

import pytest

from anchor.engine import Engine
from anchor.llm import FakeLLM
from anchor.models import Blockers, Intent, Policy, State, Verdict
from anchor.sensors import FakeSensor
from anchor.voice import FakeListener, FakeSpeaker
from tests.conftest import frame

ON = Verdict(drift=0.05, confidence=0.95, reason="editing the assignment", activity="editing the assignment in Word")
OFF = Verdict(drift=0.95, confidence=0.95, reason="crabs", activity="researching whether crabs can swim on Google")
AMBIG = Verdict(drift=0.5, confidence=0.4, reason="unclear", activity="")


def build(settings, store, clock, replies=("finish the assignment tonight",), verdicts=(), start_frame=None):
    sensor = FakeSensor([start_frame or frame(app="Microsoft Word", title="assignment.docx", domain="", dhash="0" * 16)])
    llm = FakeLLM(verdicts=list(verdicts))
    speaker, listener = FakeSpeaker(), FakeListener(list(replies))
    eng = Engine(settings, store, sensor, llm, speaker, listener, clock)
    eng.startup()
    return eng, sensor, llm, speaker, listener


def run(eng, clock, seconds):
    for _ in range(seconds):
        clock.advance(1)
        eng.tick()


def test_unchanged_screen_costs_nothing(settings, store, clock):
    eng, sensor, llm, _, _ = build(settings, store, clock, verdicts=[ON])
    run(eng, clock, 60)
    judge_calls = [c for c in llm.calls if c[0] == "judge"]
    assert len(judge_calls) == 1
    assert eng.conv.state == State.WATCHING


def test_cache_hit_avoids_second_judgement(settings, store, clock):
    eng, sensor, llm, _, _ = build(settings, store, clock, verdicts=[ON, OFF, ON])
    run(eng, clock, 3)
    sensor.set(app="Google Chrome", title="crabs can swim? - Google Search", url_domain="google.com", dhash="f" * 16)
    run(eng, clock, 3)
    sensor.set(app="Microsoft Word", title="assignment.docx", url_domain="", dhash="0" * 16)
    run(eng, clock, 3)
    assert len([c for c in llm.calls if c[0] == "judge"]) == 2
    assert eng.cache_hits == 1


def test_single_moment_does_not_trigger_and_return_drains_twice_as_fast(settings, store, clock):
    eng, sensor, llm, speaker, _ = build(settings, store, clock, verdicts=[ON, OFF, ON])
    run(eng, clock, 2)
    sensor.set(app="Google Chrome", title="crabs can swim? - Google Search", url_domain="google.com", dhash="f" * 16)
    run(eng, clock, 30)
    assert eng.drift.seconds == pytest.approx(30)
    assert eng.conv.state == State.DRIFTING
    sensor.set(app="Microsoft Word", title="assignment.docx", url_domain="", dhash="0" * 16)
    run(eng, clock, 15)
    assert eng.drift.seconds == 0
    assert eng.conv.state == State.WATCHING
    assert not any(c[2] == "roast" for c in speaker.calls)


def test_waits_for_a_pause_then_roasts_once(settings, store, clock):
    eng, sensor, llm, speaker, listener = build(
        settings, store, clock, replies=("finish the assignment tonight", "twenty minutes"), verdicts=[ON] + [OFF] * 30
    )
    patience = eng.conv.policy.patience_seconds
    run(eng, clock, 2)
    # typing constantly on a page whose pixels keep changing: no pause, no interruption even past patience
    tick = 0
    sensor.set(app="Google Chrome", title="crabs can swim? - Google Search", url_domain="google.com", dhash="f" * 16)
    for _ in range(patience + 30):
        tick += 1
        sensor.set(idle_s=0.2, dhash=("f" * 16) if tick % 2 else ("0" * 16))   # screen never still
        clock.advance(1)
        eng.tick()
    assert eng.drift.fired
    assert not [c for c in speaker.calls if c[2] == "roast"]
    assert eng.vision_calls <= (patience + 30) // settings.vision_cooldown_s + 2   # a video is not a new context every second
    # now the user stops typing → natural gap → one confrontation
    sensor.set(idle_s=5.0)
    run(eng, clock, 2)
    roasts = [c for c in speaker.calls if c[2] == "roast"]
    assert len(roasts) == 1
    assert "crabs" in roasts[0][0].lower()
    assert eng.conv.state == State.DETOUR
    assert eng.drift.seconds == 0


def test_hard_cap_catches_endless_scrolling(settings, store, clock):
    eng, sensor, llm, speaker, _ = build(
        settings, store, clock, replies=("finish the assignment tonight", "twenty minutes"), verdicts=[ON, OFF] + [OFF] * 50
    )
    patience = eng.conv.policy.patience_seconds
    run(eng, clock, 1)
    sensor.set(app="Google Chrome", title="r/all", url_domain="reddit.com", idle_s=0.1, dhash="f" * 16)
    n = 0
    while n < 2 * patience + 5 and eng.conv.state != State.DETOUR:
        n += 1
        sensor.set(title=f"r/all page {n}", dhash=("f" * 16) if n % 2 else ("0" * 16))   # never still, never idle
        clock.advance(1)
        eng.tick()
    assert eng.conv.state == State.DETOUR
    assert n >= 2 * patience - 2


def test_blockers_suppress_speech(settings, store, clock):
    eng, sensor, llm, speaker, _ = build(
        settings, store, clock, replies=("finish the assignment tonight", "twenty minutes"), verdicts=[ON, OFF]
    )
    patience = eng.conv.policy.patience_seconds
    run(eng, clock, 1)
    sensor.set(app="Google Chrome", title="crabs can swim? - Google Search", url_domain="google.com", idle_s=9.0, dhash="f" * 16)
    sensor.set_blockers(mic_in_use=True)
    run(eng, clock, patience + 5)
    assert not any(c[2] == "roast" for c in speaker.calls)
    sensor.set_blockers(mic_in_use=False, screen_locked=True)
    run(eng, clock, 5)
    assert not any(c[2] == "roast" for c in speaker.calls)
    sensor.set_blockers(screen_locked=False, do_not_disturb=True)
    run(eng, clock, 5)
    assert not any(c[2] == "roast" for c in speaker.calls)
    eng.set_muted(True)
    sensor.set_blockers(do_not_disturb=False)
    run(eng, clock, 5)
    assert speaker.calls == [] or not any(c[2] == "roast" for c in speaker.calls)
    eng.set_muted(False)
    run(eng, clock, 3)
    assert len([c for c in speaker.calls if c[2] == "roast"]) == 1


def test_vision_only_when_unsure(settings, store, clock):
    eng, sensor, llm, _, _ = build(settings, store, clock, verdicts=[ON, AMBIG, OFF, ON, ON])
    run(eng, clock, 2)
    assert eng.vision_calls == 0
    # ambiguous text verdict → escalate to vision (same fingerprint, image attached)
    sensor.set(app="Google Chrome", title="Some page", url_domain="example.com", dhash="f" * 16)
    run(eng, clock, 2)
    assert eng.vision_calls == 1
    judge_calls = [c for c in llm.calls if c[0] == "judge"]
    assert judge_calls[-1][1]["has_image"] is True
    # pixels diverge while the title stays: forced look
    sensor.set(dhash="0" * 16)
    run(eng, clock, 2)
    assert eng.vision_calls == 2
    # uninformative title → vision directly
    sensor.set(app="Safari", title="New Tab", url_domain="", dhash="1" * 16)
    run(eng, clock, 2)
    assert eng.vision_calls == 3
    assert sensor.grab_calls == 3


def test_screenshot_never_persisted(settings, store, clock):
    eng, sensor, llm, _, _ = build(settings, store, clock, verdicts=[ON, AMBIG, OFF])
    run(eng, clock, 1)
    sensor.set(app="Safari", title="New Tab", url_domain="", dhash="f" * 16)
    run(eng, clock, 2)
    assert eng.vision_calls >= 1
    cols = [r[1] for r in store.conn.execute("PRAGMA table_info(verdicts)")]
    assert not any("image" in c or "jpeg" in c or "screenshot" in c for c in cols)
    raw = settings.db_path.read_bytes()
    assert sensor.jpeg not in raw
    assert eng.last_frame.image_jpeg is None


def test_heuristic_fallback_when_model_is_down(settings, store, clock):
    eng, sensor, llm, speaker, _ = build(settings, store, clock, verdicts=[])   # FakeLLM returns None for everything
    run(eng, clock, 2)
    assert eng.last_verdict is not None and eng.last_verdict.source == "heuristic"
    sensor.set(app="Google Chrome", title="Lo-fi beats - YouTube", url_domain="youtube.com", dhash="f" * 16)
    run(eng, clock, 2)
    assert eng.last_verdict.off_task


def test_expected_surface_is_tolerated_for_a_while(settings, store, clock):
    eng, sensor, llm, speaker, _ = build(
        settings, store, clock, replies=("fix the login bug in the python backend", "ten minutes"),
        verdicts=[ON, Verdict(drift=0.3, confidence=0.9, reason="stackoverflow", tolerated=True, activity="reading Stack Overflow")],
    )
    run(eng, clock, 1)
    sensor.set(app="Google Chrome", title="python KeyError - Stack Overflow", url_domain="stackoverflow.com", idle_s=9, dhash="f" * 16)
    run(eng, clock, 300)
    assert eng.drift.seconds == 0                                    # within tolerance: part of coding
    run(eng, clock, eng.conv.policy.tolerance_seconds + eng.conv.policy.patience_seconds)
    assert any(c[2] == "roast" for c in speaker.calls)                # eventually it stops being "part of it"


def test_wrong_call_writes_refinement_and_flips(settings, store, clock):
    eng, sensor, llm, speaker, _ = build(settings, store, clock, verdicts=[ON, OFF, OFF])
    run(eng, clock, 1)
    sensor.set(app="Google Chrome", title="crabs can swim? - Google Search", url_domain="google.com", dhash="f" * 16)
    run(eng, clock, 20)
    assert eng.drift.seconds > 0
    eng.wrong_call()
    assert eng.drift.seconds == 0
    refs = store.refinements_since(0)
    assert len(refs) == 1 and "WRONG" in refs[0]
    assert eng.last_verdict.off_task is False
    # the next judgement for a new context carries the refinement
    sensor.set(title="crab facts - Google Search", dhash="e" * 16)
    run(eng, clock, 2)
    last = [c for c in llm.calls if c[0] == "judge"][-1][1]
    assert last["refinements"]


def test_hourly_ceiling_never_more_than_four(settings, store, clock):
    eng, sensor, llm, speaker, _ = build(
        settings, store, clock,
        replies=("finish the assignment tonight",) + ("no",) * 40,
        verdicts=[ON] + [OFF] * 40,
    )
    patience = eng.conv.policy.patience_seconds
    run(eng, clock, 1)
    llm.replies = []
    for i in range(12):
        sensor.set(app="Google Chrome", title=f"reddit {i}", url_domain="reddit.com", idle_s=9, dhash=("f" * 16) if i % 2 else ("0" * 16))
        run(eng, clock, patience + 2)
        if eng.conv.state == State.DETOUR:
            eng.conv.anchor.detour_until = None
            store.set_detour(eng.conv.anchor.id, None)
            eng.conv.state = State.WATCHING
    roasts = [c for c in speaker.calls if c[2] in ("roast", "warm") and "?" in c[0] and "break" in c[0].lower()]
    assert 1 <= len(roasts) <= 4


def test_missing_permission_is_spoken_not_fatal(settings, store, clock):
    sensor = FakeSensor([frame(app="Code", title="", domain="", title_available=False)])
    sensor.missing = ["Accessibility"]
    speaker, listener = FakeSpeaker(), FakeListener(["finish the assignment tonight"])
    eng = Engine(settings, store, sensor, FakeLLM(), speaker, listener, clock)
    eng.startup()
    assert any("Accessibility" in t for t, _, _ in speaker.calls)
    run(eng, clock, 3)
    assert eng.conv.state in (State.WATCHING, State.DRIFTING)
    assert eng.status().missing_permissions == ["Accessibility"]


def test_unanswered_intake_backs_off(settings, store, clock):
    sensor = FakeSensor([frame()])
    speaker, listener = FakeSpeaker(), FakeListener([])          # nobody ever answers
    eng = Engine(settings, store, sensor, FakeLLM(), speaker, listener, clock)
    eng.startup()
    asks = lambda: len([t for t, _, _ in speaker.calls if t.endswith("One sentence is enough.")])
    assert asks() == 1 and eng.conv.state == State.IDLE
    run(eng, clock, 31)
    assert asks() == 2                                             # one retry after 30 s
    run(eng, clock, 200)
    assert asks() == 2                                             # then quiet for five minutes
    run(eng, clock, 100)
    assert asks() == 3
    listener.replies = ["finish the assignment tonight"]           # someone finally answers
    run(eng, clock, 301)
    assert eng.conv.state in (State.WATCHING, State.DRIFTING) and asks() == 4


def test_demo_script_leetcode_then_game(settings, store, clock):
    """The stage script: a minute on LeetCode, then a game → spoken within 20 s, sooner at the first pause."""
    settings.demo = True
    GAME = Verdict(drift=0.99, confidence=0.99, reason="poki", activity="playing Drive Mad on poki.com")
    LEET = Verdict(drift=0.03, confidence=0.98, reason="leetcode", activity="solving Two Sum on leetcode.com")
    eng, sensor, llm, speaker, _ = build(
        settings, store, clock, replies=("solving a DSA question on leetcode", "twenty minutes"),
        verdicts=[LEET, AMBIG] + [GAME] * 20,
        start_frame=frame(app="Google Chrome", title="Two Sum - LeetCode", domain="leetcode.com", idle=0.5, dhash="a" * 16),
    )
    assert eng.conv.policy.patience_seconds == 10
    run(eng, clock, 60)                                                    # a minute of honest work
    assert eng.drift.seconds == 0 and not speaker.calls[-1][2] == "roast"
    sensor.set(title="New Tab", url_domain="newtab", dhash="b" * 16)       # opening the tab: ambiguous, holds
    run(eng, clock, 2)
    sensor.set(title="Drive Mad - Play online for free! | Poki", url_domain="poki.com", idle_s=0.1, dhash="c" * 16)
    n = 0
    while eng.conv.state != State.DETOUR and n < 30:
        n += 1
        sensor.set(dhash=("c" * 16) if n % 2 else ("d" * 16))              # game keeps animating, hands never idle
        clock.advance(1)
        eng.tick()
    assert eng.conv.state == State.DETOUR
    assert 10 <= n <= 20                                                   # hard cap: 2 × patience, no pause needed
    roast = [c for c in speaker.calls if c[2] == "roast"][0][0]
    assert "drive mad" in roast.lower() or "poki" in roast.lower()
    assert "1 minute" in roast or "minute" in roast.lower()                # names the elapsed time, never "0 minutes"


def test_status_dots(settings, store, clock):
    eng, sensor, llm, speaker, _ = build(settings, store, clock, verdicts=[ON, OFF])
    run(eng, clock, 1)
    assert eng.status().dot == "🟢"
    sensor.set(app="Google Chrome", title="r/all", url_domain="reddit.com", dhash="f" * 16)
    run(eng, clock, 3)
    assert eng.status().dot == "🟠"
    eng.set_muted(True)
    assert eng.status().dot == "🔇"
