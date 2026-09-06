"""Goal matrix: adversarial tests for the goal.md items the other suites left unproven.

Each test names the goal item(s) it proves. Everything is hermetic: FakeClock, a temp SQLite file,
scripted model / voice / sensor doubles, and the roast precompose run synchronously.
"""

from __future__ import annotations

import threading

import pytest

from anchor import roast as R
from anchor.db import Store
from anchor.engine import Engine
from anchor.llm import FakeLLM
from anchor.models import ComposeResult, Intent, Observed, Policy, PolicyDraft, Register, ReplyIntent, State, Verdict
from anchor.sensors import FakeSensor
from anchor.statemachine import ACK_LINE, HOW_LONG_LINE, Conversation
from anchor.voice import FakeListener, FakeSpeaker
from tests.conftest import frame

ROAST_TONES = {"roast", "deadpan", "mock_respect", "disbelief"}
ANCHOR = "finish the assignment tonight"
HI_ANCHOR = "मुझे आज रात असाइनमेंट खत्म करना है"
ON = Verdict(drift=0.05, confidence=0.95, reason="editing", activity="editing the assignment in Word")
OFF = Verdict(drift=0.95, confidence=0.95, reason="crabs", activity="researching whether crabs can swim on Google")
OFF_HI = Verdict(drift=0.95, confidence=0.95, reason="crabs", activity="गूगल पर केकड़ों के बारे में पढ़ना")
WORK = dict(app="Microsoft Word", title="assignment.docx", url_domain="", dhash="0" * 16)
CRABS = dict(app="Google Chrome", title="crabs can swim? - Google Search", url_domain="google.com", dhash="f" * 16)


def devanagari(text: str) -> bool:
    return any("ऀ" <= ch <= "ॿ" for ch in text)


# ----------------------------------------------------------------------------- helpers

def make_conv(settings, store, clock, replies, llm=None, muted=False):
    speaker, listener = FakeSpeaker(), FakeListener(list(replies))
    conv = Conversation(settings, store, llm or FakeLLM(), speaker, listener, clock, is_muted=lambda: muted)
    return conv, speaker, listener


def anchored(settings, store, clock, sentence=ANCHOR, replies=(), llm=None):
    conv, speaker, listener = make_conv(settings, store, clock, [sentence] + list(replies), llm=llm)
    conv.start_session()
    return conv, speaker, listener


def observed(minutes=12, confidence=0.95):
    return Observed(activity="researching whether crabs can swim on Google", app="Google Chrome",
                    title="crabs can swim? - Google Search", url_domain="google.com",
                    minutes_off_task=minutes, confidence=confidence)


def clear_detour(conv, store):
    if conv.anchor is not None and conv.anchor.detour_until is not None:
        conv.anchor.detour_until = None
        store.set_detour(conv.anchor.id, None)
        conv.state = State.WATCHING


def build(settings, store, clock, replies=(ANCHOR,), verdicts=(), llm=None, speaker=None, listener=None):
    sensor = FakeSensor([frame(app="Microsoft Word", title="assignment.docx", domain="", dhash="0" * 16)])
    llm = llm or FakeLLM(verdicts=list(verdicts))
    speaker = speaker or FakeSpeaker()
    listener = listener or FakeListener(list(replies))
    eng = Engine(settings, store, sensor, llm, speaker, listener, clock)
    eng.precompose_async = False
    eng.startup()
    return eng, sensor, llm, speaker, listener


def go(sensor, where, **more):
    sensor.set(**where, **more)


def run(eng, clock, seconds):
    for _ in range(int(seconds)):
        clock.advance(1)
        eng.tick()


def events(store):
    return list(reversed(store.recent_events(100000)))       # oldest first


def compose_indices(llm, start=0):
    return {c[1]["repeat_index"] for c in llm.calls[start:] if c[0] == "compose"}


class Killed(BaseException):
    """The process died mid-call (SIGKILL, power loss). Not an Exception, so nothing in the app catches it."""


class RoutedListener(FakeListener):
    """Scripted replies first; afterwards it answers by what was just said: the choice question gets
    ``on_choice``, everything else silence. ``last`` is wired to the conversation's last line."""

    def __init__(self, replies, on_choice="two minutes"):
        super().__init__(replies)
        self.on_choice = on_choice
        self.last = lambda: ""

    def listen(self, *, window_s=8.0, language="en"):
        if self.replies:
            return super().listen(window_s=window_s, language=language)
        self.calls.append((window_s, language))
        return self.on_choice if self.last().endswith(R.choice_line("en")) else ""


class DiesListening(FakeListener):
    """Raises ``Killed`` when asked to listen right after ``after_line`` was spoken."""

    def __init__(self, replies, after_line):
        super().__init__(replies)
        self.after_line = after_line
        self.last = lambda: ""

    def listen(self, *, window_s=8.0, language="en"):
        if self.last() == self.after_line:
            raise Killed()
        return super().listen(window_s=window_s, language=language)


class DiesSpeaking(FakeSpeaker):
    """Raises ``Killed`` when asked to speak a line starting with ``prefix``."""

    def __init__(self, prefix):
        super().__init__()
        self.prefix = prefix

    def speak(self, text, *, language="en", tone="neutral"):
        if text.startswith(self.prefix):
            raise Killed()
        super().speak(text, language=language, tone=tone)


class MutesOnReply(FakeListener):
    """The user hits Mute while the app is listening for their reply to the roast."""

    def __init__(self, replies, trigger):
        super().__init__(replies)
        self.trigger = trigger
        self.engine = None

    def listen(self, *, window_s=8.0, language="en"):
        reply = super().listen(window_s=window_s, language=language)
        if reply == self.trigger and self.engine is not None:
            self.engine.set_muted(True)
        return reply


# ----------------------------------------------------------------------------- 18 / 24: one push-back, every order

@pytest.mark.parametrize("replies, intent", [
    (["you're just a bot", ""], Intent.DEFER),                       # jab, then silence
    (["you're just a bot", "hmm"], Intent.DEFER),                    # jab, then a dodge
    (["you're just a bot", "stupid bot"], Intent.DEFER),             # jab, then another jab
    (["", "you're just a bot"], Intent.DEFER),                       # silence, then a jab
    (["hmm", "stop"], Intent.RESUME),                                # dodge, then a stop word
    (["hmm", "you're just a bot, twenty minutes"], Intent.DEFER),    # dodge, then a jab that carries an answer
    (["whatever", "this is the new main thing, I'm learning guitar"], Intent.SWITCH),
    (["meh", "it's done"], Intent.DONE),
])
def test_one_pushback_whatever_the_reply_order(settings, store, clock, replies, intent):
    llm = FakeLLM(replies=[ReplyIntent(Intent.EVASIVE)] * 10)   # a model that always says "dodge" cannot force a second push
    extra = ["writing the lab report"] if intent == Intent.DONE else []
    conv, speaker, listener = anchored(settings, store, clock, replies=replies + extra, llm=llm)
    out = conv.confront(observed())
    assert out.pushed is True and out.intent == intent
    flat = [c for c in speaker.calls if c[2] == "flat"]
    assert flat == [(R.pushback_line(ANCHOR, "en"), "en", "flat")]              # exactly one, verbatim, no joke
    comebacks = [c for c in speaker.calls if c[2] == "deadpan" and "?" not in c[0]]
    assert len(comebacks) <= 1
    assert len(listener.calls) == 1 + 2 + len(extra)                            # intake, roast reply, push reply: never a third
    assert store.count_events("PUSHED", 0) == 1
    assert store.count_events("ROAST", 0) == 1                                  # once per interruption, never twice


def test_stop_after_the_pushback_is_accepted_and_cools_the_register(settings, store, clock):
    conv, speaker, listener = anchored(settings, store, clock, replies=["hmm", "stop"])
    out = conv.confront(observed())
    assert out.intent == Intent.RESUME and out.pushed is True
    assert speaker.calls[-1][0] == R.accept_line("en")                          # no farewell, no argument
    assert conv.state == State.WATCHING and conv.anchor.detour_until is None
    assert conv.current_register() == Register.DRY
    assert len(listener.calls) == 3


def test_stop_inside_a_jab_means_stop(settings, store, clock):
    conv, speaker, listener = anchored(settings, store, clock, replies=["shut up bot"])
    out = conv.confront(observed())
    assert out.intent == Intent.RESUME and out.pushed is False
    assert not [c for c in speaker.calls if c[2] == "flat"]
    assert not [c for c in speaker.calls if c[2] == "deadpan" and "?" not in c[0]]
    assert conv.current_register() == Register.DRY and conv.state == State.WATCHING
    assert len(listener.calls) == 2


# ----------------------------------------------------------------------------- 4: one clarifying question, ever

def test_second_clarification_is_never_asked_even_if_the_model_wants_one(settings, store, clock):
    def draft(question):
        return PolicyDraft(clarified_anchor="study", language="en", policy=Policy(),
                           needs_clarification=True, clarifying_question=question)

    llm = FakeLLM(policies=[draft("Which subject?"), draft("Which chapter?")])
    conv, speaker, listener = anchored(settings, store, clock, sentence="study", replies=["umm"], llm=llm)
    questions = [t for t, _, _ in speaker.calls if t.endswith("?") and "working on right now" not in t]
    assert questions == ["Which subject?"]
    assert len(listener.calls) == 2 and conv.state == State.WATCHING and conv.anchor.verbatim == "study"


# ----------------------------------------------------------------------------- 24: the hour ceiling across switch / relaunch / detours

def test_ceiling_counts_across_a_switch_and_a_relaunch(settings, store, clock):
    replies = ["ten minutes", "ten minutes", "ten minutes", "this is the new main thing, I'm building my resume now"]
    conv, _, _ = anchored(settings, store, clock, replies=replies)
    for _ in range(4):
        assert conv.may_confront()
        conv.confront(observed())
        clear_detour(conv, store)
        clock.advance(60)
    assert "resume" in conv.anchor.verbatim and conv.confrontations_this_anchor == 0
    assert conv.may_confront() is False                 # a fresh anchor does not reopen the hour
    conv2, _, _ = make_conv(settings, store, clock, [])
    conv2.start_session()
    assert conv2.may_confront() is False                # nor does a relaunch
    clock.advance(3600)
    assert conv2.may_confront() is True


def test_engine_four_per_hour_with_detours_and_never_inside_one(settings, store, clock):
    listener = RoutedListener([ANCHOR], on_choice="two minutes")
    eng, sensor, llm, speaker, _ = build(settings, store, clock, verdicts=[ON] + [OFF] * 200, listener=listener)
    listener.last = lambda: eng.conv.last_line
    patience = eng.conv.policy.patience_seconds
    run(eng, clock, 1)
    go(sensor, CRABS, idle_s=9)
    run(eng, clock, 3600 + patience + 400)
    log = events(store)
    conf = [e["ts"] for e in log if e["state"] == "CONFRONTING"]
    assert len(conf) >= 5                               # the ceiling reopens after an hour
    for t in conf:
        assert sum(1 for u in conf if t - 3600 < u <= t) <= 4
    inside = False
    for e in log:
        if e["state"] == "DETOUR":
            inside = True
        elif e["state"] == "WATCHING" and "detour over" in e["detail"]:
            inside = False
        elif e["state"] == "CONFRONTING":
            assert not inside                           # never during a break it agreed to
    reminders = [c for c in speaker.calls if c[0].startswith("Time's up")]
    over = sum(1 for e in log if e["state"] == "WATCHING" and "detour over" in e["detail"])
    assert len(reminders) == over >= 4                  # every finished break ended with exactly one reminder
    assert store.count_events("CEILING", 0) >= 1
    assert not [c for c in speaker.calls if c[2] == "flat"]


def test_reminder_is_not_subject_to_the_roast_cooldown(settings, store, clock):
    settings.roast_cooldown_s = 3600
    eng, sensor, _, speaker, _ = build(settings, store, clock, replies=(ANCHOR, "two minutes", ""), verdicts=[ON, OFF])
    patience = eng.conv.policy.patience_seconds
    run(eng, clock, 1)
    go(sensor, CRABS, idle_s=9)
    run(eng, clock, patience + 2)
    assert eng.conv.state == State.DETOUR
    run(eng, clock, 121)
    assert speaker.calls[-1][0] == R.detour_reminder(ANCHOR, None, "en")
    assert eng.conv.state in (State.WATCHING, State.DRIFTING)           # watching again (and still on the crabs)


# ----------------------------------------------------------------------------- 20: the deadline survives a kill mid-question

def test_detour_survives_a_kill_during_the_how_long_question(settings, store, clock):
    listener = DiesListening([ANCHOR, "I need a break"], after_line=HOW_LONG_LINE["en"])
    eng, sensor, _, speaker, _ = build(settings, store, clock, verdicts=[ON, OFF], listener=listener)
    listener.last = lambda: eng.conv.last_line
    patience = eng.conv.policy.patience_seconds
    run(eng, clock, 1)
    go(sensor, CRABS, idle_s=9)
    with pytest.raises(Killed):
        run(eng, clock, patience + 2)
    assert speaker.calls[-1][0] == HOW_LONG_LINE["en"]
    store.close()
    clock.advance(40 * 60)                              # quit mid-question; relaunched forty minutes later
    store2 = Store(settings.db_path).open()
    eng2, _, _, speaker2, _ = build(settings, store2, clock, replies=[""])
    assert eng2.conv.state == State.DETOUR              # the break was agreed; only its length went unsaid
    eng2.tick()
    text = speaker2.calls[-1][0]
    assert text == R.detour_reminder(ANCHOR, 25, "en") and "late" in text.lower()   # default 15 min, now 25 late
    assert eng2.conv.state == State.WATCHING and store2.active_anchor().detour_until is None
    n = len(speaker2.calls)
    run(eng2, clock, 3)
    assert len(speaker2.calls) == n                     # once


def test_how_long_answer_reaches_disk_before_it_is_confirmed(settings, store, clock):
    speaker = DiesSpeaking(prefix="Okay, 2 minutes")
    eng, sensor, _, _, _ = build(settings, store, clock, replies=(ANCHOR, "I need a break", "two minutes"),
                                 verdicts=[ON, OFF], speaker=speaker)
    patience = eng.conv.policy.patience_seconds
    run(eng, clock, 1)
    go(sensor, CRABS, idle_s=9)
    with pytest.raises(Killed):
        run(eng, clock, patience + 2)
    store.close()
    store2 = Store(settings.db_path).open()
    eng2, _, _, speaker2, _ = build(settings, store2, clock, replies=[""])
    assert eng2.conv.state == State.DETOUR
    assert eng2.conv.anchor.detour_until == pytest.approx(clock.now() + 120, abs=2)   # the answer, not the default
    run(eng2, clock, 121)
    assert speaker2.calls[-1][0] == R.detour_reminder(ANCHOR, None, "en") and eng2.conv.state == State.WATCHING


# ----------------------------------------------------------------------------- 12 / 20: the reminder is speech too

@pytest.mark.parametrize("blocker", ["mic_in_use", "do_not_disturb", "screen_locked", "muted"])
def test_reminder_waits_out_a_call_focus_lock_or_mute_then_says_it_is_late(settings, store, clock, blocker):
    eng, sensor, _, speaker, listener = build(settings, store, clock, replies=(ANCHOR, "two minutes", ""), verdicts=[ON, OFF])
    patience = eng.conv.policy.patience_seconds
    run(eng, clock, 1)
    go(sensor, CRABS, idle_s=9)
    run(eng, clock, patience + 2)
    assert eng.conv.state == State.DETOUR
    n_spoken, n_heard = len(speaker.calls), len(listener.calls)
    if blocker == "muted":
        eng.set_muted(True)
    else:
        sensor.set_blockers(**{blocker: True})
    run(eng, clock, 120 + 180)                          # the deadline passes; three minutes go by, still blocked
    assert len(speaker.calls) == n_spoken and len(listener.calls) == n_heard
    assert eng.conv.state == State.DETOUR
    if blocker == "muted":
        eng.set_muted(False)
    else:
        sensor.set_blockers(**{blocker: False})
    run(eng, clock, 1)
    text = speaker.calls[-1][0]
    assert text == R.detour_reminder(ANCHOR, 3, "en") and "late" in text.lower()
    assert eng.conv.state == State.WATCHING and eng.conv.anchor.detour_until is None
    assert [e for e in events(store) if e["state"] == "SUPPRESSED" and "reminder waiting" in e["detail"]]


# ----------------------------------------------------------------------------- 17: repeat decay resets per anchor

def test_repeat_index_resets_on_switch_and_done_but_survives_relaunch(settings, store, clock):
    replies = ["ten minutes", "this is the new main thing, I'm building my resume now", "ten minutes",
               "it's done", "writing the lab report", "ten minutes"]
    llm = FakeLLM()
    conv, _, _ = anchored(settings, store, clock, replies=replies, llm=llm)
    n = len(llm.calls)
    conv.confront(observed())
    assert compose_indices(llm, n) == {0}
    clear_detour(conv, store)
    n = len(llm.calls)
    out = conv.confront(observed())
    assert compose_indices(llm, n) == {1} and out.intent == Intent.SWITCH
    assert conv.confrontations_this_anchor == 0
    n = len(llm.calls)
    conv.confront(observed())
    assert compose_indices(llm, n) == {0}               # a new anchor gets the full-length joke again
    clear_detour(conv, store)
    n = len(llm.calls)
    out = conv.confront(observed())
    assert compose_indices(llm, n) == {1} and out.intent == Intent.DONE
    assert conv.anchor.verbatim == "writing the lab report" and conv.confrontations_this_anchor == 0
    n = len(llm.calls)
    conv.confront(observed())
    assert compose_indices(llm, n) == {0}
    clear_detour(conv, store)
    # a relaunch on the same anchor rebuilds the count from the log: the decay continues, it does not restart
    conv2, _, _ = make_conv(settings, store, clock, ["ten minutes"], llm=llm)
    conv2.start_session()
    assert conv2.confrontations_this_anchor == 1
    n = len(llm.calls)
    conv2.confront(observed())
    assert compose_indices(llm, n) == {1}


# ----------------------------------------------------------------------------- 12 / 16 / 24: blocked means blocked, timer full or not

@pytest.mark.parametrize("blocker", ["mic_in_use", "do_not_disturb", "screen_locked", "muted"])
def test_full_drift_timer_never_confronts_while_blocked(settings, store, clock, blocker):
    eng, sensor, _, speaker, listener = build(settings, store, clock, replies=(ANCHOR, "two minutes"), verdicts=[ON, OFF])
    patience = eng.conv.policy.patience_seconds
    run(eng, clock, 1)
    if blocker == "muted":
        eng.set_muted(True)
    else:
        sensor.set_blockers(**{blocker: True})
    go(sensor, CRABS, idle_s=9)
    run(eng, clock, 3 * patience)                       # far past the hard cap
    assert eng.drift.hard_cap
    assert len(listener.calls) == 1                     # never even listened for a reply
    assert store.count_events("CONFRONTING", 0) == 0 and store.count_events("SUPPRESSED", 0) >= 1
    assert not [c for c in speaker.calls if c[2] in ROAST_TONES or c[2] == "flat"]
    if blocker == "muted":
        eng.set_muted(False)
    else:
        sensor.set_blockers(**{blocker: False})
    run(eng, clock, 2)
    assert store.count_events("CONFRONTING", 0) == 1 and eng.conv.state == State.DETOUR


def test_full_drift_timer_never_confronts_during_a_detour(settings, store, clock):
    eng, sensor, _, speaker, listener = build(settings, store, clock, replies=(ANCHOR, "twenty minutes"), verdicts=[ON, OFF])
    patience = eng.conv.policy.patience_seconds
    run(eng, clock, 1)
    go(sensor, CRABS, idle_s=9)
    run(eng, clock, patience + 2)
    assert eng.conv.state == State.DETOUR
    eng.drift.seconds = 10 * patience                   # a full timer, however it got there
    samples, heard, spoken, conf = sensor.sample_calls, len(listener.calls), len(speaker.calls), store.count_events("CONFRONTING", 0)
    run(eng, clock, 19 * 60)
    assert eng.conv.state == State.DETOUR
    assert sensor.sample_calls == samples and len(listener.calls) == heard and len(speaker.calls) == spoken
    assert store.count_events("CONFRONTING", 0) == conf


# ----------------------------------------------------------------------------- 25: mute mid-confrontation

def test_mute_mid_confrontation_lets_the_utterance_finish_and_speaks_nothing_after(settings, store, clock):
    listener = MutesOnReply([ANCHOR, "hmm", "twenty minutes"], trigger="hmm")
    eng, sensor, _, speaker, _ = build(settings, store, clock, verdicts=[ON, OFF], listener=listener)
    listener.engine = eng
    patience = eng.conv.policy.patience_seconds
    run(eng, clock, 1)
    go(sensor, CRABS, idle_s=9)
    run(eng, clock, patience + 2)
    roasts = [c for c in speaker.calls if c[2] in ROAST_TONES]
    assert len(roasts) == 1 and speaker.calls[-1] == roasts[0]          # the roast was the last thing said out loud
    assert not [c for c in speaker.calls if c[2] == "flat"]
    assert store.count_events("PUSHED", 0) == 1                          # the flow still completed, silently
    assert any(line == f"anchor (muted): {R.pushback_line(ANCHOR, 'en')}" for line in eng.transcript)
    assert eng.conv.state == State.DETOUR and eng.conv.anchor.detour_minutes == 20
    n = len(speaker.calls)
    run(eng, clock, 25 * 60)                                             # still muted: the reminder waits
    assert len(speaker.calls) == n


# ----------------------------------------------------------------------------- 15: irritation is a one-way, per-day switch

def test_irritation_cools_for_the_day_survives_relaunch_and_resets_tomorrow(settings, store, clock):
    eng, sensor, _, speaker, _ = build(settings, store, clock, replies=(ANCHOR, "ugh, I'll go back to it"), verdicts=[ON] + [OFF] * 5)
    patience = eng.conv.policy.patience_seconds
    run(eng, clock, 1)
    go(sensor, CRABS, idle_s=9)
    run(eng, clock, patience + 2)
    assert [c for c in speaker.calls if c[2] in ROAST_TONES] and eng.conv.anchor.detour_until is None
    assert eng.conv.current_register() == Register.DRY
    store.close()
    clock.advance(30 * 60)
    store2 = Store(settings.db_path).open()                              # relaunch, same day
    eng2, sensor2, _, speaker2, _ = build(settings, store2, clock, replies=["haha fine, I'll go back to it"], verdicts=[OFF] * 5)
    assert eng2.conv.anchor.id == eng.conv.anchor.id and eng2.conv.current_register() == Register.DRY
    go(sensor2, CRABS, idle_s=9)
    run(eng2, clock, patience + 2)
    assert store2.count_events("CONFRONTING", 0) == 2
    assert not [c for c in speaker2.calls if c[2] in ROAST_TONES]         # cooled + repeat: no joke at all
    assert eng2.conv.current_register() == Register.DRY                  # an amused reply never warms it back up
    store2.close()
    clock.advance(24 * 3600)
    store3 = Store(settings.db_path).open()                              # a new day starts from the default again
    eng3, _, _, _, _ = build(settings, store3, clock, replies=[])
    assert eng3.conv.current_register() == Register.PLAYFUL


# ----------------------------------------------------------------------------- 23: Hindi end to end, nothing translated

def test_hindi_path_end_to_end_is_native(settings, store, clock):
    eng, sensor, _, speaker, _ = build(settings, store, clock, replies=(HI_ANCHOR, "हम्म", "बीस मिनट", ""), verdicts=[ON, OFF_HI])
    assert speaker.calls[-1] == (ACK_LINE["hi"], "hi", "warm")
    n0 = len(speaker.calls)
    patience = eng.conv.policy.patience_seconds
    run(eng, clock, 1)
    go(sensor, CRABS, idle_s=9)
    run(eng, clock, patience + 2)
    assert eng.conv.state == State.DETOUR and eng.conv.anchor.detour_minutes == 20
    run(eng, clock, 20 * 60 + 1)
    assert eng.conv.state in (State.WATCHING, State.DRIFTING) and eng.conv.anchor.detour_until is None
    lines = speaker.calls[n0:]
    assert len(lines) == 4                                               # roast+choice, push-back, confirm, reminder
    assert all(lang == "hi" and devanagari(text) for text, lang, _ in lines)
    english = ("You said", "Still true", "Okay", "Time's up", "How long", "Quick one", "short break", "Got it",
               "minute", "Heading back", "Bold pivot", "bold pivot")
    assert not any(marker in text for text, _, _ in lines for marker in english)
    roast, push, confirm, reminder = lines
    assert roast[2] in ROAST_TONES and "केकड़ों" in roast[0] and roast[0].endswith(R.choice_line("hi"))
    assert push == (R.pushback_line(HI_ANCHOR, "hi"), "hi", "flat")
    assert confirm[0].startswith("ठीक है, 20 मिनट") and confirm[2] == "warm"
    assert reminder == (R.detour_reminder(HI_ANCHOR, None, "hi"), "hi", "warm")


# ----------------------------------------------------------------------------- 25: "wrong call" means the roast

def test_wrong_call_right_after_a_confrontation_suppresses_the_repeat(settings, store, clock):
    eng, sensor, llm, _, _ = build(settings, store, clock, replies=(ANCHOR, "I will go back to it", "ten minutes"),
                                   verdicts=[ON, OFF] + [OFF] * 5)
    patience = eng.conv.policy.patience_seconds
    run(eng, clock, 1)
    go(sensor, CRABS, idle_s=9)
    run(eng, clock, patience + 2)
    assert store.count_events("CONFRONTING", 0) == 1 and eng.conv.anchor.detour_until is None
    eng.wrong_call()
    run(eng, clock, 2 * patience + 5)                                    # still on the same page, all day if they like
    assert store.count_events("CONFRONTING", 0) == 1 and eng.drift.seconds == 0
    refs = store.refinements_since(0)
    assert len(refs) == 1 and "google chrome" in refs[0].lower() and "on-task" in refs[0]
    go(sensor, dict(app="Google Chrome", title="crab facts - Google Search", url_domain="google.com", dhash="e" * 16), idle_s=9)
    run(eng, clock, 2)
    assert [c for c in llm.calls if c[0] == "judge"][-1][1]["refinements"] == refs


def test_wrong_call_after_going_back_to_work_still_means_the_roast(settings, store, clock):
    eng, sensor, _, _, _ = build(settings, store, clock, replies=(ANCHOR, "I will go back to it", "ten minutes"),
                                 verdicts=[ON, OFF] + [OFF] * 5)
    patience = eng.conv.policy.patience_seconds
    run(eng, clock, 1)
    go(sensor, CRABS, idle_s=9)
    run(eng, clock, patience + 2)
    assert store.count_events("CONFRONTING", 0) == 1
    go(sensor, WORK, idle_s=1)
    run(eng, clock, 5)                                                   # they do go back, and only then reach the menu
    eng.wrong_call()
    run(eng, clock, 60)
    assert eng.drift.seconds == 0 and eng.conv.state == State.WATCHING  # work is still work
    refs = store.refinements_since(0)
    assert len(refs) == 1 and "google chrome" in refs[0].lower() and "Treat it as on-task" in refs[0]
    go(sensor, CRABS, idle_s=9)
    run(eng, clock, 2 * patience + 5)
    assert store.count_events("CONFRONTING", 0) == 1                     # the corrected context is not re-called


# ----------------------------------------------------------------------------- 13 / 17: a stale prepared roast is never spoken

def test_stale_precomposed_roast_is_dropped_when_they_return_to_work(settings, store, clock):
    first = ComposeResult(roast="Three minutes into researching whether crabs can swim on Google. Bold pivot.",
                          choice_line="Short break, or the new main thing?")
    second = ComposeResult(roast="One minute back on the crabs on Google. Encore pivot.",
                           choice_line="Short break, or the new main thing?")
    llm = FakeLLM(verdicts=[ON] + [OFF] * 10, compositions=[first, second])
    eng, sensor, _, speaker, _ = build(settings, store, clock, replies=(ANCHOR, "ten minutes"), llm=llm)
    patience = eng.conv.policy.patience_seconds
    run(eng, clock, 1)
    go(sensor, CRABS, idle_s=0.2)
    n = 0
    for _ in range(patience // 2 + 2):                                   # typing the whole time: no gap, but past halfway
        n += 1
        sensor.set(dhash=("f" * 16) if n % 2 else ("e" * 16))
        clock.advance(1)
        eng.tick()
    assert eng._precomposed is not None and eng._precomposed[1].roast == first.roast
    go(sensor, WORK, idle_s=0.2)
    run(eng, clock, patience // 4 + 3)                                   # back to work: drained twice as fast
    assert eng.drift.seconds == 0
    assert eng._precomposed is None                                      # the prepared line is thrown away...
    assert first.roast not in eng.conv.composer.delivered                # ...and not remembered as "spoken"
    go(sensor, CRABS, idle_s=9)
    run(eng, clock, patience + 2)
    roasts = [c for c in speaker.calls if c[2] in ROAST_TONES]
    assert len(roasts) == 1 and roasts[0][0].startswith(second.roast)   # a fresh line for a fresh episode


def test_precompose_still_in_flight_when_they_return_is_discarded_too(settings, store, clock):
    eng, sensor, _, speaker, _ = build(settings, store, clock, replies=(ANCHOR, "ten minutes"), verdicts=[ON] + [OFF] * 10)
    eng.precompose_async = True
    gate = threading.Event()
    real = eng.conv.compose_for

    def slow_compose(observed_):
        gate.wait(5)
        return real(observed_)

    eng.conv.compose_for = slow_compose
    patience = eng.conv.policy.patience_seconds
    run(eng, clock, 1)
    go(sensor, CRABS, idle_s=0.2)
    n = 0
    for _ in range(patience // 2 + 2):
        n += 1
        sensor.set(dhash=("f" * 16) if n % 2 else ("e" * 16))
        clock.advance(1)
        eng.tick()
    thread = eng._precompose_thread
    assert thread is not None and thread.is_alive() and eng._precomposed is None
    go(sensor, WORK, idle_s=0.2)
    run(eng, clock, patience // 4 + 3)
    assert eng.drift.seconds == 0
    gate.set()
    thread.join(5)
    assert eng._precomposed is None                                      # a late result for a finished episode is dropped
    assert eng.conv.composer.delivered == []
