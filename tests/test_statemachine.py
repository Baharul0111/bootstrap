"""The protected part: one push-back, three branches, one clarifying question, ceilings."""

from __future__ import annotations

import pytest

from anchor.llm import FakeLLM
from anchor.models import Intent, Observed, Policy, PolicyDraft, Register, ReplyIntent, Sentiment, State
from anchor.statemachine import Conversation
from anchor.voice import FakeListener, FakeSpeaker
from anchor import roast as R


def make_conv(settings, store, clock, replies, llm=None, muted=False):
    speaker = FakeSpeaker()
    listener = FakeListener(list(replies))
    conv = Conversation(settings, store, llm or FakeLLM(), speaker, listener, clock, is_muted=lambda: muted)
    return conv, speaker, listener


def anchored(settings, store, clock, sentence="I need to finish the assignment tonight", replies=(), llm=None):
    conv, speaker, listener = make_conv(settings, store, clock, [sentence] + list(replies), llm=llm)
    conv.start_session()
    return conv, speaker, listener


def observed(minutes=12, confidence=0.95):
    return Observed(activity="researching whether crabs can swim on Google", app="Google Chrome",
                    title="crabs can swim? - Google Search", url_domain="google.com",
                    minutes_off_task=minutes, confidence=confidence)


# ----------------------------------------------------------------------------- intake

def test_intake_stores_verbatim_and_speaks_disclosure_once(settings, store, clock):
    conv, speaker, _ = anchored(settings, store, clock)
    assert conv.state == State.WATCHING
    assert conv.anchor.verbatim == "I need to finish the assignment tonight"
    spoken = [t for t, _, _ in speaker.calls]
    assert any("computer-generated" in s for s in spoken)
    # second session on the same DB: no disclosure again
    conv2, speaker2, _ = make_conv(settings, store, clock, [])
    conv2.start_session()  # resumes
    assert not any("computer-generated" in t for t, _, _ in speaker2.calls)


def test_resume_picks_up_same_anchor(settings, store, clock):
    conv, _, _ = anchored(settings, store, clock)
    conv2, _, _ = make_conv(settings, store, clock, [])
    a = conv2.start_session()
    assert a.id == conv.anchor.id and conv2.state == State.WATCHING
    assert conv2.policy is not None


def test_vague_sentence_gets_exactly_one_question(settings, store, clock):
    conv, speaker, listener = anchored(settings, store, clock, sentence="study", replies=["the DBMS assignment, section 3"])
    spoken = [t for t, _, _ in speaker.calls]
    questions = [s for s in spoken if s.endswith("?") and "working on right now" not in s]
    assert len(questions) == 1
    assert conv.anchor.verbatim == "study"
    assert "DBMS" in conv.anchor.clarified
    assert len(listener.calls) == 2  # sentence + one answer, never a third listen at intake


def test_clear_sentence_gets_no_question(settings, store, clock):
    conv, speaker, listener = anchored(settings, store, clock)
    assert len(listener.calls) == 1


def test_hindi_sentence_sets_language(settings, store, clock):
    conv, speaker, _ = anchored(settings, store, clock, sentence="मुझे आज रात असाइनमेंट खत्म करना है")
    assert conv.anchor.language == "hi"
    assert speaker.calls[-1][1] == "hi"


def test_policy_from_task_kind(settings, store, clock):
    c1, _, _ = anchored(settings, store, clock, sentence="fix the login bug in the python backend")
    coding = c1.policy.patience_seconds
    store2 = type(store)(settings.db_path.with_name("b.db")).open()
    c2, _, _ = anchored(settings, store2, clock, sentence="write the essay on climate policy")
    writing = c2.policy.patience_seconds
    store3 = type(store)(settings.db_path.with_name("c.db")).open()
    c3, _, _ = anchored(settings, store3, clock, sentence="watch the operating systems lecture")
    assert coding > writing
    assert c3.policy.patience_seconds > coding
    assert c3.policy.pause_idle_s is None


def test_demo_tempo_cuts_patience_but_keeps_shape(settings, store, clock):
    settings.demo = True
    conv, _, _ = anchored(settings, store, clock, sentence="I am learning about AI agents")
    assert conv.policy.patience_seconds == 10 and conv.policy.pause_idle_s == 2 and conv.policy.pause_stable_s == 3
    assert store.load_policy(conv.anchor.id).patience_seconds == 180      # stored policy is the real one
    first_id = conv.anchor.id
    conv2, speaker2, _ = make_conv(settings, store, clock, ["solving a DSA question on leetcode"])
    conv2.start_session()                                                  # demo launch asks again, every time
    assert conv2.anchor.id != first_id and store.get_anchor(first_id).status == "retired"
    assert any("working on right now" in t for t, _, _ in speaker2.calls)
    assert conv2.policy.patience_seconds == 10
    store3 = type(store)(settings.db_path.with_name("w.db")).open()
    c3, _, _ = anchored(settings, store3, clock, sentence="watch the operating systems lecture")
    assert c3.policy.patience_seconds == 10 and c3.policy.pause_idle_s is None
    settings.demo = False
    conv4, _, _ = make_conv(settings, store, clock, [])
    conv4.start_session()                                                  # normal mode still resumes
    assert conv4.anchor.id == conv2.anchor.id and conv4.policy.patience_seconds == 180


def test_serious_goal_disables_humour(settings, store, clock):
    conv, _, _ = anchored(settings, store, clock, sentence="sort out my hospital bills and the loan paperwork")
    assert conv.policy.humor_ok is False


# ----------------------------------------------------------------------------- confrontation

def test_roast_then_choice_then_defer(settings, store, clock):
    conv, speaker, _ = anchored(settings, store, clock, replies=["give me twenty minutes"])
    out = conv.confront(observed())
    assert out.intent == Intent.DEFER and out.minutes == 20 and out.pushed is False
    first = out.spoken[0]
    assert "crabs" in first.lower() and "12" in first
    assert first.rstrip().endswith("?")            # the forced choice comes in the same breath
    assert conv.state == State.DETOUR
    assert conv.anchor.detour_until == pytest.approx(clock.now() + 20 * 60)
    assert store.active_anchor().detour_until == pytest.approx(clock.now() + 20 * 60)
    assert "20" in out.spoken[-1]                  # repeats the amount back


def test_break_without_amount_asks_how_long_once(settings, store, clock):
    conv, speaker, listener = anchored(settings, store, clock, replies=["I just need a short break", "two minutes"])
    out = conv.confront(observed())
    assert out.intent == Intent.DEFER and out.minutes == 2
    spoken = [t for t, _, _ in speaker.calls]
    assert spoken.count("How long do you need?") == 1
    assert "2" in spoken[-1]                                     # repeats the amount back
    assert conv.anchor.detour_until == pytest.approx(clock.now() + 120)
    assert out.pushed is False                                   # asking "how long" is not the push-back


@pytest.mark.parametrize("answer,minutes", [("One minute.", 1), ("a minute", 1), ("just a minute", 1), ("ek minute", 1), ("एक मिनट", 1), ("five", 5), ("half an hour", 30)])
def test_how_long_answer_is_always_literal(settings, store, clock, answer, minutes):
    conv, speaker, _ = anchored(settings, store, clock, replies=["Just a short break.", answer])
    out = conv.confront(observed())
    assert out.intent == Intent.DEFER and out.minutes == minutes
    assert str(minutes) in speaker.calls[-1][0]


def test_fresh_start_asks_again_and_cancels_a_pending_break(settings, store, clock):
    conv, _, _ = anchored(settings, store, clock, replies=["twenty minutes"])
    conv.confront(observed())
    assert store.active_anchor().detour_until is not None
    settings.fresh_start = True                                 # the default in real use (ANCHOR_RESUME=1 disables)
    conv2, speaker2, _ = make_conv(settings, store, clock, ["write the lab report"])
    conv2.start_session()
    assert conv2.anchor.verbatim == "write the lab report" and conv2.state == State.WATCHING
    assert any("working on right now" in t for t, _, _ in speaker2.calls)
    assert store.get_anchor(conv.anchor.id).status == "retired"
    assert store.get_anchor(conv.anchor.id).detour_until is None
    from anchor.config import Settings
    assert Settings.from_env().fresh_start is True


def test_break_without_amount_and_silence_takes_default(settings, store, clock):
    conv, speaker, listener = anchored(settings, store, clock, replies=["I need a break", ""])
    out = conv.confront(observed())
    assert out.intent == Intent.DEFER and out.minutes == 15
    assert [t for t, _, _ in speaker.calls].count("How long do you need?") == 1
    assert len(listener.calls) == 3                              # intake, reply, one how-long answer — no loop


def test_vague_amount_does_not_ask(settings, store, clock):
    conv, speaker, _ = anchored(settings, store, clock, replies=["give me a bit"])
    out = conv.confront(observed())
    assert out.intent == Intent.DEFER and out.minutes == 15
    assert "How long do you need?" not in [t for t, _, _ in speaker.calls]


def test_break_how_long_in_hindi(settings, store, clock):
    conv, speaker, _ = anchored(settings, store, clock, sentence="मुझे आज रात असाइनमेंट खत्म करना है", replies=["थोड़ा ब्रेक चाहिए", "दो मिनट"])
    out = conv.confront(observed())
    assert out.intent == Intent.DEFER and out.minutes == 2
    assert "कितना समय चाहिए?" in [t for t, _, _ in speaker.calls]


@pytest.mark.parametrize("reply", ["I will go back to LeetCode.", "okay okay", "yes", "fine, I'll stop", "haan wapas ja raha hoon"])
def test_going_back_is_accepted_without_pushback(settings, store, clock, reply):
    conv, speaker, listener = anchored(settings, store, clock, replies=[reply])
    out = conv.confront(observed())
    assert out.intent == Intent.RESUME and out.pushed is False
    assert not [c for c in speaker.calls if c[2] == "flat"]
    assert conv.state == State.WATCHING and conv.anchor.detour_until is None
    assert speaker.calls[-1][0] == R.accept_line("en")
    assert len(listener.calls) == 2


def test_clear_replies_skip_the_model_round_trip(settings, store, clock):
    llm = FakeLLM(replies=[ReplyIntent(Intent.EVASIVE)] * 5)
    conv, speaker, _ = anchored(settings, store, clock, replies=["twenty minutes"], llm=llm)
    conv.confront(observed())
    assert not [c for c in llm.calls if c[0] == "classify_reply"]        # answered locally, instantly
    conv.anchor.detour_until = None
    conv.state = State.WATCHING
    conv2, _, listener2 = make_conv(settings, store, clock, [], llm=llm)
    conv2.start_session()
    listener2.replies = ["the weather is nice", "ten minutes"]
    conv2.confront(observed())
    assert len([c for c in llm.calls if c[0] == "classify_reply"]) == 1   # only the ambiguous reply asked the model


@pytest.mark.parametrize("reply", ["stop", "shut up", "leave me alone", "chup", "बस करो"])
def test_stop_means_stop_no_pushback_and_cooler_register(settings, store, clock, reply):
    conv, speaker, listener = anchored(settings, store, clock, replies=[reply])
    out = conv.confront(observed())
    assert out.intent == Intent.RESUME and out.pushed is False
    assert not [c for c in speaker.calls if c[2] == "flat"]
    assert conv.state == State.WATCHING and conv.current_register() == Register.DRY
    assert speaker.calls[-1][0] == R.accept_line("en")                   # no farewell roast


def test_jab_gets_one_comeback_then_the_question_stands(settings, store, clock):
    conv, speaker, listener = anchored(settings, store, clock, replies=["you're just a bot", "twenty minutes"])
    out = conv.confront(observed())
    texts = [t for t, _, _ in speaker.calls]
    comebacks = [t for t, _, tone in speaker.calls if tone == "deadpan" and "?" not in t]
    assert len(comebacks) == 1 and len(comebacks[0].split()) <= 12
    assert out.pushed is True and out.intent == Intent.DEFER and out.minutes == 20
    # a second jab in the same session gets a different line, never the same one twice
    conv.anchor.detour_until = None
    conv.state = State.WATCHING
    listener.replies = ["stupid bot", "ten minutes"]
    conv.confront(observed())
    comebacks2 = [t for t, _, tone in speaker.calls if tone == "deadpan" and "?" not in t]
    assert len(comebacks2) == 2 and comebacks2[0] != comebacks2[1]


def test_roast_is_spoken_with_its_delivery_tag(settings, store, clock):
    from anchor.models import ComposeResult
    llm = FakeLLM(compositions=[ComposeResult(
        roast="Twelve minutes into researching whether crabs can swim on Google. Bold pivot.",
        choice_line="Short break, or is this the new main thing?", delivery="disbelief", mechanism="status reversal")])
    conv, speaker, _ = anchored(settings, store, clock, replies=["ten minutes"], llm=llm)
    out = conv.confront(observed())
    assert out.joke_used
    roast_call = [c for c in speaker.calls if "crabs" in c[0]][0]
    assert roast_call[2] == "disbelief"
    events = [e for e in store.recent_events(20) if e["state"] == "ROAST"]
    assert events and "delivery=disbelief" in events[0]["detail"]


def test_llm_evasive_is_overridden_by_local_resume(settings, store, clock):
    llm = FakeLLM(replies=[ReplyIntent(Intent.EVASIVE)])
    conv, speaker, _ = anchored(settings, store, clock, replies=["I'll go back to it now"], llm=llm)
    out = conv.confront(observed())
    assert out.intent == Intent.RESUME and out.pushed is False


def test_back_in_ten_minutes_is_a_break_not_a_resume(settings, store, clock):
    conv, speaker, _ = anchored(settings, store, clock, replies=["I'll be back in ten minutes"])
    out = conv.confront(observed())
    assert out.intent == Intent.DEFER and out.minutes == 10


def test_urdu_script_reply_is_treated_as_hindi_break(settings, store, clock):
    conv, speaker, _ = anchored(settings, store, clock, replies=["ایک شارٹ بریک", "two minutes"])
    out = conv.confront(observed())
    assert out.intent == Intent.DEFER and out.minutes == 2


def test_evasive_gets_exactly_one_flat_pushback_then_accepts(settings, store, clock):
    conv, speaker, listener = anchored(settings, store, clock, replies=["hmm whatever", "yeah still true, twenty minutes"])
    out = conv.confront(observed())
    assert out.pushed is True
    push = [c for c in speaker.calls if c[2] == "flat"]
    assert len(push) == 1
    assert push[0][0] == R.pushback_line(conv.anchor.verbatim, "en")
    assert conv.anchor.verbatim in push[0][0]
    assert out.intent == Intent.DEFER and out.minutes == 20


def test_silence_counts_as_evasive_and_double_evasive_defaults_to_detour(settings, store, clock):
    conv, speaker, listener = anchored(settings, store, clock, replies=["", ""])
    out = conv.confront(observed())
    assert out.pushed is True
    assert out.intent == Intent.DEFER and out.minutes == settings.default_detour_minutes == 15
    flat = [c for c in speaker.calls if c[2] == "flat"]
    assert len(flat) == 1                          # never two push-backs
    assert len(listener.calls) == 1 + 2            # intake + roast reply + push-back reply, nothing more


def test_pushback_never_repeats_even_if_llm_says_evasive_forever(settings, store, clock):
    llm = FakeLLM(replies=[ReplyIntent(Intent.EVASIVE)] * 10)
    conv, speaker, listener = anchored(settings, store, clock, replies=["meh", "meh", "meh", "meh"], llm=llm)
    out = conv.confront(observed())
    assert len([c for c in speaker.calls if c[2] == "flat"]) == 1
    assert out.intent == Intent.DEFER


def test_switch_branch_adopts_new_sentence_without_question(settings, store, clock):
    conv, speaker, listener = anchored(settings, store, clock, replies=["actually this is the new main thing, I'm building my resume now"])
    old_id = conv.anchor.id
    out = conv.confront(observed())
    assert out.intent == Intent.SWITCH
    assert store.get_anchor(old_id).status == "retired"
    assert conv.anchor.id != old_id and conv.state == State.WATCHING
    assert "resume" in conv.anchor.verbatim
    assert len(listener.calls) == 2                # intake + the reply; no clarifying question mid-flow


@pytest.mark.parametrize("reply", ["This is the new main thing.", "new main thing", "this is the main thing now", "switch"])
def test_switch_that_names_nothing_takes_the_current_activity(settings, store, clock, reply):
    conv, speaker, listener = anchored(settings, store, clock, replies=[reply])
    out = conv.confront(Observed(activity="messaging contacts on WhatsApp", app="Google Chrome",
                                 title="WhatsApp", url_domain="web.whatsapp.com", minutes_off_task=1, confidence=0.98))
    assert out.intent == Intent.SWITCH
    assert conv.anchor.verbatim == "messaging contacts on WhatsApp"
    assert len(listener.calls) == 2                                      # still no follow-up question


def test_switch_with_vague_sentence_still_asks_nothing(settings, store, clock):
    llm = FakeLLM(replies=[ReplyIntent(Intent.SWITCH, new_anchor="study")])
    conv, speaker, listener = anchored(settings, store, clock, replies=["new thing: study"], llm=llm)
    conv.confront(observed())
    assert conv.anchor.verbatim == "study"
    assert len(listener.calls) == 2


def test_done_branch_congratulates_and_asks_whats_next(settings, store, clock):
    conv, speaker, listener = anchored(settings, store, clock, replies=["it's done, I finished it", "now I'm writing the lab report"])
    old_id = conv.anchor.id
    out = conv.confront(observed())
    assert out.intent == Intent.DONE
    assert store.get_anchor(old_id).status == "completed"
    spoken = [t for t, _, _ in speaker.calls]
    assert any(t == R.congrats("en") for t in spoken)
    assert any(t == R.whats_next("en") for t in spoken)
    assert conv.anchor.verbatim == "now I'm writing the lab report" and conv.state == State.WATCHING


def test_repeat_confrontations_get_shorter(settings, store, clock):
    conv, speaker, _ = anchored(settings, store, clock, replies=["ten minutes", "ten minutes", "ten minutes"])
    lengths = []
    for _ in range(3):
        conv.anchor.detour_until = None
        store.set_detour(conv.anchor.id, None)
        conv.state = State.WATCHING
        out = conv.confront(observed())
        roast_part = out.spoken[0].replace(R.choice_line("en"), "").strip()
        lengths.append(len(roast_part.split()))
    assert lengths[0] >= lengths[1] >= lengths[2]
    assert lengths[2] <= 8


def test_irritated_reply_cools_register_for_the_day_and_never_climbs(settings, store, clock):
    conv, speaker, _ = anchored(settings, store, clock, replies=["ugh stop, twenty minutes", "twenty minutes"])
    assert conv.current_register() == Register.PLAYFUL
    conv.confront(observed())
    assert conv.current_register() == Register.DRY
    conv.anchor.detour_until = None
    conv.state = State.WATCHING
    conv.confront(observed())                      # amused/neutral reply does not warm it back up
    assert conv.current_register() == Register.DRY
    clock.advance(24 * 3600)                       # a new day resets to the default
    assert conv.current_register() == Register.PLAYFUL


def test_low_confidence_gets_hedge_not_joke(settings, store, clock):
    conv, speaker, _ = anchored(settings, store, clock, replies=["ten minutes"])
    out = conv.confront(observed(confidence=0.5))
    assert out.joke_used is False
    assert speaker.calls[-2][2] not in {"roast", "deadpan", "mock_respect", "disbelief"}


def test_serious_anchor_gets_plain_statement(settings, store, clock):
    conv, speaker, _ = anchored(settings, store, clock, sentence="deal with my hospital bills tonight", replies=["ten minutes"])
    out = conv.confront(observed())
    assert out.joke_used is False


def test_muted_conversation_speaks_nothing(settings, store, clock):
    conv, speaker, listener = make_conv(settings, store, clock, ["finish the assignment", "ten minutes"], muted=True)
    conv.start_session()
    conv.confront(observed())
    assert speaker.calls == []
    assert any(l.startswith("anchor (muted)") for l in conv.transcript)


def test_hourly_ceiling(settings, store, clock):
    conv, speaker, _ = anchored(settings, store, clock, replies=["ten minutes"] * 6)
    for i in range(4):
        assert conv.may_confront()
        conv.confront(observed())
        conv.anchor.detour_until = None
        conv.state = State.WATCHING
        clock.advance(60)
    assert conv.may_confront() is False
    clock.advance(3600)
    assert conv.may_confront() is True


def test_no_confrontation_during_detour(settings, store, clock):
    conv, _, _ = anchored(settings, store, clock, replies=["twenty minutes"])
    conv.confront(observed())
    assert conv.may_confront() is False


def test_hindi_confrontation_is_native(settings, store, clock):
    conv, speaker, _ = anchored(settings, store, clock, sentence="मुझे आज रात असाइनमेंट खत्म करना है", replies=["बीस मिनट"])
    out = conv.confront(observed())
    line = out.spoken[0]
    assert any("ऀ" <= ch <= "ॿ" for ch in line)
    assert out.intent == Intent.DEFER and out.minutes == 20
    assert speaker.calls[-2][1] == "hi"
