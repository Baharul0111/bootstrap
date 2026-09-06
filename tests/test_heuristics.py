"""Offline fallbacks: language, vagueness, sensitivity, policy, durations (EN/HI), reply intent, keyword judge."""

from __future__ import annotations

import datetime as dt

import pytest

from anchor import heuristics as H
from anchor.models import Intent, Policy, Register, Sentiment
from tests.conftest import frame

NOW_1432 = dt.datetime(2027, 1, 15, 14, 32).timestamp()
NOW_0810 = dt.datetime(2027, 1, 15, 8, 10).timestamp()


# ------------------------------------------------------------------ language
@pytest.mark.parametrize("text,lang", [
    ("I need to finish the assignment tonight", "en"),
    ("मुझे आज रात असाइनमेंट खत्म करना है", "hi"),
    ("mujhe aaj raat assignment khatam karna hai", "hi"),
    ("bees minute", "hi"),
    ("twenty minutes", "en"),
    ("", "en"),
])
def test_detect_language(text, lang):
    assert H.detect_language(text) == lang


# ------------------------------------------------------------------ vagueness & sensitivity
@pytest.mark.parametrize("text,vague", [
    ("study", True), ("work", True), ("padhai", True), ("kaam karna hai", True), ("reading", True),
    ("finish the DBMS assignment tonight", False), ("fix the login bug in the python backend", False),
    ("watch the operating systems lecture", False),
])
def test_is_vague(text, vague):
    assert H.is_vague(text) is vague


@pytest.mark.parametrize("text,sensitive", [
    ("sort out my hospital bills", True), ("file my taxes before the deadline", True),
    ("call the lawyer about the court date", True), ("मुझे अस्पताल के कागज़ पूरे करने हैं", True),
    ("karz ka hisaab karna hai", True), ("finish the DBMS assignment", False), ("write the essay", False),
])
def test_is_sensitive(text, sensitive):
    assert H.is_sensitive(text) is sensitive


# ------------------------------------------------------------------ policy
def test_policy_reference_values():
    coding = H.derive_policy_heuristic("fix the login bug in the python backend").policy
    writing = H.derive_policy_heuristic("write the essay on climate policy").policy
    watching = H.derive_policy_heuristic("watch the operating systems lecture").policy
    reading = H.derive_policy_heuristic("read chapter 4 of the textbook").policy
    general = H.derive_policy_heuristic("sort my inbox").policy
    assert coding.patience_seconds == 240 > writing.patience_seconds == 150
    assert watching.patience_seconds == 420 and watching.pause_idle_s is None and watching.pause_stable_s == 10
    assert reading.patience_seconds == 300 and general.patience_seconds == 180
    assert "stackoverflow" in coding.expected_surfaces and "youtube" in watching.expected_surfaces
    assert coding.pause_idle_s == 4


def test_policy_vague_gets_one_question_in_the_right_language():
    en = H.derive_policy_heuristic("study")
    hi = H.derive_policy_heuristic("padhai karni hai")
    assert en.needs_clarification and en.clarifying_question.endswith("?")
    assert hi.needs_clarification and hi.language == "hi" and any("ऀ" <= c <= "ॿ" for c in hi.clarifying_question)
    clear = H.derive_policy_heuristic("finish the DBMS assignment tonight")
    assert clear.needs_clarification is False and clear.clarifying_question == ""


def test_policy_humour_and_defaults():
    d = H.derive_policy_heuristic("deal with my hospital bills", Register.SPICY, 20)
    assert d.policy.humor_ok is False and d.policy.register == Register.SPICY and d.policy.default_detour_minutes == 20
    assert H.merge_clarification("study", "the DBMS assignment") == "study — the DBMS assignment"


# ------------------------------------------------------------------ durations
@pytest.mark.parametrize("text,minutes", [
    ("20 minutes", 20), ("twenty minutes", 20), ("20 min", 20), ("20m", 20), ("twenty-five minutes", 25),
    ("half an hour", 30), ("half hour", 30), ("an hour", 60), ("one hour", 60), ("1.5 hours", 90),
    ("two hours", 120), ("quarter of an hour", 15), ("fifteen minutes", 15), ("ten more minutes", 10),
    ("give me 5 mins", 5),
    ("a bit", 15), ("a while", 15), ("some time", 15), ("a few minutes", 15), ("later", 15), ("just a minute", 15),
    ("thoda der", 15), ("thodi der", 15), ("थोड़ी देर", 15), ("kuch der", 15), ("baad mein", 15),
    ("bees minute", 20), ("das minute", 10), ("pandrah minute", 15), ("aadha ghanta", 30), ("aadhe ghante", 30),
    ("ek ghanta", 60), ("do ghante", 120), ("पंद्रह मिनट", 15), ("बीस मिनट", 20), ("आधा घंटा", 30), ("एक घंटा", 60),
    ("dedh ghanta", 90),
])
def test_parse_duration_phrases(text, minutes):
    assert H.parse_duration_minutes(text, 15, NOW_1432) == minutes


@pytest.mark.parametrize("text,now,minutes", [
    ("till nine", NOW_1432, 388),       # 9 pm tonight
    ("till 9", NOW_1432, 388),
    ("until 9:30", NOW_1432, 418),
    ("till 9 pm", NOW_1432, 388),
    ("until noon", NOW_1432, 21 * 60 + 28),
    ("by 10", NOW_1432, 7 * 60 + 28),   # 10 pm
    ("till nine", NOW_0810, 50),        # 9 am this morning
    ("9 baje tak", NOW_0810, 50),
    ("sade nau tak", NOW_0810, 80),
    ("नौ बजे तक", NOW_0810, 50),
    ("till 9 am", NOW_1432, 18 * 60 + 28),
])
def test_parse_clock_times(text, now, minutes):
    assert H.parse_duration_minutes(text, 15, now) == pytest.approx(minutes, abs=1)


@pytest.mark.parametrize("text", ["no", "I am done", "yeah", "what", "", "sure thing", "this is the new main thing"])
def test_parse_duration_none(text):
    assert H.parse_duration_minutes(text, 15, NOW_1432) is None


def test_yeah_is_not_an_hour():
    assert H.parse_duration_minutes("yeah still true, twenty minutes", 15, NOW_1432) == 20


# ------------------------------------------------------------------ reply intent
def test_classify_done_beats_defer():
    r = H.classify_reply_keywords("it's done, give me a minute", 15, NOW_1432)
    assert r.intent == Intent.DONE


@pytest.mark.parametrize("text", ["done", "I finished it", "already did that", "ho gaya", "khatam", "हो गया", "पूरा हो गया"])
def test_classify_done(text):
    assert H.classify_reply_keywords(text, 15, NOW_1432).intent == Intent.DONE


@pytest.mark.parametrize("text,anchor_contains", [
    ("actually this is the new main thing, I'm building my resume now", "resume"),
    ("now I'm working on my resume", "resume"),
    ("switch to the lab report", "lab report"),
    ("ab main resume bana raha hoon", "resume bana raha hoon"),
    ("अब मैं रिज़्यूमे बना रहा हूँ", "रिज़्यूमे"),
])
def test_classify_switch_extracts_new_anchor(text, anchor_contains):
    r = H.classify_reply_keywords(text, 15, NOW_1432)
    assert r.intent == Intent.SWITCH
    assert anchor_contains in r.new_anchor
    assert "new main thing" not in r.new_anchor.lower()


@pytest.mark.parametrize("text,minutes", [
    ("twenty minutes", 20), ("give me a break", 15), ("remind me later", 15), ("half an hour", 30),
    ("thoda der baad", 15), ("बीस मिनट", 20), ("wait", 15),
])
def test_classify_defer(text, minutes):
    r = H.classify_reply_keywords(text, 15, NOW_1432)
    assert r.intent == Intent.DEFER and r.minutes == minutes


@pytest.mark.parametrize("text", [
    "I will go back to LeetCode.", "going back to it", "okay okay", "yes", "Yeah.", "haan", "fine, I'll stop",
    "wapas ja raha hoon", "मैं वापस जा रहा हूँ", "let me get back to work",
])
def test_classify_resume(text):
    assert H.classify_reply_keywords(text, 15, NOW_1432).intent == Intent.RESUME


def test_resume_with_amount_is_defer():
    r = H.classify_reply_keywords("I'll be back in ten minutes", 15, NOW_1432)
    assert r.intent == Intent.DEFER and r.minutes == 10


def test_urdu_script_counts_as_hindi():
    assert H.detect_language("ایک شارٹ بریک") == "hi"


@pytest.mark.parametrize("text", ["", "   ", "hmm", "whatever", "idk", "kya", "not sure", "pata nahi", "the weather is nice"])
def test_classify_evasive(text):
    assert H.classify_reply_keywords(text, 15, NOW_1432).intent == Intent.EVASIVE


@pytest.mark.parametrize("text,sentiment", [
    ("ugh stop it, twenty minutes", Sentiment.IRRITATED), ("chup, band karo", Sentiment.IRRITATED),
    ("haha fine, twenty minutes", Sentiment.AMUSED), ("twenty minutes", Sentiment.NEUTRAL),
    ("बकवास बंद करो", Sentiment.IRRITATED),
])
def test_classify_sentiment(text, sentiment):
    assert H.classify_reply_keywords(text, 15, NOW_1432).sentiment == sentiment


def test_classify_language():
    assert H.classify_reply_keywords("बीस मिनट", 15, NOW_1432).language == "hi"
    assert H.classify_reply_keywords("twenty minutes", 15, NOW_1432).language == "en"


# ------------------------------------------------------------------ titles, activity, judge
@pytest.mark.parametrize("title,app,bad", [
    ("", "Safari", True), ("New Tab", "Google Chrome", True), ("Untitled", "TextEdit", True),
    ("Google Chrome", "Google Chrome", True), ("crabs can swim? - Google Search", "Google Chrome", False),
    ("assignment.docx", "Microsoft Word", False),
])
def test_is_uninformative_title(title, app, bad):
    assert H.is_uninformative_title(title, app) is bad


def test_describe_activity():
    a = H.describe_activity(frame(title="crabs can swim? - Google Search - Google Chrome", domain="google.com"))
    assert a.startswith("reading") and "google.com" in a and "Google Chrome" not in a
    assert len(a.split()) <= 12
    assert H.describe_activity(frame(app="Steam", title="", domain="")) == "using Steam"
    assert "watching" in H.describe_activity(frame(title="Lo-fi beats - YouTube", domain="youtube.com"))


def test_judge_heuristic_branches():
    coding = Policy(expected_surfaces=["stackoverflow", "google"], task_kind="coding")
    assert H.judge_heuristic("fix the login bug", coding, frame(title="Lo-fi beats - YouTube", domain="youtube.com")).drift >= 0.9
    v = H.judge_heuristic("fix the login bug", coding, frame(title="python KeyError - Stack Overflow", domain="stackoverflow.com"))
    assert v.tolerated and v.drift < 0.5
    v = H.judge_heuristic("fix the login bug", coding, frame(app="Code", title="login.py — repo", domain=""))
    assert v.drift <= 0.1 and v.confidence >= 0.7
    v = H.judge_heuristic("fix the login bug", coding, frame(app="Safari", title="New Tab", domain=""))
    assert v.confidence < 0.6 and v.drift == 0.5
    v = H.judge_heuristic("fix the login bug", coding, frame(app="Preview", title="scan.pdf", domain=""))
    assert v.confidence < 0.6
    for v in [H.judge_heuristic("x", coding, frame())]:
        assert v.source == "heuristic" and v.activity


def test_judge_heuristic_youtube_ok_for_watching_task():
    watching = Policy(expected_surfaces=["youtube", "nptel"], task_kind="watching")
    v = H.judge_heuristic("watch the OS lecture", watching, frame(title="OS Lecture 4 - YouTube", domain="youtube.com"))
    assert v.tolerated and v.drift < 0.5


def test_elapsed_phrase():
    assert H.elapsed_phrase(12, "en") == "12 minutes" and H.elapsed_phrase(1, "en") == "1 minute"
    assert H.elapsed_phrase(0, "en") == "under a minute"
    assert H.elapsed_phrase(12, "hi") == "12 मिनट" and "कम" in H.elapsed_phrase(0, "hi")
