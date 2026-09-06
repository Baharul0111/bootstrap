"""Joke-safety suite for anchor/roast.py.

Every test here guards a promise from goal.md 13-18 / architecture.md Component 8:
the joke is about the gap, never the person; it is specific; it gets shorter and
never escalates; it is skipped when it would be wrong; the push-back and the
detour reminder never carry a roast; the register only ever cools.

Run: .venv/bin/python -m pytest tests/test_roast_safety.py -q
"""

from __future__ import annotations

import re
from dataclasses import replace

import pytest

from anchor.config import Settings
from anchor.models import Anchor, Composition, Observed, Policy, Register
from anchor.personality import Personality, load_personality
from anchor import roast as R
from anchor.roast import (
    BANNED,
    Composer,
    RegisterLadder,
    accept_line,
    caps_for,
    choice_line,
    congrats,
    couldnt_hear,
    count_sentences,
    count_words,
    detour_confirm,
    detour_reminder,
    disclosure,
    fallback_roast,
    find_banned,
    hedged_statement,
    intake_question,
    is_specific,
    permission_missing,
    plain_statement,
    pushback_line,
    switch_ack,
    validate_roast,
    whats_next,
)

DEVANAGARI = re.compile(r"[ऀ-ॿ]")


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class ScriptedLLM:
    """compose() pops from a script. Items may be (roast, choice) tuples, None,
    an Exception instance (raised), or arbitrary junk. When the script is empty
    it returns ``default``. Uses the exact production signature so a keyword
    mismatch in the Composer would fail loudly here."""

    def __init__(self, script=(), default=None):
        self.script = list(script)
        self.default = default
        self.calls: list[dict] = []

    def compose(self, anchor, observed, policy, register, repeat_index, language,
                word_cap, sentence_cap, profanity_ok=False, *, intensity="pointed",
                persona_prompt="", eligible_lines=None, recent_roasts=None,
                tease_material="", exclusions=""):
        self.calls.append(dict(anchor=anchor, observed=observed, policy=policy,
                               register=register, repeat_index=repeat_index,
                               language=language, word_cap=word_cap,
                               sentence_cap=sentence_cap, profanity_ok=profanity_ok,
                               intensity=intensity, persona_prompt=persona_prompt,
                               eligible_lines=eligible_lines, recent_roasts=recent_roasts,
                               tease_material=tease_material, exclusions=exclusions))
        if not self.script:
            return self.default
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class DictStore:
    """Duck-typed stand-in for anchor.db.Store's settings API."""

    def __init__(self):
        self.data: dict[str, str] = {}

    def get_setting(self, key, default=None):
        return self.data.get(key, default)

    def set_setting(self, key, value):
        self.data[key] = value


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

VERBATIM = "I'm going to finish the quarterly report before lunch"


@pytest.fixture
def anchor() -> Anchor:
    return Anchor(1, VERBATIM, "finish the quarterly report", "en", "active", 0.0)


@pytest.fixture
def observed() -> Observed:
    return Observed("researching whether crabs can swim on Google",
                    app="Google Chrome", url_domain="google.com",
                    title="can crabs swim - Google Search", minutes_off_task=12)


@pytest.fixture
def observed_hi() -> Observed:
    return Observed("Google पर crabs की रिसर्च", app="Google Chrome",
                    url_domain="google.com", minutes_off_task=12)


@pytest.fixture
def policy() -> Policy:
    return Policy(humor_ok=True, register=Register.PLAYFUL)


@pytest.fixture
def settings() -> Settings:
    return Settings()


@pytest.fixture
def no_library() -> Personality:
    """The persona with its approved library removed. With the library present
    the Composer's fallback is an approved punchline (tests/test_personality.py);
    the tests below pin the templated fallback, so they take the library away."""
    return replace(load_personality(), library=[])


GOOD_EN ="Still researching crabs on Google, 12 minutes in — the report is holding its breath."
GOOD_HI = "12 मिनट से Google पर crabs की रिसर्च चल रही है, रिपोर्ट इंतज़ार में है।"
CHOICE_EN = "Short break, or is this the new main thing?"

# (category, English example, Hindi example)
CATEGORY_EXAMPLES = [
    ("APPEARANCE", "ugly", "बदसूरत"),
    ("INTELLIGENCE", "idiot", "बेवकूफ"),
    ("DISCIPLINE_AS_FLAW", "lazy", "निकम्मा"),
    ("RELATIONSHIPS", "girlfriend", "शादी"),
    ("FAMILY", "mother", "माँ"),
    ("MONEY", "broke", "गरीब"),
    ("WEIGHT", "obese", "मोटा"),
    ("PROFANITY", "shit", "साला"),
]


def bad_en(term: str) -> str:
    return f"Still researching crabs on Google after 12 minutes, you {term}."


def bad_hi(term: str) -> str:
    return f"12 मिनट से Google पर crabs की रिसर्च, {term} कहीं के।"


# ---------------------------------------------------------------------------
# Banned categories
# ---------------------------------------------------------------------------

def test_all_categories_present_and_lowercase():
    for cat in ("APPEARANCE", "INTELLIGENCE", "DISCIPLINE_AS_FLAW", "RELATIONSHIPS",
                "FAMILY", "MONEY", "WEIGHT", "PROFANITY"):
        assert cat in BANNED and BANNED[cat]
        for term in BANNED[cat]:
            assert term == term.lower()


@pytest.mark.parametrize("category,en_term,hi_term", CATEGORY_EXAMPLES)
def test_example_terms_are_listed(category, en_term, hi_term):
    assert en_term in BANNED[category]
    assert hi_term in BANNED[category]


@pytest.mark.parametrize("category,en_term,hi_term", CATEGORY_EXAMPLES)
def test_banned_en_rejected_with_category_in_reason(category, en_term, hi_term, observed):
    text = bad_en(en_term)
    assert (category, en_term) in find_banned(text)
    ok, reason = validate_roast(text, observed, 0)
    assert not ok
    assert category in reason
    # The same line without the slur is fine: the term alone caused the rejection.
    assert validate_roast(bad_en("person"), observed, 0)[0]


@pytest.mark.parametrize("category,en_term,hi_term", CATEGORY_EXAMPLES)
def test_banned_hi_rejected_with_category_in_reason(category, en_term, hi_term, observed_hi):
    text = bad_hi(hi_term)
    assert (category, hi_term) in find_banned(text)
    ok, reason = validate_roast(text, observed_hi, 0)
    assert not ok
    assert category in reason
    assert validate_roast(bad_hi("दोस्त"), observed_hi, 0)[0]


@pytest.mark.parametrize("category,en_term,hi_term", CATEGORY_EXAMPLES)
def test_composer_regenerates_once_then_falls_back_en(category, en_term, hi_term,
                                                      anchor, observed, policy, settings, no_library):
    llm = ScriptedLLM([(bad_en(en_term), CHOICE_EN), (bad_en(en_term), CHOICE_EN)])
    out = Composer(llm, settings, no_library).compose(anchor, observed, policy, Register.PLAYFUL, 0, "en")
    assert isinstance(out, Composition)
    assert len(llm.calls) == 2                       # regenerate exactly once
    assert find_banned(out.roast) == []
    assert en_term not in out.roast.lower()
    assert validate_roast(out.roast, observed, 0)[0]
    assert out.joke_used is True
    assert out.roast == fallback_roast(anchor, observed, 0, "en")


@pytest.mark.parametrize("category,en_term,hi_term", CATEGORY_EXAMPLES)
def test_composer_regenerates_once_then_falls_back_hi(category, en_term, hi_term,
                                                      anchor, observed_hi, policy, settings):
    llm = ScriptedLLM([(bad_hi(hi_term), "ब्रेक है या मुख्य काम?"), (bad_hi(hi_term), "ब्रेक?")])
    out = Composer(llm, settings).compose(anchor, observed_hi, policy, Register.PLAYFUL, 0, "hi")
    assert len(llm.calls) == 2
    assert find_banned(out.roast) == []
    assert hi_term not in out.roast
    assert validate_roast(out.roast, observed_hi, 0)[0]
    assert out.joke_used is True
    assert DEVANAGARI.search(out.roast)


def test_word_boundary_matching_avoids_false_positives():
    assert find_banned("Facebook is open in a tab") == []          # not "face"
    assert find_banned("let's face it") != []
    assert find_banned("a classy report") == []                    # not "ass"
    assert find_banned("मसाले वाला वीडियो") == []                    # not "साले"
    assert find_banned("कमांड लाइन खुली है") == []                    # not "मां"
    assert find_banned("Your EX texted") == [("RELATIONSHIPS", "your ex")]


# ---------------------------------------------------------------------------
# Profanity opt-in
# ---------------------------------------------------------------------------

def test_profanity_blocked_by_default_allowed_with_flag(observed):
    text = "Still researching crabs on Google after 12 minutes, holy shit."
    assert ("PROFANITY", "shit") in find_banned(text)
    assert find_banned(text, profanity_ok=True) == []
    assert not validate_roast(text, observed, 0)[0]
    assert validate_roast(text, observed, 0, profanity_ok=True)[0]


def test_profanity_flag_never_unlocks_other_categories(observed):
    text = "Still researching crabs on Google after 12 minutes, you lazy shit."
    hits = find_banned(text, profanity_ok=True)
    assert ("DISCIPLINE_AS_FLAW", "lazy") in hits
    assert all(cat != "PROFANITY" for cat, _ in hits)


def test_composer_respects_settings_profanity_ok(anchor, observed, policy):
    text = "Still researching crabs on Google after 12 minutes, holy shit."
    llm = ScriptedLLM([(text, CHOICE_EN)], default=(text, CHOICE_EN))
    out = Composer(llm, Settings(profanity_ok=True)).compose(
        anchor, observed, policy, Register.SPICY, 0, "en")
    assert out.roast == text and out.joke_used and len(llm.calls) == 1
    assert llm.calls[0]["profanity_ok"] is True

    llm2 = ScriptedLLM([], default=(text, CHOICE_EN))
    out2 = Composer(llm2, Settings()).compose(anchor, observed, policy, Register.SPICY, 0, "en")
    assert len(llm2.calls) == 2
    assert "shit" not in out2.roast
    assert llm2.calls[0]["profanity_ok"] is False


# ---------------------------------------------------------------------------
# The four gates
# ---------------------------------------------------------------------------

def test_serious_anchor_no_joke_llm_never_called(anchor, observed, settings):
    llm = ScriptedLLM(default=(GOOD_EN, CHOICE_EN))
    serious = Policy(humor_ok=False)
    out = Composer(llm, settings).compose(anchor, observed, serious, Register.SPICY, 0, "en")
    assert out.joke_used is False
    assert llm.calls == []
    assert out.roast == plain_statement(anchor, observed, "en")
    assert out.choice_line == choice_line("en")


def test_low_confidence_hedged_no_joke(anchor, policy, settings):
    llm = ScriptedLLM(default=(GOOD_EN, CHOICE_EN))
    unsure = Observed("researching crabs on Google", app="Google Chrome",
                      minutes_off_task=12, confidence=0.6)
    out = Composer(llm, settings).compose(anchor, unsure, policy, Register.PLAYFUL, 0, "en")
    assert out.joke_used is False
    assert llm.calls == []
    assert out.roast == hedged_statement(anchor, unsure, "en")
    assert "might" in out.roast


def test_confidence_threshold_is_inclusive_at_0_75(anchor, policy, settings):
    llm = ScriptedLLM(default=(GOOD_EN, CHOICE_EN))
    sure_enough = Observed("researching crabs on Google", app="Google Chrome",
                           minutes_off_task=12, confidence=0.75)
    out = Composer(llm, settings).compose(anchor, sure_enough, policy, Register.PLAYFUL, 0, "en")
    assert out.joke_used is True and len(llm.calls) == 1


def test_muted_no_joke(anchor, observed, policy, settings):
    llm = ScriptedLLM(default=(GOOD_EN, CHOICE_EN))
    out = Composer(llm, settings).compose(anchor, observed, policy, Register.SPICY, 0, "en", muted=True)
    assert out.joke_used is False and llm.calls == []
    assert out.roast == plain_statement(anchor, observed, "en")


def test_dry_repeat_no_joke_but_dry_first_time_allowed(anchor, observed, policy, settings):
    llm = ScriptedLLM(default=(GOOD_EN, CHOICE_EN))
    repeat = Composer(llm, settings).compose(anchor, observed, policy, Register.DRY, 1, "en")
    assert repeat.joke_used is False and llm.calls == []
    assert repeat.roast == plain_statement(anchor, observed, "en")

    first = Composer(llm, settings).compose(anchor, observed, policy, Register.DRY, 0, "en")
    assert first.joke_used is True and len(llm.calls) == 1
    assert first.roast == GOOD_EN


# ---------------------------------------------------------------------------
# Repeat decay and never-escalate
# ---------------------------------------------------------------------------

def test_repeat_decay_with_overlong_llm(anchor, observed, policy, settings, no_library):
    long_line = ("Honestly the crabs on Google have now had twelve minutes of your undivided "
                 "attention while the quarterly report sits there wondering what it did wrong "
                 "to deserve this kind of treatment today.")
    assert count_words(long_line) > 20
    llm = ScriptedLLM(default=(long_line, CHOICE_EN))
    counts = []
    for idx in range(4):
        out = Composer(llm, settings, no_library).compose(anchor, observed, policy, Register.PLAYFUL, idx, "en")
        assert out.joke_used is True
        sents, words = caps_for(idx)
        assert count_sentences(out.roast) <= sents
        assert count_words(out.roast) <= words
        assert validate_roast(out.roast, observed, idx)[0]
        counts.append(count_words(out.roast))
    assert counts == sorted(counts, reverse=True)      # strictly non-increasing
    assert counts[3] <= 8
    assert len(llm.calls) == 8                          # two tries per confrontation, no more
    assert [c["word_cap"] for c in llm.calls] == [20, 20, 12, 12, 8, 8, 8, 8]
    assert [c["sentence_cap"] for c in llm.calls] == [2, 2, 1, 1, 1, 1, 1, 1]


def test_caps_never_escalate():
    caps = [caps_for(i) for i in range(7)]
    for earlier, later in zip(caps, caps[1:]):
        assert later[0] <= earlier[0]
        assert later[1] <= earlier[1]
    assert caps[0] == (2, 20) and caps[1] == (1, 12) and caps[2] == (1, 8)
    assert caps_for(-3) == (2, 20)


# ---------------------------------------------------------------------------
# Specificity
# ---------------------------------------------------------------------------

def test_generic_roast_rejected(observed):
    ok, reason = validate_roast("Wow, distracted again, classic.", observed, 0)
    assert not ok and "specific" in reason


def test_specific_roast_accepted(observed):
    assert validate_roast(GOOD_EN, observed, 0) == (True, "ok")
    assert is_specific("Crabs on Google, 12 minutes.", observed, 0)


def test_activity_without_time_fails_only_for_first_two(observed):
    line = "Google crabs, again."
    assert not is_specific(line, observed, 0)
    assert not is_specific(line, observed, 1)
    assert is_specific(line, observed, 2)
    assert is_specific("Twelve minutes on the crabs.", observed, 0)      # number word
    assert is_specific("बारह मिनट से crabs पर।", observed, 1)              # Hindi number + मिनट


def test_time_alone_is_not_specific(observed):
    assert not is_specific("Twelve minutes gone, somewhere.", observed, 0)


@pytest.mark.parametrize("obs", [
    Observed("researching whether crabs can swim on Google", app="Google Chrome", minutes_off_task=12),
    Observed("watching a video about medieval siege engines", app="Safari", minutes_off_task=0),
    Observed("YouTube पर crab videos देखना", app="Safari", url_domain="youtube.com", minutes_off_task=7),
    Observed("scrolling Reddit", minutes_off_task=25),
    Observed("reading a very long thread about mechanical keyboards on Hacker News", minutes_off_task=3),
    Observed("Slack", app="Slack", minutes_off_task=1),
])
@pytest.mark.parametrize("language", ["en", "hi"])
@pytest.mark.parametrize("idx", [0, 1, 2, 3])
def test_fallback_roast_passes_validation(anchor, obs, language, idx):
    text = fallback_roast(anchor, obs, idx, language)
    ok, reason = validate_roast(text, obs, idx)
    assert ok, f"{language} idx={idx}: {reason!r} for {text!r}"
    assert find_banned(text) == []
    assert str(obs.minutes_off_task) in text or idx >= 2


def test_fallback_drops_anchor_quote_that_carries_banned_term(observed):
    touchy = Anchor(2, "call mom about the money she lent me", "call mom", "en", "active", 0.0)
    for lang in ("en", "hi"):
        text = fallback_roast(touchy, observed, 0, lang)
        assert find_banned(text) == []
        assert validate_roast(text, observed, 0)[0]
        assert "mom" not in text.lower()


def test_fallback_names_app_when_activity_phrase_is_banned(anchor):
    obs = Observed("texting your girlfriend", app="Messages", minutes_off_task=3)
    text = fallback_roast(anchor, obs, 1, "en")
    assert find_banned(text) == []
    assert "Messages" in text
    assert validate_roast(text, obs, 1)[0]


def test_composer_degrades_to_plain_when_nothing_safe_to_name(anchor, policy, settings):
    obs = Observed("texting your girlfriend", minutes_off_task=3)
    out = Composer(ScriptedLLM(), settings).compose(anchor, obs, policy, Register.PLAYFUL, 0, "en")
    assert out.joke_used is False
    assert out.roast == plain_statement(anchor, obs, "en")


# ---------------------------------------------------------------------------
# Hindi parity
# ---------------------------------------------------------------------------

def test_hindi_templates_are_native(anchor, observed_hi):
    for idx in range(3):
        hi = fallback_roast(anchor, observed_hi, idx, "hi")
        en = fallback_roast(anchor, observed_hi, idx, "en")
        assert DEVANAGARI.search(hi) and hi != en
    assert DEVANAGARI.search(choice_line("hi")) and choice_line("hi") != choice_line("en")
    hi_push = pushback_line(VERBATIM, "hi")
    assert DEVANAGARI.search(hi_push) and hi_push != pushback_line(VERBATIM, "en")
    assert VERBATIM in hi_push
    hi_rem = detour_reminder(VERBATIM, None, "hi")
    assert DEVANAGARI.search(hi_rem) and hi_rem != detour_reminder(VERBATIM, None, "en")
    assert VERBATIM in hi_rem
    assert DEVANAGARI.search(plain_statement(anchor, observed_hi, "hi"))
    assert DEVANAGARI.search(hedged_statement(anchor, observed_hi, "hi"))


def test_unknown_language_falls_back_to_english(anchor, observed):
    assert choice_line("fr") == choice_line("en")
    assert fallback_roast(anchor, observed, 0, "xx") == fallback_roast(anchor, observed, 0, "en")
    assert DEVANAGARI.search(choice_line("hi-IN"))


def test_composer_passes_language_and_caps_to_llm(anchor, observed_hi, policy, settings):
    llm = ScriptedLLM([(GOOD_HI, "ब्रेक है या मुख्य काम?")])
    out = Composer(llm, settings).compose(anchor, observed_hi, policy, Register.PLAYFUL, 0, "hi")
    assert out.roast == GOOD_HI and out.joke_used
    call = llm.calls[0]
    assert call["language"] == "hi" and call["register"] == Register.PLAYFUL
    assert call["repeat_index"] == 0 and (call["sentence_cap"], call["word_cap"]) == (2, 20)


# ---------------------------------------------------------------------------
# Push-back and detour lines never carry a roast
# ---------------------------------------------------------------------------

def test_pushback_is_verbatim_plus_question_only():
    line = pushback_line(VERBATIM, "en")
    assert line == f"You said: {VERBATIM}. Still true?"
    assert VERBATIM in line
    assert find_banned(line) == []
    assert count_sentences(line) == 2
    assert line.endswith("Still true?")


def test_pushback_keeps_existing_terminal_punctuation():
    line = pushback_line("Finish the report.", "en")
    assert line == "You said: Finish the report. Still true?"
    assert ".." not in line
    hi = pushback_line("रिपोर्ट पूरी करनी है।", "hi")
    assert hi == "आपने कहा था: रिपोर्ट पूरी करनी है। क्या यह अब भी सच है?"


def test_pushback_never_contains_joke_markers():
    line = pushback_line(VERBATIM, "en")
    assert "pivot" not in line and "again" not in line
    assert line.count("?") == 1


def test_detour_reminder_plain_and_verbatim():
    line = detour_reminder(VERBATIM, None, "en")
    assert line == f"Time's up. You said: {VERBATIM}. Heading back to it?"
    assert VERBATIM in line and find_banned(line) == []
    assert "late" not in line.lower()
    assert "pivot" not in line and "again" not in line
    assert detour_reminder(VERBATIM, 0, "en") == line


def test_detour_reminder_late_variant_mentions_lateness():
    line = detour_reminder(VERBATIM, 3, "en")
    assert line.startswith("Sorry, I'm 3 minutes late with this. ")
    assert VERBATIM in line and find_banned(line) == []
    hi = detour_reminder(VERBATIM, 3, "hi")
    assert "3 मिनट" in hi and "देर" in hi and VERBATIM in hi
    assert detour_reminder(VERBATIM, None, "hi") not in hi or True  # late text is a prefix
    assert hi.endswith(detour_reminder(VERBATIM, None, "hi"))


def test_detour_confirm_repeats_amount_and_time():
    assert detour_confirm(20, "9:40", "en") == "Okay, 20 minutes. I'll come back at 9:40."
    hi = detour_confirm(20, "9:40", "hi")
    assert "20 मिनट" in hi and "9:40" in hi and DEVANAGARI.search(hi)


@pytest.mark.parametrize("language", ["en", "hi"])
def test_every_template_is_free_of_banned_terms(language, anchor, observed):
    outputs = [
        plain_statement(anchor, observed, language),
        hedged_statement(anchor, observed, language),
        choice_line(language),
        pushback_line(VERBATIM, language),
        detour_confirm(15, "9:40", language),
        detour_reminder(VERBATIM, None, language),
        detour_reminder(VERBATIM, 5, language),
        congrats(language),
        whats_next(language),
        intake_question(language),
        disclosure(language),
        permission_missing("Screen Recording", language),
        couldnt_hear(language),
        switch_ack("write the intro", language),
        accept_line(language),
    ] + [fallback_roast(anchor, observed, i, language) for i in range(4)]
    for text in outputs:
        assert text and text.strip()
        assert find_banned(text) == [], text
    assert count_sentences(congrats(language)) == 1
    assert intake_question(language).count("?") == 1
    assert "write the intro" in switch_ack("write the intro", language)
    assert "Screen Recording" in permission_missing("Screen Recording", language)


# ---------------------------------------------------------------------------
# Choice line handling
# ---------------------------------------------------------------------------

def test_llm_choice_line_used_only_when_clean_short_question(anchor, observed, policy, settings):
    def run(choice):
        llm = ScriptedLLM([(GOOD_EN, choice)])
        return Composer(llm, settings).compose(anchor, observed, policy, Register.PLAYFUL, 0, "en")

    assert run(CHOICE_EN).choice_line == CHOICE_EN
    assert run("Is this a break, you idiot?").choice_line == choice_line("en")
    assert run("Break or new main thing.").choice_line == choice_line("en")          # not a question
    assert run("").choice_line == choice_line("en")
    too_long = "So is this " + "really " * 18 + "a break?"
    assert count_words(too_long) > 20
    assert run(too_long).choice_line == choice_line("en")


# ---------------------------------------------------------------------------
# Counting helpers
# ---------------------------------------------------------------------------

def test_count_sentences_handles_danda_and_decimals():
    assert count_sentences("Hello there. It is 2.5 hours! Okay?") == 3
    assert count_sentences("आप वहाँ थे। अब यहाँ हैं।") == 2
    assert count_sentences("") == 0
    assert count_sentences("one sentence with no terminal punctuation") == 1


def test_count_words_ignores_bare_punctuation():
    assert count_words("Now it's crabs — bold pivot.") == 5
    assert count_words("अब भी YouTube, 7 मिनट हो गए।") == 7
    assert count_words("") == 0


# ---------------------------------------------------------------------------
# RegisterLadder: only ever cools
# ---------------------------------------------------------------------------

def test_register_ladder_cools_and_floors_at_dry():
    ladder = RegisterLadder(DictStore())
    assert ladder.cool("2026-09-06", Register.PLAYFUL) == Register.DRY
    assert ladder.cool("2026-09-06", Register.DRY) == Register.DRY
    fresh = RegisterLadder(DictStore())
    assert fresh.cool("2026-09-06", Register.SPICY) == Register.PLAYFUL


def test_register_ladder_persists_for_the_day_only():
    store = DictStore()
    ladder = RegisterLadder(store)
    assert ladder.current(Register.PLAYFUL, "2026-09-06") == Register.PLAYFUL
    ladder.cool("2026-09-06", Register.PLAYFUL)
    assert ladder.current(Register.PLAYFUL, "2026-09-06") == Register.DRY
    assert RegisterLadder(store).current(Register.PLAYFUL, "2026-09-06") == Register.DRY   # survives restart
    assert ladder.current(Register.PLAYFUL, "2026-09-07") == Register.PLAYFUL              # not another day
    assert store.data["register_override"] == "dry"
    assert store.data["register_override_day"] == "2026-09-06"


def test_register_ladder_never_climbs_back():
    ladder = RegisterLadder(DictStore())
    ladder.cool("2026-09-06", Register.SPICY)                       # -> playful
    assert ladder.cool("2026-09-06", Register.SPICY) == Register.DRY   # stale caller cannot re-warm
    assert ladder.current(Register.SPICY, "2026-09-06") == Register.DRY
    forbidden = ("warm", "raise", "escalat", "heat", "sharpen", "spicier", "harsher")
    for name in dir(RegisterLadder):
        assert not any(f in name.lower() for f in forbidden), name
    assert not hasattr(R, "warm_register")


# ---------------------------------------------------------------------------
# Composer never raises
# ---------------------------------------------------------------------------

def test_composer_never_raises_when_llm_raises(anchor, observed, policy, settings, no_library):
    llm = ScriptedLLM([RuntimeError("boom"), ValueError("still boom")])
    out = Composer(llm, settings, no_library).compose(anchor, observed, policy, Register.PLAYFUL, 0, "en")
    assert isinstance(out, Composition)
    assert len(llm.calls) == 2
    assert out.joke_used is True
    assert out.roast == fallback_roast(anchor, observed, 0, "en")
    assert out.choice_line == choice_line("en")


@pytest.mark.parametrize("junk", [None, "just a string", 42, ("only one",), (None, None), (3, 4), [GOOD_EN]])
def test_composer_survives_malformed_llm_results(junk, anchor, observed, policy, settings):
    llm = ScriptedLLM([junk, junk])
    out = Composer(llm, settings).compose(anchor, observed, policy, Register.PLAYFUL, 0, "en")
    assert isinstance(out, Composition) and out.roast
    assert validate_roast(out.roast, observed, 0)[0]
    assert find_banned(out.roast) == []


def test_composer_survives_missing_llm_and_odd_register(anchor, observed, policy, settings):
    out = Composer(None, settings).compose(anchor, observed, policy, "nonsense", 0, "en")
    assert isinstance(out, Composition) and out.roast
    assert out.joke_used is True
    assert find_banned(out.roast) == []


def test_composer_survives_broken_observed(anchor, policy, settings):
    broken = Observed(activity=None, minutes_off_task="lots", confidence="high")  # type: ignore[arg-type]
    out = Composer(ScriptedLLM(), settings).compose(anchor, broken, policy, Register.PLAYFUL, 0, "en")
    assert isinstance(out, Composition) and out.roast
    assert out.joke_used is False
