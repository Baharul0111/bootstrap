"""Conversation state machine — the protected part of Anchor.

States: IDLE → ANCHORING → WATCHING → DRIFTING → CONFRONTING → [PUSHED] → {DETOUR | SWITCH | DONE} → WATCHING

Guarantees that live here as code, not as prompt wording:
- ``push_used`` is a local boolean inside ``confront``; the model never gets a second chance to push.
- At most one clarifying question per anchor, and none at all on SWITCH.
- A detour stores an ABSOLUTE wall-clock deadline in SQLite; ``check_detour`` compares it to the clock, so
  sleep, lid-close and relaunch cannot lose it. A late reminder says it is late.
- The push-back and the detour reminder never carry a joke.
"""

from __future__ import annotations

import datetime as dt
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Optional

from . import heuristics as H
from . import roast as R
from .clock import Clock
from .config import Settings
from .db import Store
from .models import (
    Anchor,
    Composition,
    Intent,
    Observed,
    Policy,
    PolicyDraft,
    Register,
    ReplyIntent,
    Sentiment,
    State,
)

ACK_LINE = {
    "en": "Got it. I'll stay out of your way.",
    "hi": "ठीक है। मैं चुप रहूँगा।",
}
HOW_LONG_LINE = {
    "en": "How long do you need?",
    "hi": "कितना समय चाहिए?",
}
_STOP_WORDS = ["stop", "shut up", "leave me alone", "go away", "enough", "chup", "band karo", "bas karo",
               "चुप", "बंद करो", "बस करो"]
_JAB_WORDS = ["just a bot", "you're a bot", "youre a bot", "stupid bot", "dumb bot", "useless bot", "shut up bot",
              "you're just an app", "who asked you"]
_COMEBACKS = ["A bot that noticed. That's one of us.", "Bot, yes. Distracted, no.",
              "Small program. Still ahead on points."]


def _persona():
    try:
        from . import personality as P
        return P
    except Exception:
        return None


def is_stop(text: str) -> bool:
    P = _persona()
    if P is not None and hasattr(P, "is_stop"):
        try:
            return bool(P.is_stop(text))
        except Exception:
            pass
    low = (text or "").lower()
    return any(w in low for w in _STOP_WORDS)


def is_jab(text: str) -> bool:
    P = _persona()
    if P is not None and hasattr(P, "is_jab"):
        try:
            return bool(P.is_jab(text))
        except Exception:
            pass
    low = (text or "").lower()
    return any(w in low for w in _JAB_WORDS)


def apply_tempo(policy: Policy, settings: Settings) -> Policy:
    """Fast tempo: same policy shape, a fixed short patience, quicker pauses.

    ``settings.patience_s`` (default 12 s) wins for every task; ``None`` keeps the task-derived patience.
    ``ANCHOR_DEMO=1`` uses the demo patience. Applied to the in-memory policy only; the stored policy
    stays the task-derived one, so switching pace never rewrites history.
    """
    demo = getattr(settings, "demo", False)
    patience = DEMO_PATIENCE_S if demo else getattr(settings, "patience_s", None)
    if not patience:
        return policy
    policy.patience_seconds = int(patience)
    policy.pause_idle_s = None if policy.pause_idle_s is None else 2
    policy.pause_stable_s = 3
    policy.tolerance_seconds = 120     # a helper surface (search, AI chat, docs) still counts as the task for 2 min
    return policy


DEMO_PATIENCE_S = 10   # a drift is called 10–20 s after it starts (hard cap = 2× patience)


@dataclass
class ConfrontOutcome:
    intent: Intent
    minutes: Optional[int] = None
    new_anchor: Optional[str] = None
    pushed: bool = False
    joke_used: bool = False
    spoken: list[str] = field(default_factory=list)


class Conversation:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        llm,
        speaker,
        listener,
        clock: Optional[Clock] = None,
        composer=None,
        ladder=None,
        is_muted: Optional[Callable[[], bool]] = None,
    ):
        self.settings = settings
        self.store = store
        self.llm = llm
        self.speaker = speaker
        self.listener = listener
        self.clock = clock or Clock()
        self.composer = composer or R.Composer(llm, settings)
        self.ladder = ladder or R.RegisterLadder(store)
        self._is_muted = is_muted or (lambda: settings.silent)

        self.state: State = State.IDLE
        self.anchor: Optional[Anchor] = None
        self.policy: Optional[Policy] = None
        self.confrontations_this_anchor = 0
        self.transcript: deque[str] = deque(maxlen=30)
        self.last_line = ""
        self.last_confrontation_ts: Optional[float] = None
        self.comebacks_used: set[str] = set()

    # ------------------------------------------------------------------ helpers
    @property
    def muted(self) -> bool:
        return bool(self._is_muted())

    def now(self) -> float:
        return self.clock.now()

    def today(self) -> str:
        return dt.datetime.fromtimestamp(self.now()).strftime("%Y-%m-%d")

    def language(self) -> str:
        if self.anchor is not None:
            return self.anchor.language
        return self.store.get_setting("last_language", "en") or "en"

    def _set_state(self, state: State, detail: str = "") -> None:
        self.state = state
        anchor_id = self.anchor.id if self.anchor else None
        self.store.log_event(anchor_id, state.value, detail, self.now())

    def say(self, text: str, *, tone: str = "neutral", language: Optional[str] = None) -> None:
        text = (text or "").strip()
        if not text:
            return
        language = language or self.language()
        if self.muted:
            self.transcript.append(f"anchor (muted): {text}")
        else:
            self.transcript.append(f"anchor: {text}")
            try:
                self.speaker.speak(text, language=language, tone=tone)
            except Exception:
                pass
        self.last_line = text

    def hear(self, *, window_s: Optional[float] = None, language: Optional[str] = None) -> str:
        language = language or self.language()
        window = window_s if window_s is not None else self.settings.listen_window_s
        try:
            text = (self.listener.listen(window_s=window, language=language) or "").strip()
        except Exception:
            text = ""
        self.transcript.append(f"you: {text}" if text else "you: (silence)")
        if text:
            self.last_line = text
        return text

    def current_register(self) -> Register:
        default = Register(self.settings.default_register)
        try:
            return self.ladder.current(default, self.today())
        except Exception:
            return default

    def _cool_if_irritated(self, intent: ReplyIntent) -> None:
        if intent.sentiment == Sentiment.IRRITATED:
            try:
                self.ladder.cool(self.today(), self.current_register())
            except Exception:
                pass

    # ------------------------------------------------------------------ session
    def start_session(self) -> Optional[Anchor]:
        """Resume the active anchor from disk, or ask for one."""
        pending = self.store.active_anchor()
        fresh = getattr(self.settings, "fresh_start", False) or (
            getattr(self.settings, "demo", False) and pending is not None and pending.detour_until is None
        )
        if pending is not None and fresh:
            # Owner's choice: every launch starts with the question. Any pending break is cancelled with it.
            self.store.set_detour(pending.id, None, None)
            self.store.set_status(pending.id, "retired", self.now())
            self.store.log_event(pending.id, "RETIRED", "launch starts fresh", self.now())
            pending = None
        if pending is not None:
            self.anchor = pending
            self.policy = apply_tempo(
                self.store.load_policy(pending.id) or H.derive_policy_heuristic(
                    pending.clarified, Register(self.settings.default_register), self.settings.default_detour_minutes
                ).policy,
                self.settings,
            )
            self.confrontations_this_anchor = self.store.count_events("CONFRONTING", pending.created_at, pending.id)
            if pending.detour_until is not None:
                self._set_state(State.DETOUR, "resumed during detour")
            else:
                self._set_state(State.WATCHING, "resumed")
            return pending
        return self.anchoring()

    def _derive(self, sentence: str, language: str, clarification: Optional[str] = None) -> PolicyDraft:
        draft = None
        try:
            draft = self.llm.derive_policy(sentence, language, clarification) if self.llm else None
        except Exception:
            draft = None
        if draft is None:
            text = H.merge_clarification(sentence, clarification) if clarification else sentence
            draft = H.derive_policy_heuristic(
                text, Register(self.settings.default_register), self.settings.default_detour_minutes
            )
            if clarification:
                draft.needs_clarification = False
        # Local Devanagari/Hinglish detection wins for Hindi; otherwise trust the draft.
        if language == "hi":
            draft.language = "hi"
        draft.policy.language = draft.language
        draft.policy.default_detour_minutes = self.settings.default_detour_minutes
        if not draft.policy.humor_ok or H.is_sensitive(sentence):
            draft.policy.humor_ok = False
        return draft

    def anchoring(self, prompt: Optional[str] = None) -> Optional[Anchor]:
        """Ask aloud, take one sentence verbatim, ask at most ONE clarifying question."""
        self._set_state(State.ANCHORING)
        lang = self.language()
        if self.store.get_setting("first_run_done") != "1":
            self.say(R.disclosure(lang), tone="warm", language=lang)
            self.store.set_setting("first_run_done", "1")
        self.say(prompt or R.intake_question(lang), tone="warm", language=lang)
        reply = self.hear(window_s=self.settings.listen_window_s * 1.5, language=lang)
        if not reply:
            self.say(R.couldnt_hear(lang), tone="warm", language=lang)
            reply = self.hear(window_s=self.settings.listen_window_s * 1.5, language=lang)
        if not reply:
            self._set_state(State.IDLE, "no answer")
            return None
        return self.adopt(reply, allow_clarify=True)

    def adopt(self, sentence: str, *, allow_clarify: bool) -> Anchor:
        """Make ``sentence`` the anchor. ``allow_clarify`` is True only at intake, never on SWITCH."""
        language = H.detect_language(sentence)
        draft = self._derive(sentence, language)
        questions_asked = 0
        if allow_clarify and draft.needs_clarification and draft.clarifying_question:
            questions_asked += 1                      # exactly one; there is no loop here
            self.say(draft.clarifying_question, tone="warm", language=draft.language)
            answer = self.hear(language=draft.language)
            if answer:
                draft = self._derive(sentence, language, clarification=answer)
        assert questions_asked <= 1
        anchor = self.store.create_anchor(sentence, draft.clarified_anchor or sentence, draft.language, self.now())
        self.store.save_policy(anchor.id, draft.policy)
        self.store.set_setting("last_language", draft.language)
        self.anchor = anchor
        self.policy = apply_tempo(draft.policy, self.settings)
        self.confrontations_this_anchor = 0
        self.say(ACK_LINE.get(draft.language, ACK_LINE["en"]), tone="warm", language=draft.language)
        self._set_state(State.WATCHING, f"anchored: {sentence}")
        self.prefetch_lines()
        return anchor

    def prefetch_lines(self) -> None:
        """Pre-synthesize the fixed lines this anchor can need, in the background, so replies feel instant."""
        speaker = self.speaker
        if self.anchor is None or self.muted or not hasattr(speaker, "prefetch"):
            return
        lang = self.language()
        verbatim = self.anchor.verbatim
        lines = [
            (R.pushback_line(verbatim, lang), "flat"),
            (HOW_LONG_LINE.get(lang, HOW_LONG_LINE["en"]), "warm"),
            (R.accept_line(lang), "warm"),
            (R.detour_reminder(verbatim, None, lang), "warm"),
            (R.congrats(lang), "warm"),
            (R.whats_next(lang), "warm"),
        ]

        def work() -> None:
            for text, tone in lines:
                try:
                    speaker.prefetch(text, language=lang, tone=tone)
                except Exception:
                    return

        threading.Thread(target=work, name="anchor-prefetch", daemon=True).start()

    # ------------------------------------------------------------------ confrontation
    def may_confront(self, now: Optional[float] = None) -> bool:
        now = self.now() if now is None else now
        if self.anchor is None or self.policy is None:
            return False
        if self.anchor.detour_until is not None:
            return False
        if self.state not in (State.WATCHING, State.DRIFTING):
            return False
        recent = self.store.count_events("CONFRONTING", now - 3600)
        return recent < self.settings.max_confrontations_per_hour

    def comeback(self) -> str:
        """One short comeback when the user jabs at the companion; each line at most once per session."""
        lines: list[str] = []
        P = _persona()
        if P is not None:
            try:
                lines = list(P.load_personality().comebacks)
            except Exception:
                lines = []
        lines = lines or list(_COMEBACKS)
        for line in lines:
            if line not in self.comebacks_used:
                self.comebacks_used.add(line)
                return line
        return ""

    def classify(self, reply: str) -> ReplyIntent:
        if not reply or not reply.strip():
            return ReplyIntent(Intent.EVASIVE, language=self.language())
        if is_stop(reply):
            # "Stop" means stop: no push-back, no farewell joke, gentler for the rest of the day.
            return ReplyIntent(Intent.RESUME, sentiment=Sentiment.IRRITATED, language=H.detect_language(reply))
        default_minutes = self.policy.default_detour_minutes if self.policy else self.settings.default_detour_minutes
        now_local = dt.datetime.fromtimestamp(self.now()).strftime("%H:%M")
        local = H.classify_reply_keywords(reply, default_minutes, self.now())
        intent = None
        if local.intent != Intent.EVASIVE:
            intent = local                       # unambiguous locally: answer within a breath, no model round-trip
        else:
            try:
                intent = self.llm.classify_reply(reply, self.anchor, self.language(), default_minutes, now_local) if self.llm else None
            except Exception:
                intent = None
        if intent is None:
            intent = local
        if intent.intent == Intent.DEFER:
            # minutes stays None ONLY when the reply carries no amount at all ("I just need a short break"):
            # the branch then asks "How long?" once. Vague amounts ("a bit") take the default.
            local_minutes = H.parse_duration_minutes(reply, -1, self.now())
            if local_minutes is None:
                intent.minutes = None
            elif local_minutes == -1:
                intent.minutes = intent.minutes or default_minutes
            else:
                intent.minutes = intent.minutes or local_minutes
        if intent.intent == Intent.SWITCH and not (intent.new_anchor or "").strip():
            intent.new_anchor = (local.new_anchor or reply).strip()
        if local.sentiment == Sentiment.IRRITATED:
            intent.sentiment = Sentiment.IRRITATED
        if intent.intent == Intent.EVASIVE and local.intent == Intent.RESUME:
            intent.intent = Intent.RESUME          # "I'll go back to it" is an answer, not a dodge
        return intent

    def compose_for(self, observed: Observed) -> Composition:
        """The roast + choice line for the NEXT confrontation. Safe to call ahead of time from a worker thread."""
        assert self.anchor is not None and self.policy is not None
        lang = self.language()
        try:
            return self.composer.compose(
                self.anchor, observed, self.policy, self.current_register(), self.confrontations_this_anchor,
                lang, muted=self.muted,
            )
        except Exception:
            return Composition(R.plain_statement(self.anchor, observed, lang), R.choice_line(lang), False)

    def discard_composition(self, comp: Optional[Composition]) -> None:
        """Forget a line that was composed ahead of time but never spoken, so it neither counts as a recent
        roast (anti-repeat, model context) nor uses up an approved library line. Never raises."""
        if comp is None:
            return
        try:
            delivered = getattr(self.composer, "delivered", None)
            if isinstance(delivered, list) and comp.roast in delivered:
                del delivered[len(delivered) - 1 - delivered[::-1].index(comp.roast)]   # the most recent copy
            used = getattr(self.composer, "used_ids", None)
            if isinstance(used, set) and comp.roast_id:
                used.discard(comp.roast_id)
        except Exception:
            pass

    def confront(self, observed: Observed, precomposed: Optional[Composition] = None) -> ConfrontOutcome:
        """Roast + forced choice, ONE optional flat push-back, then accept whatever comes."""
        assert self.anchor is not None and self.policy is not None
        push_used = False                                  # owned by this function, never by the model
        spoken: list[str] = []
        self.last_confrontation_ts = self.now()
        self._set_state(State.CONFRONTING, observed.activity)
        lang = self.language()
        comp = precomposed or self.compose_for(observed)
        self.confrontations_this_anchor += 1
        line = f"{comp.roast} {comp.choice_line}".strip()
        tone = (comp.delivery or "roast") if comp.joke_used else "warm"
        self.store.log_event(
            self.anchor.id, "ROAST",
            f"joke={comp.joke_used} id={comp.roast_id} mechanism={comp.mechanism} delivery={comp.delivery} text={comp.roast}",
            self.now(),
        )
        self.say(line, tone=tone, language=lang)
        spoken.append(line)

        reply = self.hear(language=lang)
        if reply and is_jab(reply) and not is_stop(reply):
            back = self.comeback()                       # at most one, then the question stands
            if back:
                self.say(back, tone="deadpan", language=lang)
                spoken.append(back)
        intent = self.classify(reply)
        self._cool_if_irritated(intent)

        if intent.intent == Intent.EVASIVE and not push_used:
            push_used = True
            self._set_state(State.PUSHED)
            push = R.pushback_line(self.anchor.verbatim, lang)   # flat, verbatim, no joke — ever
            self.say(push, tone="flat", language=lang)
            spoken.append(push)
            reply = self.hear(language=lang)
            intent = self.classify(reply)
            self._cool_if_irritated(intent)
            if intent.intent == Intent.EVASIVE:
                intent = ReplyIntent(Intent.DEFER, minutes=self.policy.default_detour_minutes, language=lang)

        outcome = self.apply(intent)
        outcome.pushed = push_used
        outcome.joke_used = comp.joke_used
        outcome.spoken = spoken + outcome.spoken
        return outcome

    # ------------------------------------------------------------------ branches
    def apply(self, intent: ReplyIntent) -> ConfrontOutcome:
        assert self.anchor is not None and self.policy is not None
        lang = self.language()
        if intent.intent == Intent.DEFER:
            spoken: list[str] = []
            if intent.minutes is None:
                # No amount was given: ask once, then take whatever comes (silence → the default).
                # The break itself is already agreed, so its default deadline goes to disk BEFORE the question:
                # a quit or crash mid-answer still ends in a reminder instead of a lost break.
                default = max(1, int(self.policy.default_detour_minutes))
                self.anchor.detour_until = self.now() + default * 60
                self.anchor.detour_minutes = default
                self.store.set_detour(self.anchor.id, self.anchor.detour_until, default)
                ask = HOW_LONG_LINE.get(lang, HOW_LONG_LINE["en"])
                self.say(ask, tone="warm", language=lang)
                spoken.append(ask)
                answer = self.hear(language=lang)
                parsed = (
                    H.parse_duration_minutes(answer, self.policy.default_detour_minutes, self.now(), literal=True)
                    if answer else None
                )
                if parsed is None and answer:
                    llm_intent = None
                    try:
                        now_local = dt.datetime.fromtimestamp(self.now()).strftime("%H:%M")
                        llm_intent = self.llm.classify_reply(
                            answer, self.anchor, lang, self.policy.default_detour_minutes, now_local
                        ) if self.llm else None
                    except Exception:
                        llm_intent = None
                    if llm_intent is not None and llm_intent.intent == Intent.DEFER and llm_intent.minutes:
                        parsed = llm_intent.minutes
                intent.minutes = parsed or self.policy.default_detour_minutes
            minutes = int(intent.minutes or self.policy.default_detour_minutes)
            minutes = max(1, minutes)
            until = self.now() + minutes * 60
            self.store.set_detour(self.anchor.id, until, minutes)     # absolute deadline, on disk
            self.anchor.detour_until = until
            self.anchor.detour_minutes = minutes
            back_at = dt.datetime.fromtimestamp(until).strftime("%H:%M")
            line = R.detour_confirm(minutes, back_at, lang)
            self.say(line, tone="warm", language=lang)
            self._set_state(State.DETOUR, f"{minutes} min until {back_at}")
            return ConfrontOutcome(Intent.DEFER, minutes=minutes, spoken=spoken + [line])

        if intent.intent == Intent.SWITCH:
            new_sentence = (intent.new_anchor or "").strip()
            old = self.anchor
            self.store.set_status(old.id, "retired", self.now())
            self._set_state(State.SWITCH, new_sentence)
            self.adopt(new_sentence, allow_clarify=False)             # no follow-up question mid-flow
            return ConfrontOutcome(Intent.SWITCH, new_anchor=new_sentence)

        if intent.intent == Intent.RESUME:
            line = R.accept_line(lang)
            self.say(line, tone="warm", language=lang)
            self._set_state(State.WATCHING, "resumed the anchor")
            return ConfrontOutcome(Intent.RESUME, spoken=[line])

        if intent.intent == Intent.DONE:
            old = self.anchor
            self.store.set_status(old.id, "completed", self.now())
            self._set_state(State.DONE)
            line = R.congrats(lang)
            self.say(line, tone="warm", language=lang)
            self.anchor = None
            self.policy = None
            self.anchoring(prompt=R.whats_next(lang))
            return ConfrontOutcome(Intent.DONE, spoken=[line])

        # EVASIVE reaching here means the push-back was already used; accept the default detour.
        return self.apply(ReplyIntent(Intent.DEFER, minutes=self.policy.default_detour_minutes, language=lang))

    # ------------------------------------------------------------------ detour
    def check_detour(self, now: Optional[float] = None) -> bool:
        """Fire the plain reminder once the absolute deadline has passed. Returns True when it fired."""
        now = self.now() if now is None else now
        if self.anchor is None or self.anchor.detour_until is None:
            return False
        if now < self.anchor.detour_until:
            return False
        late_s = now - self.anchor.detour_until
        late_minutes = int(late_s // 60) if late_s > self.settings.late_threshold_s else None
        self.store.set_detour(self.anchor.id, None, None)
        self.anchor.detour_until = None
        self.anchor.detour_minutes = None
        lang = self.language()
        self._set_state(State.WATCHING, f"detour over (late {late_minutes or 0} min)")
        self.say(R.detour_reminder(self.anchor.verbatim, late_minutes, lang), tone="warm", language=lang)
        reply = self.hear(language=lang)
        if reply:
            intent = self.classify(reply)
            self._cool_if_irritated(intent)
            if intent.intent in (Intent.DEFER, Intent.SWITCH, Intent.DONE) and (
                intent.intent != Intent.DEFER or H.parse_duration_minutes(reply, 0, now) is not None
                or H.classify_reply_keywords(reply, 0, now).intent == Intent.DEFER
            ):
                self.apply(intent)
            elif intent.intent == Intent.RESUME:
                self.say(R.accept_line(lang), tone="warm", language=lang)
        return True
