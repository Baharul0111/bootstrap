"""The 1 Hz loop: sensor → change gate → verdict cache → judge → drift → pause gate → conversation.

Runs on a worker thread. The menu-bar shell only reads ``status()`` and flips ``muted``.
During a detour the loop does nothing but compare the stored deadline to the clock.
"""

from __future__ import annotations

import datetime as dt
import math
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

from . import heuristics as H
from .clock import Clock
from .config import Settings
from .db import Store
from .drift import DriftAccumulator
from .gate import ChangeGate, GateDecision
from .models import Blockers, ContextFrame, Observed, State, Verdict
from .pause import PauseGate, blocker_reason, may_speak
from .statemachine import Conversation

DOTS = {
    State.IDLE: "⚪",
    State.ANCHORING: "⚪",
    State.WATCHING: "🟢",
    State.DRIFTING: "🟠",
    State.CONFRONTING: "🔴",
    State.PUSHED: "🔴",
    State.DETOUR: "🔵",
    State.SWITCH: "🟢",
    State.DONE: "⚪",
}

WRONG_CALL_WINDOW_S = 600   # "wrong call" this soon after a roast means the roast, not a later on-task judgement


@dataclass
class EngineStatus:
    state: State
    dot: str
    last_line: str
    anchor_text: str
    muted: bool
    missing_permissions: list[str] = field(default_factory=list)
    drift_seconds: float = 0.0


class Transcript(deque):
    """The bounded transcript plus a count of every line ever appended and a consistent snapshot.

    A reader on the main thread can tell exactly which lines are new even after the deque has
    wrapped (30 identical-looking lines are no longer ambiguous), and never trips over
    "deque mutated during iteration" while the worker is appending.
    """

    def __init__(self, maxlen: int = 30) -> None:
        super().__init__(maxlen=maxlen)
        self.total = 0
        self._lock = threading.Lock()

    def append(self, line) -> None:
        with self._lock:
            super().append(line)
            self.total += 1

    def snapshot(self) -> tuple[list, int]:
        """(lines, total ever appended) read atomically."""
        with self._lock:
            return list(self), self.total


class Engine:
    def __init__(self, settings: Settings, store: Store, sensor, llm, speaker, listener, clock: Optional[Clock] = None):
        self.settings = settings
        self.store = store
        self.sensor = sensor
        self.llm = llm
        self.clock = clock or Clock()
        self._muted = bool(settings.silent)
        self.conv = Conversation(
            settings, store, llm, speaker, listener, self.clock, is_muted=lambda: self._muted
        )
        self.transcript: Transcript = Transcript(maxlen=30)
        self.conv.transcript = self.transcript    # one counted, lock-protected deque shared by both
        self.gate = ChangeGate(settings.hamming_threshold)
        self.drift: Optional[DriftAccumulator] = None
        self.pause: Optional[PauseGate] = None
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._started = False
        self.missing_permissions: list[str] = []
        self.last_frame: Optional[ContextFrame] = None
        self.last_decision: Optional[GateDecision] = None
        self.last_verdict: Optional[Verdict] = None
        self.last_fingerprint: str = ""
        self.drift_started_at: Optional[float] = None
        self._last_tick_ts: Optional[float] = None
        self._idle_retry_at: Optional[float] = None
        self._idle_retries = 0
        self.judge_calls = 0
        self.vision_calls = 0
        self.cache_hits = 0
        self._vision_at: dict[str, float] = {}
        self._precomposed: Optional[tuple[str, object]] = None
        self._precompose_thread: Optional[threading.Thread] = None
        self._precompose_gen = 0                       # bumped whenever a prepared line becomes stale
        self._confronted: Optional[tuple[str, Verdict, float]] = None   # (fingerprint, verdict, when) of the last roast
        self._detour_blocked_by = ""                   # why a due reminder is waiting; logged once per reason
        self.precompose_async = True

    # ------------------------------------------------------------------ lifecycle
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self.run_forever, name="anchor-engine", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Ask the worker to finish and wait up to 3 s. Never raises, even from the worker thread itself."""
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            try:
                thread.join(timeout=3.0)
            except RuntimeError:
                pass

    def run_forever(self) -> None:
        try:
            self.startup()
        except Exception as exc:  # never die silently; the dot shows IDLE
            self.transcript.append(f"anchor: startup problem: {exc}")
        while not self._stop.is_set():
            t0 = self.clock.now()
            try:
                self.tick()
            except Exception as exc:
                self.transcript.append(f"anchor: tick problem: {exc}")
            elapsed = self.clock.now() - t0
            self.clock.sleep(max(0.0, self.settings.tick_s - elapsed))

    def startup(self) -> None:
        self.store.open()
        self._started = True
        try:
            self.missing_permissions = list(self.sensor.missing_permissions())
        except Exception:
            self.missing_permissions = []
        lang = self.store.get_setting("last_language", "en") or "en"
        from . import roast as R
        for name in self.missing_permissions:
            self.conv.say(R.permission_missing(name, lang), tone="warm", language=lang)
        self.conv.start_session()
        self._rebuild_for_policy()

    def _rebuild_for_policy(self) -> None:
        policy = self.conv.policy
        if policy is None:
            self.drift = None
            self.pause = None
            return
        self.drift = DriftAccumulator(policy)
        self.pause = PauseGate(policy)
        self.gate.reset()
        self.drift_started_at = None
        self.last_verdict = None
        self.last_fingerprint = ""
        self._confronted = None
        self._drop_precomposed()

    # ------------------------------------------------------------------ status / controls
    def status(self) -> EngineStatus:
        with self._lock:
            state = self.conv.state
            anchor = self.conv.anchor
            drift = self.drift          # the worker may swap this for None mid-call; read it once
            dot = "🔇" if self._muted else DOTS.get(state, "⚪")
            return EngineStatus(
                state=state,
                dot=dot,
                last_line=self.conv.last_line,
                anchor_text=anchor.verbatim if anchor else "",
                muted=self._muted,
                missing_permissions=list(self.missing_permissions),
                drift_seconds=drift.seconds if drift is not None else 0.0,
            )

    @property
    def muted(self) -> bool:
        return self._muted

    def set_muted(self, muted: bool) -> None:
        self._muted = bool(muted)
        self.transcript.append("anchor: muted" if muted else "anchor: unmuted")

    def toggle_mute(self) -> bool:
        self.set_muted(not self._muted)
        return self._muted

    def wrong_call(self) -> None:
        """The user says the last call was wrong: record it, flip it, and forget the build-up.

        "The last call" is the roast when there was one recently and everything judged since is on-task
        (they went back to work before reaching the menu); otherwise it is the latest judgement."""
        with self._lock:
            anchor = self.conv.anchor
            fp = self.last_fingerprint
            v = self.last_verdict
            now = self.clock.now()
            roasted, self._confronted = self._confronted, None
            if roasted is not None and (v is None or not v.off_task) and now - roasted[2] <= WRONG_CALL_WINDOW_S:
                fp, v = roasted[0], roasted[1]
            if anchor is None or not fp or v is None:
                self.store.add_refinement(anchor.id if anchor else None, "", "The user flagged the last call as wrong.", now)
                self.transcript.append("you: wrong call (nothing to correct yet)")
                return
            was = "off-task" if v.off_task else "on-task"
            should = "on-task" if v.off_task else "off-task"
            correction = (
                f"Context '{fp}' was judged {was} ({v.reason or v.activity}); the user says that was WRONG. "
                f"Treat it as {should}."
            )
            self.store.add_refinement(anchor.id, fp, correction, now)
            flipped = Verdict(
                drift=1.0 - v.drift, confidence=0.9, reason="user correction", tolerated=False,
                source="refinement", activity=v.activity,
            )
            self.store.put_verdict(anchor.id, fp, flipped, self.last_frame.dhash if self.last_frame else "", now)
            if fp == self.last_fingerprint:
                self.last_verdict = flipped
            drift = self.drift          # read once: the worker may replace it while we hold the lock
            if v.off_task and drift is not None:
                drift.reset()
                self.drift_started_at = None
                self._drop_precomposed()
            self.transcript.append(f"you: wrong call → now treating '{fp.split('|')[0]}' as {should}")

    # ------------------------------------------------------------------ the loop body
    def tick(self) -> None:
        if not self._started:
            self.startup()
        now = self.clock.now()
        dt_s = 1.0 if self._last_tick_ts is None else min(max(now - self._last_tick_ts, 0.0), 5.0)
        self._last_tick_ts = now
        conv = self.conv

        if conv.state == State.DETOUR:
            # Fully paused: no sensing, no judging, no cost. Only the absolute deadline is compared — and once it
            # is due, the reminder waits like any other speech for a call, Focus, a locked screen or Mute to end
            # (it then says how late it is).
            due = conv.anchor is not None and conv.anchor.detour_until is not None and now >= conv.anchor.detour_until
            if due:
                waiting: Blockers = self.sensor.blockers()
                waiting.muted = self._muted
                if not may_speak(waiting):
                    reason = blocker_reason(waiting)
                    if reason != self._detour_blocked_by:
                        self._detour_blocked_by = reason
                        self.store.log_event(conv.anchor.id, "SUPPRESSED", f"reminder waiting: {reason}", now)
                    return
            self._detour_blocked_by = ""
            if conv.check_detour(now):
                self._rebuild_for_policy()
            return

        if conv.anchor is None or conv.policy is None:
            if conv.state in (State.IDLE, State.DONE):
                # Nobody answered: ask again once after 30 s, then only every five minutes. Asking is not nagging,
                # but an empty room should not be talked at all afternoon.
                if self._idle_retry_at is None:
                    self._idle_retries = 0
                    self._idle_retry_at = now + 30
                elif now >= self._idle_retry_at:
                    self._idle_retries += 1
                    self._idle_retry_at = now + 300
                    if conv.anchoring() is not None:
                        self._idle_retry_at = None
                    self._rebuild_for_policy()
            return

        if self.drift is None or self.pause is None:
            self._rebuild_for_policy()
        assert self.drift is not None and self.pause is not None

        frame = self.sensor.sample()
        decision = self.gate.update(frame)
        with self._lock:
            self.last_frame = frame
            self.last_decision = decision

        if decision.changed:
            verdict = self.judge_context(frame, decision, now)
            with self._lock:
                self.last_verdict = verdict
                self.last_fingerprint = decision.fingerprint
        else:
            verdict = self.last_verdict           # nothing happened: think nothing, send nothing

        before = self.drift.seconds
        if verdict is None:
            self.drift.hold(dt_s)
        else:
            self.drift.tick(verdict, dt_s)
        if before == 0 and self.drift.seconds > 0:
            self.drift_started_at = now
        elif self.drift.seconds == 0:
            self.drift_started_at = None
            self._drop_precomposed()                   # they came back before the gap: a prepared line is stale

        ready = self.pause.update(frame, decision, self.drift.seconds, dt_s)
        conv.state = State.DRIFTING if self.drift.seconds > 0 else State.WATCHING

        halfway = self.drift.seconds >= 0.5 * self.conv.policy.patience_seconds
        if (self.drift.fired or halfway) and verdict is not None and verdict.off_task and not verdict.tolerated:
            self._precompose(frame, decision.fingerprint, verdict, now)   # have the line (and its audio) ready early
        if not self.drift.fired or not ready:
            return

        blockers: Blockers = self.sensor.blockers()
        blockers.muted = self._muted
        if not may_speak(blockers):
            self.store.log_event(conv.anchor.id, "SUPPRESSED", blocker_reason(blockers), now)
            return
        if not conv.may_confront(now):
            self.store.log_event(conv.anchor.id, "CEILING", "4 per hour reached", now)
            self.drift.drain(self.settings.tick_s * 60)   # cool down instead of retrying every second
            return
        last = conv.last_confrontation_ts
        if last is not None and now - last < self.settings.roast_cooldown_s:
            return                                          # one unsolicited roast per episode, then a cooldown

        observed = self._observed(frame, verdict, now)
        precomposed = self._take_precomposed(decision.fingerprint)
        with self._lock:
            self._confronted = (decision.fingerprint, verdict, now) if verdict is not None else None
        outcome = conv.confront(observed, precomposed=precomposed)
        self.drift.reset()
        self.pause.reset()
        self.drift_started_at = None
        self._drop_precomposed()                       # anything still in flight was for the episode that just ended
        if outcome.intent.value in ("SWITCH", "DONE"):
            self._rebuild_for_policy()

    # ------------------------------------------------------------------ speaking ahead of the gap
    def _observed(self, frame: ContextFrame, verdict: Optional[Verdict], now: float) -> Observed:
        minutes = max(1, int(math.ceil(max(0.0, now - (self.drift_started_at or now)) / 60)))
        activity = (verdict.activity if verdict and verdict.activity else "") or H.describe_activity(frame)
        return Observed(
            activity=activity, app=frame.app, title=frame.title, url_domain=frame.url_domain,
            minutes_off_task=minutes, confidence=verdict.confidence if verdict else 0.5,
        )

    def _precompose(self, frame: ContextFrame, fingerprint: str, verdict: Optional[Verdict], now: float) -> None:
        """Compose the roast (and synthesize its audio) while we wait for a natural pause, so the gap is not wasted."""
        with self._lock:
            if self._precomposed is not None and self._precomposed[0] == fingerprint:
                return
            if self._precompose_thread is not None and self._precompose_thread.is_alive():
                return
            observed = self._observed(frame, verdict, now)
            conv = self.conv
            gen = self._precompose_gen

            def work() -> None:
                try:
                    comp = conv.compose_for(observed)
                except Exception:
                    return
                with self._lock:
                    fresh = gen == self._precompose_gen
                    if fresh:
                        self._precomposed = (fingerprint, comp)
                if not fresh:
                    conv.discard_composition(comp)     # the episode ended while composing: never speak this
                    return
                speaker = conv.speaker
                if hasattr(speaker, "prefetch") and not conv.muted:
                    try:
                        speaker.prefetch(f"{comp.roast} {comp.choice_line}".strip(), language=conv.language(),
                                         tone="roast" if comp.joke_used else "warm")
                    except Exception:
                        pass

            if self.precompose_async:
                self._precompose_thread = threading.Thread(target=work, name="anchor-precompose", daemon=True)
                self._precompose_thread.start()
            else:
                work()

    def _take_precomposed(self, fingerprint: str):
        t = self._precompose_thread
        if t is not None and t.is_alive():
            t.join(timeout=3.0)
        with self._lock:
            item = self._precomposed
            self._precomposed = None
        if item is not None and item[0] == fingerprint:
            return item[1]
        if item is not None:
            self.conv.discard_composition(item[1])     # prepared for a context they have since left
        return None

    def _drop_precomposed(self) -> None:
        """Forget any prepared line, held or still in flight: the episode it was written for is over."""
        with self._lock:
            self._precompose_gen += 1
            item, self._precomposed = self._precomposed, None
        if item is not None:
            self.conv.discard_composition(item[1])

    # ------------------------------------------------------------------ judging
    def judge_context(self, frame: ContextFrame, decision: GateDecision, now: float) -> Verdict:
        anchor, policy = self.conv.anchor, self.conv.policy
        assert anchor is not None and policy is not None
        fp = decision.fingerprint
        if decision.force_vision:
            last_look = self._vision_at.get(fp)
            if last_look is not None and now - last_look < self.settings.vision_cooldown_s and self.last_verdict is not None:
                return self.last_verdict                       # a playing video is not a new context every second
            self._vision_at[fp] = now
            self.store.invalidate_verdict(anchor.id, fp)      # content changed under the same title
        else:
            cached = self.store.get_verdict(anchor.id, fp, now, self.settings.cache_ttl_s)
            if cached is not None:
                self.cache_hits += 1
                return cached[0]

        day_start = dt.datetime.fromtimestamp(now).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        refinements = self.store.refinements_since(day_start)
        needs_vision = decision.force_vision or H.is_uninformative_title(frame.title, frame.app)
        verdict: Optional[Verdict] = None

        if not needs_vision:
            self.judge_calls += 1
            verdict = self._llm_judge(anchor, policy, frame, refinements, None)
            if verdict is not None and (verdict.confidence < 0.6 or 0.35 <= verdict.drift <= 0.65):
                needs_vision = True

        if needs_vision:
            image = None
            try:
                image = self.sensor.grab_jpeg()
            except Exception:
                image = None
            if image:
                self.vision_calls += 1
                v2 = self._llm_judge(anchor, policy, frame, refinements, image)
                image = None                                   # forget it immediately
                if v2 is not None:
                    verdict = v2

        if verdict is None:
            verdict = H.judge_heuristic(anchor.clarified or anchor.verbatim, policy, frame)
        if not verdict.activity:
            verdict.activity = H.describe_activity(frame)
        if verdict.drift <= 0.05:
            verdict.tolerated = False      # this IS the task, not a helper surface: it must drain, never expire
        self.store.put_verdict(anchor.id, fp, verdict, frame.dhash, now)
        self.store.log_event(anchor.id, "VERDICT", f"{fp} drift={verdict.drift:.2f} conf={verdict.confidence:.2f} {verdict.source}", now)
        return verdict

    def _llm_judge(self, anchor, policy, frame, refinements, image) -> Optional[Verdict]:
        if self.llm is None:
            return None
        try:
            return self.llm.judge(anchor, policy, frame, refinements, image)
        except Exception:
            return None


def build_engine(settings: Optional[Settings] = None) -> Engine:
    """Wire the real macOS sensor, OpenAI models, voice and the SQLite store."""
    settings = settings or Settings.from_env()
    from .llm import LLM
    from .sensors import MacSensor
    from .voice import make_voice

    store = Store(settings.db_path)
    sensor = MacSensor(settings)
    llm = LLM(settings)
    speaker, listener = make_voice(settings)
    return Engine(settings, store, sensor, llm, speaker, listener)
