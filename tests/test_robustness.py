"""Robustness: thread safety, SQLite corruption and contention, slow sensors, deaf or mute voice paths,
the transcript tail after the deque wraps, run.sh, and the doctor's key hygiene.

Hermetic except ``test_doctor_subprocess_never_prints_the_key``, which runs the real doctor once.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from anchor import sensors as S
from anchor.clock import FakeClock
from anchor.config import Settings
from anchor.db import Store
from anchor.engine import Engine
from anchor.llm import FakeLLM
from anchor.models import State, Verdict
from anchor.sensors import FakeSensor, MacSensor
from anchor.voice import (
    FakeListener,
    FakeSpeaker,
    FileListener,
    OpenAIListener,
    OpenAISpeaker,
    SayFallbackSpeaker,
)
from tests.conftest import frame

PROJECT_ROOT = Path(__file__).resolve().parent.parent
GOAL = "finish the assignment tonight"
OFF = Verdict(drift=0.95, confidence=0.95, reason="crabs", activity="researching whether crabs can swim on Google")
ON = Verdict(drift=0.05, confidence=0.95, reason="editing", activity="editing the assignment in Word")


# --------------------------------------------------------------------------- #
# 1. Engine thread safety
# --------------------------------------------------------------------------- #


class BlockingListener:
    """Answers the goal question once, then every ``listen`` blocks like a real microphone until released."""

    def __init__(self, goal: str = GOAL) -> None:
        self.replies = [goal]
        self.entered = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def listen(self, *, window_s: float = 8.0, language: str = "en") -> str:
        self.calls += 1
        if self.replies:
            return self.replies.pop(0)
        self.entered.set()
        self.release.wait(window_s)
        return ""


class FlakySensor(FakeSensor):
    """``sample()`` raises on every second call."""

    def sample(self):
        if self.sample_calls % 2 == 1:
            self.sample_calls += 1
            raise RuntimeError("sensor hiccup")
        return super().sample()


class StoppingClock(FakeClock):
    """Fake clock that asks the engine to stop after ``limit`` loop iterations."""

    def __init__(self, engine_ref: dict, limit: int) -> None:
        super().__init__()
        self.ref = engine_ref
        self.limit = limit
        self.sleeps = 0

    def sleep(self, seconds: float) -> None:
        super().sleep(seconds)
        self.sleeps += 1
        if self.sleeps >= self.limit:
            self.ref["engine"]._stop.set()


def _demo(settings: Settings) -> Settings:
    settings.demo = True                      # patience 10 s, pause after 3 still seconds
    settings.max_confrontations_per_hour = 12
    settings.roast_cooldown_s = 0
    return settings


def _engine(settings, store, *, listener=None, sensor=None, clock=None, verdicts=None):
    sensor = sensor or FakeSensor([frame(app="Google Chrome", title="crabs can swim? - Google Search",
                                         domain="google.com", dhash="f" * 16)])
    llm = FakeLLM(verdicts=list(verdicts if verdicts is not None else [OFF] * 500))
    eng = Engine(settings, store, sensor, llm, FakeSpeaker(), listener or FakeListener([GOAL]), clock or FakeClock())
    return eng, sensor, llm


def _record_tick_errors(eng: Engine) -> list:
    errors: list = []
    original = eng.tick

    def tick():
        try:
            original()
        except Exception as exc:  # run_forever swallows these; keep a copy for the assertion
            errors.append(exc)
            raise

    eng.tick = tick  # type: ignore[method-assign]
    return errors


def test_stop_during_confrontation_returns_within_three_seconds(settings, store):
    listener = BlockingListener()
    eng, _, _ = _engine(_demo(settings), store, listener=listener)
    eng.start()
    assert listener.entered.wait(10), "the engine never reached a confrontation"
    assert eng.status().state == State.CONFRONTING

    started = time.monotonic()
    eng.stop()                                   # the listener is still blocked inside conv.confront
    elapsed = time.monotonic() - started
    assert elapsed < 3.6, f"stop() took {elapsed:.2f}s"
    assert eng._stop.is_set()
    assert eng._thread.is_alive(), "the worker is still inside the listener; stop() must not wait for it"

    listener.release.set()                       # the microphone window ends; the worker finishes the tick and exits
    eng._thread.join(5)
    assert not eng._thread.is_alive()
    started = time.monotonic()
    eng.stop()                                   # a second stop never raises and is instant once the worker is gone
    assert time.monotonic() - started < 0.5


def test_second_start_is_a_noop_and_stop_before_start_is_safe(settings, store):
    eng, _, _ = _engine(_demo(settings), store)
    eng.stop()                                   # nothing running yet
    eng.start()
    first = eng._thread
    eng.start()
    assert eng._thread is first
    assert sum(1 for t in threading.enumerate() if t.name == "anchor-engine") == 1
    eng.stop()
    first.join(3)
    assert not first.is_alive()


def test_stop_called_from_the_worker_thread_does_not_raise(settings, store):
    eng, _, _ = _engine(_demo(settings), store)
    errors = _record_tick_errors(eng)
    inner = eng.tick

    def tick_then_stop():
        inner()
        eng.stop()                               # joining oneself would raise RuntimeError without the guard

    eng.tick = tick_then_stop  # type: ignore[method-assign]
    eng.start()
    eng._thread.join(5)
    assert not eng._thread.is_alive()
    assert errors == []


def test_controls_from_main_thread_while_worker_ticks(settings, store):
    eng, _, _ = _engine(_demo(settings), store, verdicts=[OFF, ON] * 400)
    errors = _record_tick_errors(eng)
    eng.start()
    deadline = time.monotonic() + 0.6
    calls = 0
    while time.monotonic() < deadline:
        status = eng.status()
        assert status.dot and isinstance(status.muted, bool)
        eng.set_muted(calls % 2 == 0)
        eng.toggle_mute()
        eng.wrong_call()
        calls += 1
    assert calls > 20
    assert eng._thread.is_alive()
    eng.stop()
    assert errors == [], f"tick raised under concurrent controls: {errors[:3]}"
    assert not any("problem" in line for line in eng.transcript)


def test_run_forever_survives_startup_and_tick_exceptions(settings, store):
    ref: dict = {}
    clock = StoppingClock(ref, limit=14)
    sensor = FlakySensor([frame(app="Google Chrome", title="crabs", domain="google.com", dhash="f" * 16)])
    eng, _, llm = _engine(_demo(settings), store, sensor=sensor, clock=clock)
    ref["engine"] = eng
    real_startup = eng.startup
    startups = {"n": 0}

    def flaky_startup():
        startups["n"] += 1
        if startups["n"] == 1:
            raise RuntimeError("store exploded on first open")
        real_startup()

    eng.startup = flaky_startup  # type: ignore[method-assign]
    eng.start()
    eng._thread.join(10)
    assert not eng._thread.is_alive(), "run_forever did not come back after the clock asked it to stop"

    lines = list(eng.transcript)
    assert any("startup problem" in line for line in lines)
    assert sum("tick problem" in line for line in lines) >= 5
    assert sensor.sample_calls >= 10
    assert len([c for c in llm.calls if c[0] == "judge"]) >= 1      # the good samples were still judged
    assert eng.status().state in (State.WATCHING, State.DRIFTING, State.CONFRONTING, State.DETOUR)


def test_status_tolerates_drift_being_cleared_by_the_worker(settings, store):
    eng, _, _ = _engine(settings, store)
    eng.startup()
    eng.tick()
    assert eng.status().drift_seconds >= 0
    eng.drift = None                             # what _rebuild_for_policy does once the goal is DONE
    assert eng.status().drift_seconds == 0.0
    eng.wrong_call()                             # must not touch a drift that is gone


# --------------------------------------------------------------------------- #
# 2. SQLite store
# --------------------------------------------------------------------------- #


def test_concurrent_writes_from_two_threads_do_not_corrupt(store):
    anchor = store.create_anchor("write the report", "write the report", "en", 1.0)
    errors: list = []

    def writer(tag: str) -> None:
        try:
            for i in range(300):
                store.log_event(anchor.id, "VERDICT", f"{tag}-{i}", float(i))
                store.put_verdict(anchor.id, f"{tag}|{i % 7}", OFF, "0" * 16, float(i))
                store.set_setting(f"k-{tag}", str(i))
        except Exception as exc:
            errors.append(exc)

    threads = [threading.Thread(target=writer, args=(t,)) for t in ("a", "b")]
    for t in threads:
        t.start()
    for t in threads:
        t.join(30)
    assert errors == []
    assert store.count_events("VERDICT", 0.0) == 600
    assert store.conn.execute("SELECT COUNT(*) FROM verdicts").fetchone()[0] == 14
    assert store.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_garbage_file_is_renamed_aside_and_recreated(tmp_path):
    path = tmp_path / "anchor.db"
    path.write_bytes(b"this is not a database" * 100)
    st = Store(path).open()
    try:
        aside = tmp_path / "anchor.db.corrupt"
        assert aside.exists() and aside.read_bytes().startswith(b"this is not")
        assert st.create_anchor("x", "x", "en", 1.0).id == 1
        assert st.active_anchor() is not None
    finally:
        st.close()

    path.write_bytes(b"\x00" * 4096)              # a second corruption must not clobber the first copy
    st = Store(path).open()
    try:
        assert (tmp_path / "anchor.db.corrupt1").exists()
        assert st.active_anchor() is None
    finally:
        st.close()


def test_subtle_corruption_reported_by_integrity_check_is_renamed_aside(tmp_path):
    path = tmp_path / "anchor.db"
    st = Store(path).open()
    st.create_anchor("old goal", "old goal", "en", 1.0)
    st.close()
    with open(path, "r+b") as fh:                # a bogus freelist count: integrity_check returns rows, never raises
        fh.seek(36)
        fh.write((5).to_bytes(4, "big"))
    st = Store(path).open()
    try:
        assert (tmp_path / "anchor.db.corrupt").exists()
        assert st.active_anchor() is None
        assert st.conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        st.close()


def test_set_detour_none_clears_both_columns(store):
    anchor = store.create_anchor("x", "x", "en", 1.0)
    store.set_detour(anchor.id, 1_800_000_900.0, 15)
    got = store.get_anchor(anchor.id)
    assert (got.detour_until, got.detour_minutes) == (1_800_000_900.0, 15)
    store.set_detour(anchor.id, None)
    got = store.get_anchor(anchor.id)
    assert got.detour_until is None and got.detour_minutes is None


def test_count_events_window_is_inclusive_at_since(store):
    a = store.create_anchor("a", "a", "en", 1.0)
    b = store.create_anchor("b", "b", "en", 1.0)
    for ts in (100.0, 200.0, 300.0):
        store.log_event(a.id, "CONFRONTING", "", ts)
    store.log_event(b.id, "CONFRONTING", "", 200.0)
    store.log_event(a.id, "SUPPRESSED", "", 200.0)
    assert store.count_events("CONFRONTING", 200.0) == 3
    assert store.count_events("CONFRONTING", 200.0000001) == 1
    assert store.count_events("CONFRONTING", 300.0) == 1
    assert store.count_events("CONFRONTING", 300.5) == 0
    assert store.count_events("CONFRONTING", 200.0, anchor_id=a.id) == 2
    assert store.count_events("CONFRONTING", 0.0, anchor_id=b.id) == 1
    assert store.count_events("ROAST", 0.0) == 0


# --------------------------------------------------------------------------- #
# 3. Sensors
# --------------------------------------------------------------------------- #


def _chrome_sensor(monkeypatch, settings: Settings | None = None) -> MacSensor:
    sensor = MacSensor(settings or Settings())
    monkeypatch.setattr(sensor, "_frontmost", lambda: ("Google Chrome", "com.google.Chrome", 4242))
    monkeypatch.setattr(sensor, "_title_cg", lambda pid: "")
    monkeypatch.setattr(sensor, "_screen_hash", lambda: ("", False))
    monkeypatch.setattr(sensor, "_idle_s", lambda: 0.0)
    return sensor


def test_sample_with_hanging_osascript_stays_under_the_cap_and_caches_the_url_failure(monkeypatch):
    knobs = SimpleNamespace(delay=1.5, calls=[])

    def hanging_run(cmd, **kwargs):
        knobs.calls.append(cmd)
        time.sleep(min(knobs.delay, float(kwargs.get("timeout", 1.5))))
        raise subprocess.TimeoutExpired(cmd, kwargs.get("timeout", 1.5))

    monkeypatch.setattr(S.subprocess, "run", hanging_run)
    monkeypatch.setattr(S, "_ax_trusted", lambda: True)   # the System Events title read runs osascript too
    sensor = _chrome_sensor(monkeypatch)

    t0 = time.perf_counter()
    first = sensor.sample()
    t_first = time.perf_counter() - t0
    assert first.app == "Google Chrome" and first.title == "" and first.url_domain == ""
    assert len(knobs.calls) == 2, "title and URL each cost one osascript on the first sample"
    assert t_first < 3.5

    t0 = time.perf_counter()
    sensor.sample()
    t_second = time.perf_counter() - t0
    assert len(knobs.calls) == 3, "the URL failure is cached; only the title was retried"
    assert t_second < 3.5

    knobs.delay = 0.0
    sensor._url_failures["com.google.Chrome"] -= S.URL_FAILURE_TTL_S + 1   # sixty seconds later
    sensor.sample()
    assert len(knobs.calls) == 5, "after the TTL the URL is asked for again"


def test_exclude_titles_sends_neither_title_nor_url_domain(monkeypatch):
    settings = Settings()
    settings.exclude_titles = True
    sensor = _chrome_sensor(monkeypatch, settings)
    asked: list = []
    monkeypatch.setattr(S, "_ax_trusted", lambda: True)
    monkeypatch.setattr(sensor, "_title_ax", lambda: "How to swim - YouTube")
    monkeypatch.setattr(sensor, "_browser_url", lambda b, e: asked.append(b) or "https://www.youtube.com/watch")
    monkeypatch.setattr(S.subprocess, "run", MagicMock(side_effect=AssertionError("osascript must not run")))

    got = sensor.sample()
    assert got.app == "Google Chrome"
    assert got.title == "" and got.title_available is True
    assert got.url_domain == ""
    assert asked == [], "app-name-only mode must not even ask the browser"


@pytest.mark.skipif(S.Image is None or S.dhash is None, reason="needs Pillow and dhash")
def test_grab_jpeg_returns_none_after_a_failed_capture_not_a_stale_frame(monkeypatch):
    monkeypatch.setattr(S, "MAC_AVAILABLE", True)
    monkeypatch.setattr(S, "IMAGING_AVAILABLE", True)
    monkeypatch.setattr(S, "_screen_capture_allowed", lambda: True)
    sensor = MacSensor(Settings())
    monkeypatch.setattr(sensor, "_frontmost", lambda: ("TestApp", "com.test.app", 1))
    monkeypatch.setattr(sensor, "_read_title", lambda pid: ("t", True))
    monkeypatch.setattr(sensor, "_idle_s", lambda: 0.0)
    image = S.Image.fromarray(np.full((32, 48, 3), 200, dtype=np.uint8))
    monkeypatch.setattr(sensor, "_capture", lambda: image)

    good = sensor.sample()
    assert good.screen_available and len(good.dhash) == 16
    assert sensor.grab_jpeg()[:2] == b"\xff\xd8"

    def capture_boom():
        raise RuntimeError("CGWindowListCreateImage failed")

    monkeypatch.setattr(sensor, "_capture", capture_boom)
    bad = sensor.sample()
    assert bad.screen_available is False and bad.dhash == ""
    assert sensor.grab_jpeg() is None, "a failed capture must not hand out the previous screen"

    monkeypatch.setattr(sensor, "_capture", lambda: None)
    sensor.sample()
    assert sensor.grab_jpeg() is None


# --------------------------------------------------------------------------- #
# 4. Voice
# --------------------------------------------------------------------------- #


def _fake_audio(monkeypatch, *, vad_raises=False, mic_raises=False, speech=False):
    writes: list = []

    class Stream:
        def __init__(self, *a, **k):
            if mic_raises:
                raise OSError("no default input device")

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, frames):
            time.sleep(frames / 16000)
            return np.zeros((frames, 1), dtype=np.int16), False

        def write(self, data):
            writes.append(bytes(data))

    class Vad:
        def __init__(self, mode):
            pass

        def is_speech(self, frame, rate):
            if vad_raises:
                raise RuntimeError("vad rejected the frame")
            return speech

    monkeypatch.setitem(sys.modules, "sounddevice", SimpleNamespace(InputStream=Stream, RawOutputStream=Stream))
    monkeypatch.setitem(sys.modules, "webrtcvad", SimpleNamespace(Vad=Vad))
    return writes


def test_listener_returns_empty_fast_when_the_vad_raises(monkeypatch):
    _fake_audio(monkeypatch, vad_raises=True)
    client = MagicMock()
    t0 = time.monotonic()
    assert OpenAIListener(Settings(), client=client).listen(window_s=2.0) == ""
    assert time.monotonic() - t0 < 1.0
    client.audio.transcriptions.create.assert_not_called()


def test_listener_silent_stream_is_capped_at_the_window(monkeypatch):
    _fake_audio(monkeypatch, speech=False)
    client = MagicMock()
    t0 = time.monotonic()
    assert OpenAIListener(Settings(), client=client).listen(window_s=0.4) == ""
    assert time.monotonic() - t0 < 0.4 + 0.5
    client.audio.transcriptions.create.assert_not_called()


def test_listener_returns_empty_when_the_microphone_cannot_open(monkeypatch):
    _fake_audio(monkeypatch, mic_raises=True)
    assert OpenAIListener(Settings(), client=MagicMock()).listen(window_s=1.0) == ""


def test_say_fallback_never_raises_when_say_is_missing(monkeypatch):
    monkeypatch.setattr(S.subprocess, "run", MagicMock(side_effect=FileNotFoundError("say")))
    spk = SayFallbackSpeaker()
    spk.speak("hello", language="en")
    spk.speak("नमस्ते", language="hi", tone="roast")
    assert spk._has_lekha() is False
    spk.speak("", language="en")


def test_file_listener_consumes_one_line_per_call_and_tolerates_a_missing_file(tmp_path):
    path = tmp_path / "replies.txt"
    path.write_text("give me ten minutes\n\nthis is the new goal\n", encoding="utf-8")
    listener = FileListener(str(path))
    assert listener.listen(window_s=0.5) == "give me ten minutes"
    assert path.read_text(encoding="utf-8") == "\nthis is the new goal\n"
    assert listener.listen(window_s=1.0) == "this is the new goal"    # the blank line is skipped, not returned
    assert path.read_text(encoding="utf-8") == ""
    t0 = time.monotonic()
    assert listener.listen(window_s=0.5) == ""                        # silence, after the window
    assert 0.4 < time.monotonic() - t0 < 1.5

    missing = FileListener(str(tmp_path / "nope" / "replies.txt"))
    assert missing.listen(window_s=0.5) == ""
    folder = FileListener(str(tmp_path))                              # a directory is not a reply file either
    assert folder.listen(window_s=0.5) == ""
    assert len(listener.calls) == 3


class _TTSResponse:
    def __init__(self, pcm: bytes) -> None:
        self.pcm = pcm

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_bytes(self, size):
        yield self.pcm


def test_prefetch_cache_is_bounded_and_speak_pops(monkeypatch):
    writes = _fake_audio(monkeypatch)
    client = MagicMock()
    client.audio.speech.with_streaming_response.create.side_effect = (
        lambda **kw: _TTSResponse(f"pcm:{kw['input']}".encode().ljust(8, b"\0"))
    )
    spk = OpenAISpeaker(Settings(), client=client, fallback=FakeSpeaker())
    for i in range(40):
        assert spk.prefetch(f"line {i}", language="en", tone="warm")
    create = client.audio.speech.with_streaming_response.create
    assert create.call_count == 40
    assert len(spk._cache) == 32
    assert ("line 0", "en", "warm") not in spk._cache and ("line 8", "en", "warm") in spk._cache
    assert spk.prefetch("line 39", language="en", tone="warm") and create.call_count == 40   # already cached

    spk.speak("line 39", language="en", tone="warm")
    assert create.call_count == 40 and writes[-1].startswith(b"pcm:line 39")
    assert ("line 39", "en", "warm") not in spk._cache and len(spk._cache) == 31
    spk.speak("line 39", language="en", tone="warm")
    assert create.call_count == 41, "a popped line is synthesized again, never replayed from the cache"
    assert spk.fallback.calls == []


# --------------------------------------------------------------------------- #
# 5. run.sh
# --------------------------------------------------------------------------- #

RUN_SH = PROJECT_ROOT / "run.sh"
VENV_PYTHON = PROJECT_ROOT / ".venv" / "bin" / "python"


def test_run_sh_parses_under_bash():
    assert subprocess.run(["bash", "-n", str(RUN_SH)], capture_output=True, text=True).returncode == 0


def test_run_sh_fallback_dependency_list_has_no_quotes():
    script = f'eval "$(sed -n \'/^deps_from_pyproject()/,/^}}/p\' "{RUN_SH}")"; deps_from_pyproject'
    proc = subprocess.run(["bash", "-c", script], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    lines = proc.stdout.splitlines()
    assert "openai>=1.60" in lines and "pytest" in lines
    assert all(line.strip() and '"' not in line for line in lines), "one bare requirement per line"


@pytest.mark.skipif(not VENV_PYTHON.exists(), reason="needs the project .venv")
def test_run_sh_version_works_from_another_cwd(tmp_path):
    proc = subprocess.run([str(RUN_SH), "version"], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().startswith("anchor ")


@pytest.mark.skipif(not VENV_PYTHON.exists(), reason="needs the project .venv")
def test_run_sh_version_does_not_need_a_key(tmp_path, monkeypatch):
    env = {k: v for k, v in os.environ.items() if k not in ("OPENAI_API_KEY", "ANCHOR_ENV")}
    env["ANCHOR_ENV"] = str(tmp_path / "absent.env")          # no .env anywhere the script looks
    proc = subprocess.run([str(RUN_SH), "version"], cwd=tmp_path, capture_output=True, text=True, timeout=60, env=env)
    assert proc.returncode == 0, proc.stderr


# --------------------------------------------------------------------------- #
# 6. Menu-bar shell: the transcript tail never repeats or skips a line
# --------------------------------------------------------------------------- #


class _TailEngine:
    def __init__(self, counted: bool = True) -> None:
        from collections import deque

        from anchor.engine import Transcript

        self.transcript = Transcript(maxlen=30) if counted else deque(maxlen=30)
        self.muted = False

    def status(self):
        return SimpleNamespace(state=State.WATCHING, dot="🟢", last_line="", anchor_text="", muted=self.muted,
                               missing_permissions=[])

    def start(self): ...
    def stop(self): ...
    def toggle_mute(self):
        self.muted = not self.muted
        return self.muted

    def wrong_call(self): ...


@pytest.mark.parametrize("pattern, counted", [("distinct", True), ("alternating", True), ("distinct", False)])
def test_refresh_two_hundred_times_prints_every_line_exactly_once(pattern, counted):
    import io

    from anchor.app import AnchorApp

    engine = _TailEngine(counted=counted)
    out = io.StringIO()
    shell = AnchorApp(engine, stream=out)
    expected: list[str] = []
    n = 0
    for i in range(200):
        for _ in range(1 + i % 3):               # 1, 2 or 3 new lines between refreshes
            if pattern == "distinct":
                line = f"line {n}"
            else:
                line = "anchor: What are you working on?" if n % 2 == 0 else "you: (silence)"
            engine.transcript.append(line)
            expected.append(line)
            n += 1
        shell._refresh()
    shell._refresh()
    assert out.getvalue().splitlines() == expected
    assert shell._tail.printed == len(expected)


def test_engine_transcript_is_shared_with_the_conversation_and_counted(settings, store):
    from anchor.engine import Transcript

    eng, _, _ = _engine(settings, store)
    assert isinstance(eng.transcript, Transcript)
    assert eng.conv.transcript is eng.transcript
    eng.startup()                                # "What are you working on?" / the goal / the confirmation ...
    lines, total = eng.transcript.snapshot()
    assert total == len(lines) >= 2
    eng.set_muted(True)
    assert eng.transcript.snapshot()[1] == total + 1


def test_tail_drains_a_transcript_being_appended_from_another_thread():
    from anchor.app import TranscriptTail
    from anchor.engine import Transcript

    transcript = Transcript(maxlen=30)
    done = threading.Event()

    def appender():
        for i in range(500):
            transcript.append(f"line {i}")
            time.sleep(0.001)
        done.set()

    threading.Thread(target=appender, daemon=True).start()
    tail = TranscriptTail()
    drained: list[str] = []
    while not done.is_set():
        drained += tail.drain(transcript)
    drained += tail.drain(transcript)
    assert drained == [f"line {i}" for i in range(500)]


# --------------------------------------------------------------------------- #
# 7. The doctor never prints the key (real subprocess)
# --------------------------------------------------------------------------- #


def _key_tail_from_dotenv() -> str:
    """Last six characters of the key in .env, read here and never printed. "" when absent."""
    try:
        from dotenv import dotenv_values

        key = (dotenv_values(PROJECT_ROOT / ".env").get("OPENAI_API_KEY") or "").strip()
    except Exception:
        key = ""
    return key[-6:] if len(key) >= 12 else ""


@pytest.mark.skipif(not VENV_PYTHON.exists() or not _key_tail_from_dotenv(), reason="needs .venv and a key in .env")
def test_doctor_subprocess_never_prints_the_key():
    tail = _key_tail_from_dotenv()
    proc = subprocess.run(
        [str(VENV_PYTHON), "-m", "anchor", "doctor"], cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120,
    )
    output = proc.stdout + proc.stderr
    assert "OpenAI API key" in proc.stdout
    assert proc.stdout.count("\n") >= 7
    leaked = tail in output                      # compared here, never echoed into an assertion message
    assert not leaked, "the doctor's output contains the API key"
