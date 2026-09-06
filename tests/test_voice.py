"""Hermetic tests for anchor.voice: no network, no audio hardware, no ``say``."""

from __future__ import annotations

import subprocess
import sys
import time
import wave
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from anchor.config import Settings
from anchor.voice import (
    HINDI_BRIEF,
    ROAST_DELIVERY,
    TONE_BRIEFS,
    FakeListener,
    FakeSpeaker,
    OpenAIListener,
    OpenAISpeaker,
    SayFallbackSpeaker,
    make_voice,
)

WINDOW_S = 0.3  # short windows keep the timing tests quick


@pytest.fixture
def fake_audio(monkeypatch):
    """Replace sounddevice and webrtcvad with fakes; returns knobs (speech flag, PCM writes, VAD frame sizes)."""
    knobs = SimpleNamespace(speech=False, writes=[], frame_lengths=set())

    class FakeInputStream:  # silent microphone that runs at real-time pace
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self, frames):
            time.sleep(frames / 16000)
            return np.zeros((frames, 1), dtype=np.int16), False

    class FakeOutputStream:  # speaker that just collects what was written
        def __init__(self, *args, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def write(self, data):
            knobs.writes.append(bytes(data))

    class FakeVad:
        def __init__(self, mode):
            pass

        def is_speech(self, frame, rate):
            knobs.frame_lengths.add((len(frame), rate))
            return knobs.speech

    monkeypatch.setitem(
        sys.modules, "sounddevice", SimpleNamespace(InputStream=FakeInputStream, RawOutputStream=FakeOutputStream)
    )
    monkeypatch.setitem(sys.modules, "webrtcvad", SimpleNamespace(Vad=FakeVad))
    return knobs


class FakeTTSResponse:
    def __init__(self, chunks):
        self.chunks = chunks

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_bytes(self, size):
        yield from self.chunks


def tts_client(chunks):
    client = MagicMock()
    client.audio.speech.with_streaming_response.create.return_value = FakeTTSResponse(list(chunks))
    return client


# --- fakes -----------------------------------------------------------------


def test_fake_speaker_records_calls():
    spk = FakeSpeaker()
    spk.speak("hi")
    spk.speak("namaste", language="hi", tone="warm")
    assert spk.calls == [("hi", "en", "neutral"), ("namaste", "hi", "warm")]


def test_fake_listener_pops_replies_then_returns_empty():
    lst = FakeListener(["ten minutes", "done"])
    assert lst.listen() == "ten minutes"
    assert lst.listen(window_s=3.0, language="hi") == "done"
    assert lst.listen() == ""
    assert lst.calls == [(8.0, "en"), (3.0, "hi"), (8.0, "en")]
    assert FakeListener().listen() == ""


# --- speaker ---------------------------------------------------------------


def test_tone_briefs_has_the_seven_tones():
    assert set(TONE_BRIEFS) == {"roast", "deadpan", "mock_respect", "disbelief", "flat", "warm", "neutral"}
    assert all(isinstance(v, str) and v.strip() for v in TONE_BRIEFS.values())


def test_roast_brief_is_the_persona_delivery_paragraph():
    assert TONE_BRIEFS["roast"] == ROAST_DELIVERY
    assert ROAST_DELIVERY.startswith("Speak in clear, natural English with a light Indian conversational cadence.")
    assert ROAST_DELIVERY.endswith("never add a greeting, explanation, or extra joke.")
    assert "Land the punchline cleanly and stop." in ROAST_DELIVERY


@pytest.mark.parametrize(
    "tag, refinement",
    [
        ("deadpan", "almost matter-of-fact"),
        ("mock_respect", "briefly polite before the sting"),
        ("disbelief", "slight incredulity without raising volume"),
    ],
)
def test_delivery_tags_extend_the_persona_paragraph_with_their_own_refinement(tag, refinement):
    brief = TONE_BRIEFS[tag]
    assert brief.startswith(ROAST_DELIVERY)
    assert refinement in brief[len(ROAST_DELIVERY):]
    others = {"almost matter-of-fact", "briefly polite before the sting", "slight incredulity without raising volume"}
    assert all(other not in brief for other in others - {refinement})


def test_flat_brief_is_unchanged_and_joke_free():
    assert TONE_BRIEFS["flat"] == "Flat, calm, matter-of-fact. No humour, no warmth, no edge."
    assert "No humour" in TONE_BRIEFS["flat"]
    assert ROAST_DELIVERY not in TONE_BRIEFS["flat"]
    assert TONE_BRIEFS["warm"] == "Warm, plain, friendly. Brief."
    assert TONE_BRIEFS["neutral"] == "Clear, friendly and brief."


def test_openai_speaker_falls_back_when_tts_raises(fake_audio):
    client = MagicMock()
    client.audio.speech.with_streaming_response.create.side_effect = RuntimeError("network down")
    fallback = FakeSpeaker()
    OpenAISpeaker(Settings(), client=client, fallback=fallback).speak("Back to the essay?", language="hi", tone="roast")
    assert fallback.calls == [("Back to the essay?", "hi", "roast")]
    assert fake_audio.writes == []


def test_openai_speaker_falls_back_when_audio_device_fails(fake_audio):
    def broken_stream(*args, **kwargs):
        raise OSError("no output device")

    sys.modules["sounddevice"].RawOutputStream = broken_stream
    fallback = FakeSpeaker()
    OpenAISpeaker(Settings(), client=tts_client([b"\x01\x00" * 8]), fallback=fallback).speak("hello")
    assert fallback.calls == [("hello", "en", "neutral")]


@pytest.mark.parametrize(
    "tone, language, expected_brief",
    [
        ("roast", "hi", TONE_BRIEFS["roast"].replace(
            "Speak in clear, natural English with a light Indian conversational cadence.",
            "Speak natural, everyday Hindi (Devanagari text), with a light conversational cadence.")),
        ("warm", "hi", TONE_BRIEFS["warm"] + HINDI_BRIEF),
        ("flat", "en", TONE_BRIEFS["flat"]),
        ("mock_respect", "en", TONE_BRIEFS["mock_respect"]),
        ("disbelief", "en", TONE_BRIEFS["disbelief"]),
    ],
)
def test_openai_speaker_streams_pcm_with_tone_and_language_brief(fake_audio, tone, language, expected_brief):
    chunks = [b"\x01\x00" * 100, b"\x00" * 5, b"\x00" * 3]  # includes an odd-length chunk
    client = tts_client(chunks)
    fallback = FakeSpeaker()
    settings = Settings()
    OpenAISpeaker(settings, client=client, fallback=fallback).speak("Still here?", language=language, tone=tone)

    kwargs = client.audio.speech.with_streaming_response.create.call_args.kwargs
    assert kwargs["instructions"] == expected_brief
    assert kwargs["model"] == settings.tts_model
    assert kwargs["voice"] == settings.voice
    assert kwargs["input"] == "Still here?"
    assert kwargs["response_format"] == "pcm"
    assert b"".join(fake_audio.writes) == b"".join(chunks)
    assert all(len(w) % 2 == 0 for w in fake_audio.writes)  # never writes half a sample
    assert fallback.calls == []


def test_openai_speaker_skips_blank_text(fake_audio):
    client = MagicMock()
    OpenAISpeaker(Settings(), client=client, fallback=FakeSpeaker()).speak("   ")
    client.audio.speech.with_streaming_response.create.assert_not_called()


def test_say_fallback_picks_lekha_for_hindi_and_never_raises(monkeypatch):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs.get("input")))
        if cmd[:3] == ["say", "-v", "?"]:
            return SimpleNamespace(stdout="Albert  en_US  # Hello\nLekha   hi_IN  # नमस्ते\n", returncode=0)
        return SimpleNamespace(stdout="", returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_run)
    spk = SayFallbackSpeaker()
    spk.speak("-dash first", language="hi")
    spk.speak("plain", language="en", tone="roast")
    assert (["say", "-v", "Lekha"], "-dash first") in calls
    assert (["say"], "plain") in calls

    monkeypatch.setattr(subprocess, "run", MagicMock(side_effect=OSError("no say here")))
    SayFallbackSpeaker().speak("still fine", language="hi")  # must not raise


# --- listener --------------------------------------------------------------


def test_listener_returns_empty_on_silence_without_calling_api(fake_audio):
    fake_audio.speech = False
    client = MagicMock()
    started = time.monotonic()
    result = OpenAIListener(Settings(), client=client).listen(window_s=WINDOW_S)
    elapsed = time.monotonic() - started
    assert result == ""
    assert elapsed < WINDOW_S + 0.5
    client.audio.transcriptions.create.assert_not_called()
    assert fake_audio.frame_lengths == {(960, 16000)}  # 30 ms int16 frames at 16 kHz


@pytest.mark.parametrize("language, sent", [("en", "en"), ("hi", "hi"), ("fr", None)])
def test_listener_hard_cap_when_speech_never_ends(fake_audio, language, sent):
    fake_audio.speech = True
    client = MagicMock()
    client.audio.transcriptions.create.return_value = SimpleNamespace(text="  fifteen minutes  ")
    settings = Settings()
    started = time.monotonic()
    result = OpenAIListener(settings, client=client).listen(window_s=WINDOW_S, language=language)
    elapsed = time.monotonic() - started

    assert result == "fifteen minutes"
    assert WINDOW_S - 0.05 <= elapsed < WINDOW_S + 1.0
    client.audio.transcriptions.create.assert_called_once()
    kwargs = client.audio.transcriptions.create.call_args.kwargs
    assert kwargs["model"] == settings.stt_model
    assert kwargs["language"] == sent
    name, wav_bytes = kwargs["file"]
    assert name == "reply.wav"
    assert wav_bytes[:4] == b"RIFF" and wav_bytes[8:12] == b"WAVE"
    with wave.open(BytesIO(wav_bytes)) as w:
        assert (w.getnchannels(), w.getsampwidth(), w.getframerate()) == (1, 2, 16000)
        assert 480 <= w.getnframes() <= (WINDOW_S / 0.03 + 1) * 480  # capped at the window


def test_listener_returns_empty_when_transcription_fails(fake_audio):
    fake_audio.speech = True
    client = MagicMock()
    client.audio.transcriptions.create.side_effect = RuntimeError("timeout")
    assert OpenAIListener(Settings(), client=client).listen(window_s=0.1) == ""


# --- factory ---------------------------------------------------------------


def test_make_voice_without_key_uses_say_and_a_deaf_listener(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    speaker, listener = make_voice(Settings())
    assert isinstance(speaker, SayFallbackSpeaker)
    assert listener.listen() == ""


def test_make_voice_with_key_uses_openai_pair(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-not-real")
    speaker, listener = make_voice(Settings())
    assert isinstance(speaker, OpenAISpeaker)
    assert isinstance(listener, OpenAIListener)
