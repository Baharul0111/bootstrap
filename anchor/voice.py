"""Voice I/O: OpenAI text-to-speech and speech-to-text, with a macOS ``say`` fallback.

Everything here implements the ``Speaker`` / ``Listener`` protocols from
``anchor.models`` and never raises: a failed speak falls back to ``say``, a
failed listen returns "". Audio libraries are imported lazily so the module
loads on machines without a sound device.
"""

from __future__ import annotations

import io
import subprocess
import threading
import time
import wave
from collections import deque
from typing import Optional

from anchor.config import Settings, api_key_present
from anchor.models import Listener, Speaker

TONE_BRIEFS = {  # instructions for TTS by tone
    "roast": "Light, dry, amused, never mocking. Conversational, brief.",
    "flat": "Flat, calm, matter-of-fact. No humour, no warmth, no edge.",
    "warm": "Warm, plain, friendly. Brief.",
    "neutral": "Clear, friendly and brief.",
}
HINDI_BRIEF = " Speak natural, everyday Hindi."

TTS_RATE = 24_000            # gpt-4o-mini-tts "pcm" output: 24 kHz mono int16
STT_RATE = 16_000            # capture rate for webrtcvad and transcription
FRAME_SAMPLES = 480          # 30 ms at 16 kHz, the largest frame webrtcvad accepts
TRAIL_SILENCE_FRAMES = 30    # 0.9 s of silence ends an utterance
PREROLL_FRAMES = 10          # 300 ms kept from just before speech onset


class SayFallbackSpeaker:
    """macOS ``say``. Uses the Lekha voice for Hindi when it is installed. Never raises."""

    def __init__(self) -> None:
        self._lekha: Optional[bool] = None

    def _has_lekha(self) -> bool:
        if self._lekha is None:
            try:
                out = subprocess.run(["say", "-v", "?"], capture_output=True, text=True, timeout=10).stdout
                self._lekha = any(line.split()[:1] == ["Lekha"] for line in out.splitlines())
            except Exception:
                self._lekha = False
        return self._lekha

    def speak(self, text: str, *, language: str = "en", tone: str = "neutral") -> None:
        """``tone`` is accepted for protocol compatibility; ``say`` has no tone control."""
        cmd = ["say"]
        if language == "hi" and self._has_lekha():
            cmd += ["-v", "Lekha"]
        try:  # text goes in on stdin so a reply starting with "-" is never read as an option
            subprocess.run(cmd, input=text, text=True, capture_output=True, timeout=30)
        except Exception:
            pass


class OpenAISpeaker:
    """Streams gpt-4o-mini-tts PCM straight to the speakers; falls back to ``say`` on any failure."""

    def __init__(self, settings: Settings, client=None, fallback: Optional[Speaker] = None) -> None:
        self.settings = settings
        self._client = client
        self.fallback: Speaker = fallback if fallback is not None else SayFallbackSpeaker()
        self._lock = threading.Lock()  # one utterance at a time, whichever thread asks

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def speak(self, text: str, *, language: str = "en", tone: str = "neutral") -> None:
        """Blocks until playback has finished. Never raises."""
        if not text.strip():
            return
        with self._lock:
            try:
                self._stream(text, language, tone)
            except Exception:
                try:
                    self.fallback.speak(text, language=language, tone=tone)
                except Exception:
                    pass

    def _stream(self, text: str, language: str, tone: str) -> None:
        import sounddevice as sd

        brief = TONE_BRIEFS.get(tone, TONE_BRIEFS["neutral"]) + (HINDI_BRIEF if language == "hi" else "")
        with self.client.audio.speech.with_streaming_response.create(
            model=self.settings.tts_model,
            voice=self.settings.voice,
            input=text,
            instructions=brief,
            response_format="pcm",
        ) as resp, sd.RawOutputStream(samplerate=TTS_RATE, channels=1, dtype="int16") as out:
            pending = bytearray()
            for chunk in resp.iter_bytes(4096):
                pending += chunk
                usable = len(pending) - len(pending) % 2  # whole int16 samples only
                if usable:
                    out.write(bytes(pending[:usable]))
                    del pending[:usable]
        # leaving the output stream's context stops it, which waits for playback to drain


class OpenAIListener:
    """Records one reply from the default microphone and transcribes it. Never raises."""

    def __init__(self, settings: Settings, client=None) -> None:
        self.settings = settings
        self._client = client

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI()
        return self._client

    def listen(self, *, window_s: float = 8.0, language: str = "en") -> str:
        """"" when nobody spoke within ``window_s`` or anything failed; the API is not called for silence."""
        try:
            wav = self._capture(window_s)
            if wav is None:
                return ""
            tr = self.client.audio.transcriptions.create(
                model=self.settings.stt_model,
                file=("reply.wav", wav),
                language=language if language in {"en", "hi"} else None,
            )
            return (tr.text or "").strip()
        except Exception:
            return ""

    def _capture(self, window_s: float) -> Optional[bytes]:
        """16 kHz mono WAV of one utterance, or None if no speech started before the deadline."""
        import sounddevice as sd
        import webrtcvad

        vad = webrtcvad.Vad(2)
        deadline = time.monotonic() + window_s  # hard cap on total capture, whatever the VAD says
        preroll: deque = deque(maxlen=PREROLL_FRAMES)
        frames: list = []
        speaking, silent = False, 0
        with sd.InputStream(samplerate=STT_RATE, channels=1, dtype="int16", blocksize=FRAME_SAMPLES) as mic:
            while time.monotonic() < deadline:
                data, _overflowed = mic.read(FRAME_SAMPLES)
                frame = bytes(data)
                is_speech = vad.is_speech(frame, STT_RATE)
                if not speaking:
                    if is_speech:
                        speaking = True
                        frames.extend(preroll)
                        frames.append(frame)
                    else:
                        preroll.append(frame)
                    continue
                frames.append(frame)
                silent = 0 if is_speech else silent + 1
                if silent >= TRAIL_SILENCE_FRAMES:
                    break
        if not speaking:
            return None
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(STT_RATE)
            w.writeframes(b"".join(frames))
        return buf.getvalue()


class FakeSpeaker:
    """Test double: records every call as (text, language, tone)."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, str, str]] = []

    def speak(self, text: str, *, language: str = "en", tone: str = "neutral") -> None:
        self.calls.append((text, language, tone))


class FakeListener:
    """Test double: hands out scripted replies in order, then "" once they run out."""

    def __init__(self, replies: Optional[list[str]] = None) -> None:
        self.replies: list[str] = list(replies or [])
        self.calls: list[tuple[float, str]] = []

    def listen(self, *, window_s: float = 8.0, language: str = "en") -> str:
        self.calls.append((window_s, language))
        return self.replies.pop(0) if self.replies else ""


def make_voice(settings: Settings) -> tuple[Speaker, Listener]:
    """The OpenAI pair when a key is configured; otherwise ``say`` plus a listener that hears nothing."""
    if api_key_present():
        return OpenAISpeaker(settings), OpenAIListener(settings)
    return SayFallbackSpeaker(), FakeListener()
