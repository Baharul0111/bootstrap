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

ROAST_DELIVERY = (  # the persona pack's VOICE DELIVERY paragraph, verbatim; performance only, never joke content
    "Speak in clear, natural English with a light Indian conversational cadence. Sound like a dry, "
    "quick-witted teammate who has just noticed something embarrassing. Relaxed confidence, understated "
    "amusement, crisp consonants. Keep it understandable on one hearing. A tiny pause before the final "
    "reveal is enough; do not insert a pause into every clause. Land the punchline cleanly and stop. No "
    "shouting, giggling, canned laughter, cartoon goblin growl, exaggerated accent, or motivational tone. "
    "Don't sound wounded or angry. Read the supplied words exactly; never add a greeting, explanation, or "
    "extra joke."
)

TONE_BRIEFS = {  # instructions for TTS by tone; the three delivery tags refine the roast base
    "roast": ROAST_DELIVERY,
    "deadpan": ROAST_DELIVERY + " Delivery: deadpan, almost matter-of-fact.",
    "mock_respect": ROAST_DELIVERY + " Delivery: mock respect, briefly polite before the sting.",
    "disbelief": ROAST_DELIVERY + " Delivery: disbelief, slight incredulity without raising volume.",
    "flat": "Flat, calm, matter-of-fact. No humour, no warmth, no edge.",
    "warm": "Warm, plain, friendly. Brief.",
    "neutral": "Clear, friendly and brief.",
}
HINDI_BRIEF = " Speak natural, everyday Hindi."
ACCENT_BRIEF = (
    " Accent: authentic Indian English, exactly how an educated woman from Delhi or Mumbai speaks English every day."
    " Indian English intonation and rhythm (syllable-timed, gentle rise-and-fall melody), Indian vowel colouring,"
    " retroflex t and d, clear rolled r, softened w/v. Absolutely not American and not British. Natural, not a caricature."
)
_ROAST_TAGS = ("roast", "deadpan", "mock_respect", "disbelief")


def tts_brief(tone: str, language: str) -> str:
    """The performance instruction for one utterance. Hindi roasts keep the persona's manner, in Hindi."""
    brief = TONE_BRIEFS.get(tone, TONE_BRIEFS["neutral"])
    if language != "hi":
        return brief + ACCENT_BRIEF
    if tone in _ROAST_TAGS:
        return brief.replace(
            "Speak in clear, natural English with a light Indian conversational cadence.",
            "Speak natural, everyday Hindi (Devanagari text), with a light conversational cadence.",
        )
    return brief + HINDI_BRIEF

TTS_RATE = 24_000            # gpt-4o-mini-tts "pcm" output: 24 kHz mono int16
STT_RATE = 16_000            # capture rate for webrtcvad and transcription
FRAME_SAMPLES = 480          # 30 ms at 16 kHz, the largest frame webrtcvad accepts
TRAIL_SILENCE_FRAMES = 14    # 0.42 s of silence ends an utterance (the reply must feel immediate)
PREROLL_FRAMES = 10          # 300 ms kept from just before speech onset
ONSET_FRAMES = 3             # 90 ms of confirmed speech starts an utterance
MAX_UTTERANCE_FRAMES = 300   # 9 s: the longest reply we will ever wait to transcribe


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
        self._cache: dict[tuple[str, str, str], bytes] = {}   # pre-synthesized PCM for lines we know are coming
        self._cache_lock = threading.Lock()

    def prefetch(self, text: str, *, language: str = "en", tone: str = "neutral") -> bool:
        """Synthesize ``text`` into memory ahead of time so a later ``speak`` starts instantly. Never raises."""
        key = (text.strip(), language, tone)
        if not key[0]:
            return False
        with self._cache_lock:
            if key in self._cache:
                return True
        try:
            pcm = self._synthesize(text, language, tone)
        except Exception:
            return False
        with self._cache_lock:
            if len(self._cache) >= 32:
                self._cache.pop(next(iter(self._cache)))
            self._cache[key] = pcm
        return True

    def _synthesize(self, text: str, language: str, tone: str) -> bytes:
        brief = tts_brief(tone, language)
        with self.client.audio.speech.with_streaming_response.create(
            model=self.settings.tts_model,
            voice=self.settings.voice,
            input=text,
            instructions=brief,
            response_format="pcm",
        ) as resp:
            return b"".join(resp.iter_bytes(4096))

    def _play(self, pcm: bytes) -> None:
        import sounddevice as sd

        usable = len(pcm) - len(pcm) % 2
        with sd.RawOutputStream(samplerate=TTS_RATE, channels=1, dtype="int16") as out:
            for i in range(0, usable, 4096):
                out.write(pcm[i:i + 4096])

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
                with self._cache_lock:
                    cached = self._cache.pop((text.strip(), language, tone), None)
                if cached:
                    self._play(cached)
                else:
                    self._stream(text, language, tone)
            except Exception:
                try:
                    self.fallback.speak(text, language=language, tone=tone)
                except Exception:
                    pass

    def _stream(self, text: str, language: str, tone: str) -> None:
        import sounddevice as sd

        brief = tts_brief(tone, language)
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

        vad = webrtcvad.Vad(3)                  # most aggressive: room noise and the speaker's tail are not speech
        deadline = time.monotonic() + window_s  # hard cap on total capture, whatever the VAD says
        preroll: deque = deque(maxlen=PREROLL_FRAMES)
        frames: list = []
        speaking, silent, onset = False, 0, 0
        noise_floor = 0.0
        levels: list = []

        def loudness(raw: bytes) -> float:
            try:
                import numpy as np

                arr = np.frombuffer(raw, dtype=np.int16)
                return float(np.abs(arr.astype(np.int32)).mean()) if arr.size else 0.0
            except Exception:
                return 1e9                      # no numpy: fall back to the VAD alone

        with sd.InputStream(samplerate=STT_RATE, channels=1, dtype="int16", blocksize=FRAME_SAMPLES) as mic:
            while time.monotonic() < deadline:
                data, _overflowed = mic.read(FRAME_SAMPLES)
                frame = bytes(data)
                level = loudness(frame)
                if not speaking and len(levels) < 8:
                    levels.append(level)        # the first ~240 ms are the room: they set the noise floor
                    noise_floor = sorted(levels)[len(levels) // 2]
                # speech = the VAD agrees AND the frame is clearly louder than the room
                is_speech = vad.is_speech(frame, STT_RATE) and (
                    noise_floor <= 0.0 or level > max(2.5 * noise_floor, 250.0)   # a real mic always has a floor
                )
                if not speaking:
                    preroll.append(frame)
                    onset = onset + 1 if is_speech else 0
                    if onset >= ONSET_FRAMES:   # a real word, not a click
                        speaking = True
                        frames.extend(preroll)
                    continue
                frames.append(frame)
                silent = 0 if is_speech else silent + 1
                if silent >= TRAIL_SILENCE_FRAMES:
                    break
                if len(frames) >= MAX_UTTERANCE_FRAMES:
                    break                       # a reply is a sentence, not a speech
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


class FileListener:
    """Text stand-in for the microphone: replies are lines appended to a file (``ANCHOR_REPLY_FILE``).

    Used for scripted end-to-end runs on a real desktop where nobody is speaking. ``listen`` waits up to
    ``window_s`` for a non-empty line, consumes it, and returns it; silence returns "" like the real one.
    """

    def __init__(self, path: str) -> None:
        self.path = path
        self.calls: list[tuple[float, str]] = []

    def listen(self, *, window_s: float = 8.0, language: str = "en") -> str:
        import os
        import time

        self.calls.append((window_s, language))
        deadline = time.monotonic() + max(0.5, window_s)
        while time.monotonic() < deadline:
            try:
                if os.path.exists(self.path):
                    with open(self.path, "r", encoding="utf-8") as fh:
                        lines = fh.read().splitlines()
                    if lines:
                        first, rest = lines[0], lines[1:]
                        with open(self.path, "w", encoding="utf-8") as fh:
                            fh.write("\n".join(rest) + ("\n" if rest else ""))
                        if first.strip():
                            return first.strip()
            except Exception:
                pass
            time.sleep(0.25)
        return ""


def make_voice(settings: Settings) -> tuple[Speaker, Listener]:
    """The OpenAI pair when a key is configured; otherwise ``say`` plus a listener that hears nothing.

    ``ANCHOR_REPLY_FILE=path`` swaps the microphone for a text file (scripted desktop runs)."""
    import os

    reply_file = os.environ.get("ANCHOR_REPLY_FILE", "").strip()
    speaker: Speaker = OpenAISpeaker(settings) if api_key_present() else SayFallbackSpeaker()
    if reply_file:
        return speaker, FileListener(reply_file)
    if api_key_present():
        batch = OpenAIListener(settings)
        if os.environ.get("ANCHOR_STREAMING_STT", "1").strip().lower() not in {"0", "false", "no", "off"}:
            try:
                from .streaming_stt import RealtimeListener

                return speaker, RealtimeListener(settings, fallback=batch)
            except Exception:
                pass
        return speaker, batch
    return speaker, FakeListener()
