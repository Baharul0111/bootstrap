"""Streaming speech-to-text: audio goes to OpenAI's realtime transcription socket WHILE the person speaks.

The text is usually complete within a few hundred milliseconds of the last word, instead of after a
whole round trip once the microphone closes. The batch listener stays as the safety net: the same
audio is kept locally, and if the stream fails or is slow, the recording is transcribed the old way.
Never raises; silence returns "".
"""

from __future__ import annotations

import base64
import io
import json
import os
import queue
import threading
import time
import wave
from collections import deque
from typing import Optional

from anchor.config import Settings

RT_URL = "wss://api.openai.com/v1/realtime?intent=transcription"
CAPTURE_RATE = 48_000        # webrtcvad accepts 48 kHz; every other sample gives the 24 kHz the socket wants
SEND_RATE = 24_000
FRAME_MS = 30
FRAME_SAMPLES = CAPTURE_RATE * FRAME_MS // 1000   # 1440
TRAIL_SILENCE_FRAMES = 14    # 0.42 s
ONSET_FRAMES = 3
PREROLL_FRAMES = 10
MAX_UTTERANCE_FRAMES = 300   # 9 s
STREAM_GRACE_S = 1.2         # how long to wait for the streamed transcript after the local end-of-speech


class RealtimeListener:
    """Implements ``models.Listener`` with a streaming socket plus a local recording fallback."""

    def __init__(self, settings: Settings, client=None, fallback=None) -> None:
        self.settings = settings
        self._client = client
        self.fallback = fallback          # an OpenAIListener-like object with ``client`` for batch transcription
        self.last_mode = ""               # "stream" | "batch" | "silence" — for diagnostics

    # ------------------------------------------------------------------ public
    def listen(self, *, window_s: float = 8.0, language: str = "en") -> str:
        try:
            return self._listen(window_s, language)
        except Exception:
            if self.fallback is not None:
                try:
                    self.last_mode = "batch"
                    return self.fallback.listen(window_s=window_s, language=language)
                except Exception:
                    return ""
            return ""

    # ------------------------------------------------------------------ internals
    def _listen(self, window_s: float, language: str) -> str:
        import numpy as np
        import sounddevice as sd
        import webrtcvad

        key = os.environ.get("OPENAI_API_KEY", "")
        if not key:
            raise RuntimeError("no key")
        lang = language if language in {"en", "hi"} else None
        sock = _Socket(key, self.settings.stt_model, lang)
        sock.start()

        vad = webrtcvad.Vad(3)
        deadline = time.monotonic() + window_s
        preroll: deque = deque(maxlen=PREROLL_FRAMES)
        frames: list[bytes] = []          # 24 kHz mono int16, for the batch fallback
        speaking, silent, onset = False, 0, 0
        noise_floor, levels = 0.0, []
        spoke_at = None

        server_done = False
        with sd.InputStream(samplerate=CAPTURE_RATE, channels=1, dtype="int16", blocksize=FRAME_SAMPLES) as mic:
            while time.monotonic() < deadline:
                data, _ = mic.read(FRAME_SAMPLES)
                arr = np.frombuffer(bytes(data), dtype=np.int16)
                level = float(np.abs(arr.astype(np.int32)).mean()) if arr.size else 0.0
                frame24 = arr[::2].tobytes()
                sock.send_audio(frame24)                         # stream everything; the server's VAD is the authority
                # The server finished a turn and no new one has opened: the reply is ready. Return NOW.
                if sock.transcripts and not sock.turn_open and sock.completed_at is not None \
                        and time.monotonic() - sock.completed_at >= 0.15:
                    server_done = True
                    break
                # Local voice detection is only the safety net (a bad connection, or a server that hears nothing).
                is_speech = vad.is_speech(bytes(data), CAPTURE_RATE) and level > 300.0
                if not speaking:
                    preroll.append(frame24)
                    onset = onset + 1 if is_speech else 0
                    if onset >= ONSET_FRAMES:
                        speaking = True
                        frames.extend(preroll)
                    continue
                frames.append(frame24)
                silent = 0 if is_speech else silent + 1
                if silent >= TRAIL_SILENCE_FRAMES or len(frames) >= MAX_UTTERANCE_FRAMES:
                    spoke_at = time.monotonic()
                    break

        if server_done:
            text = sock.wait_transcript(0.0)
            sock.close()
            self.last_mode = "stream"
            return (text or "").strip()

        if not speaking and not sock.transcripts and not sock.turn_open:
            sock.close()
            self.last_mode = "silence"
            return ""

        # Local end of speech first: give the server a moment to finish its turn, then fall back to batch.
        text = sock.wait_transcript(STREAM_GRACE_S)
        sock.close()
        if text:
            self.last_mode = "stream"
            return text.strip()

        # Fallback: transcribe what we recorded (24 kHz WAV) with the batch endpoint.
        self.last_mode = "batch"
        if not frames:
            return ""
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SEND_RATE)
            w.writeframes(b"".join(frames))
        client = self._client
        if client is None:
            from openai import OpenAI

            client = self._client = OpenAI(timeout=20, max_retries=1)
        tr = client.audio.transcriptions.create(
            model=self.settings.stt_model, file=("reply.wav", buf.getvalue()), language=lang
        )
        return (tr.text or "").strip()


class _Socket:
    """One realtime transcription session: send 24 kHz PCM16 chunks, collect the completed transcript."""

    def __init__(self, key: str, model: str, language: Optional[str]) -> None:
        self.key, self.model, self.language = key, model, language
        self.transcripts: list[str] = []      # one entry per server-detected turn, in order
        self._partial: list[str] = []
        self._turn_open = False               # server said speech started and has not completed that turn yet
        self.completed_at: Optional[float] = None
        self._q: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._done = threading.Event()
        self._ready = threading.Event()
        self._ws = None
        self._threads: list[threading.Thread] = []

    @property
    def turn_open(self) -> bool:
        return self._turn_open

    def start(self) -> None:
        t = threading.Thread(target=self._run, name="anchor-stt-stream", daemon=True)
        t.start()
        self._threads.append(t)

    def send_audio(self, pcm24: bytes) -> None:
        self._q.put(pcm24)

    def wait_transcript(self, timeout: float) -> Optional[str]:
        """After the local end of speech: give the server a moment to finish its last turn, then join all turns."""
        self._q.put(None)                      # no more audio
        if not self.transcripts and not self._turn_open:
            return None                        # the server never heard a turn: go straight to the batch path
        if timeout <= 0:
            return " ".join(t for t in self.transcripts if t).strip() or None
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            if self.transcripts and not self._turn_open:
                break
            if self._done.is_set():
                break
            time.sleep(0.05)
        text = " ".join(t for t in self.transcripts if t).strip()
        if not text and self._partial:
            text = "".join(self._partial).strip()
        return text or None

    def close(self) -> None:
        """Never blocks the reply: the socket is closed on a background thread."""
        self._done.set()
        ws = self._ws
        if ws is None:
            return

        def _close() -> None:
            try:
                ws.close()
            except Exception:
                pass

        threading.Thread(target=_close, name="anchor-stt-close", daemon=True).start()

    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            from websockets.sync.client import connect
        except Exception:
            self._done.set()
            return
        try:
            with connect(
                RT_URL,
                additional_headers={"Authorization": f"Bearer {self.key}"},
                open_timeout=4,
                close_timeout=1,
            ) as ws:
                self._ws = ws
                transcription = {"model": self.model}
                if self.language:
                    transcription["language"] = self.language
                session = {                       # the GA realtime shape (the beta shape is disabled server-side)
                    "type": "transcription",
                    "audio": {
                        "input": {
                            "format": {"type": "audio/pcm", "rate": SEND_RATE},
                            "transcription": transcription,
                            "turn_detection": {"type": "server_vad", "silence_duration_ms": 200},
                            "noise_reduction": {"type": "near_field"},
                        }
                    },
                }
                ws.send(json.dumps({"type": "session.update", "session": session}))

                reader = threading.Thread(target=self._read, args=(ws,), daemon=True)
                reader.start()
                pending = bytearray()
                while not self._done.is_set():
                    try:
                        chunk = self._q.get(timeout=0.2)
                    except queue.Empty:
                        continue
                    if chunk is None:
                        if pending:
                            ws.send(json.dumps({"type": "input_audio_buffer.append",
                                                "audio": base64.b64encode(bytes(pending)).decode()}))
                        # a little silence lets the server's own turn detector close the turn promptly
                        ws.send(json.dumps({"type": "input_audio_buffer.append",
                                            "audio": base64.b64encode(b"\x00" * (SEND_RATE * 2 // 2)).decode()}))
                        break
                    pending += chunk
                    if len(pending) >= 4800:               # 100 ms at 24 kHz
                        ws.send(json.dumps({"type": "input_audio_buffer.append",
                                            "audio": base64.b64encode(bytes(pending)).decode()}))
                        pending.clear()
                reader.join(timeout=STREAM_GRACE_S + 0.5)
        except Exception:
            pass
        finally:
            self._done.set()

    def _read(self, ws) -> None:
        try:
            while not self._done.is_set():
                try:
                    raw = ws.recv(timeout=0.5)
                except TimeoutError:
                    continue                        # nothing arrived in this half second; keep listening
                if raw is None:
                    continue
                msg = json.loads(raw)
                kind = msg.get("type", "")
                if kind == "input_audio_buffer.speech_started":
                    self._turn_open = True
                elif kind == "conversation.item.input_audio_transcription.delta":
                    self._partial.append(msg.get("delta", ""))
                elif kind == "conversation.item.input_audio_transcription.completed":
                    self.transcripts.append((msg.get("transcript") or "").strip())
                    self._partial.clear()
                    self._turn_open = False
                    self.completed_at = time.monotonic()
                elif kind == "error":
                    self._done.set()
                    return
        except Exception:
            return
