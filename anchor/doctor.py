"""``python -m anchor doctor`` — one line per check, ✓/✗, and a hint that points at setup.md.

Required checks (a failure returns 1): OpenAI key, microphone, /usr/bin/say, writable DB
folder, Python ≥ 3.11. Optional checks (reported, never fatal): Accessibility and Screen
Recording — Anchor degrades without them and says so.

The API key is never printed. Every line goes through ``_redact`` as a second guard.
"""

from __future__ import annotations

import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, TextIO

from .config import Settings, api_key_present, load_env

SAY_PATH = "/usr/bin/say"
MIN_PYTHON = (3, 11)
OPENAI_TIMEOUT_S = 15


@dataclass
class Check:
    name: str
    ok: bool
    required: bool
    hint: str = ""      # what to do when it fails (mentions the setup.md number)
    detail: str = ""    # short note when it passes


def _redact(text: str) -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key and key in text:
        text = text.replace(key, "***")
    return text


# ---- individual checks ---------------------------------------------------- #


def _list_model_ids(timeout: float = OPENAI_TIMEOUT_S) -> set[str]:
    """Network call, isolated so tests can replace it."""
    from openai import OpenAI

    client = OpenAI(timeout=timeout)
    return {m.id for m in client.models.list()}


def check_api_key(settings: Settings) -> Check:
    load_env()
    name = "OpenAI API key"
    if not api_key_present():
        return Check(name, False, True, hint="put OPENAI_API_KEY=... in .env (setup.md #1)")
    try:
        ids = _list_model_ids()
    except Exception as exc:
        return Check(
            name, False, True,
            hint=f"key rejected or no network ({type(exc).__name__}); check the key in .env (setup.md #1)",
        )
    wanted = [settings.judge_model, settings.judge_fallback_model, settings.tts_model, settings.stt_model]
    missing = [m for m in wanted if m not in ids]
    if missing:
        return Check(
            name, False, True,
            hint="key works but these models are not visible to it: "
                 + ", ".join(missing) + " — check model names in anchor/config.py (setup.md #1)",
        )
    return Check(name, True, True, detail=f"valid; {len(wanted)}/{len(wanted)} models visible")


def check_accessibility(_settings: Settings) -> Check:
    name = "Accessibility"
    try:
        import ApplicationServices

        ok = bool(ApplicationServices.AXIsProcessTrusted())
    except Exception:
        ok = False
    return Check(
        name, ok, False,
        hint="System Settings → Privacy & Security → Accessibility → add your terminal (setup.md #2); "
             "until then Anchor sees app names only, not window titles",
        detail="window titles readable",
    )


def check_screen_recording(_settings: Settings) -> Check:
    name = "Screen Recording"
    try:
        import Quartz

        ok = bool(Quartz.CGPreflightScreenCaptureAccess())
    except Exception:
        ok = False
    return Check(
        name, ok, False,
        hint="System Settings → Privacy & Security → Screen & System Audio Recording → add your terminal "
             "(setup.md #3); until then no screen-change detection or escalation screenshots",
        detail="screen hash and escalation screenshots available",
    )


def check_microphone(_settings: Settings) -> Check:
    name = "Microphone"
    try:
        import sounddevice

        devices = sounddevice.query_devices()
        inputs = [d for d in devices if int(d.get("max_input_channels", 0)) > 0]
    except Exception:
        inputs = []
    return Check(
        name, bool(inputs), True,
        hint="no audio input device found — connect or enable a microphone and grant Microphone "
             "permission to your terminal (setup.md #4)",
        detail=f"{len(inputs)} input device(s)",
    )


def check_say(_settings: Settings) -> Check:
    ok = os.access(SAY_PATH, os.X_OK) or shutil.which("say") is not None
    return Check(
        "macOS voice (say)", ok, True,
        hint=f"{SAY_PATH} is missing; it is the offline fallback voice (setup.md #7)",
        detail=SAY_PATH,
    )


def check_db_path(settings: Settings) -> Check:
    parent = Path(settings.db_path).expanduser().parent
    probe = parent
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    ok = probe.is_dir() and os.access(probe, os.W_OK)
    return Check(
        "Database folder", ok, True,
        hint=f"cannot write to {parent}; set ANCHOR_DB=/some/writable/path/anchor.db (setup.md #8)",
        detail=str(parent),
    )


def check_python(_settings: Settings) -> Check:
    ok = sys.version_info >= MIN_PYTHON
    version = ".".join(str(v) for v in sys.version_info[:3])
    return Check(
        "Python ≥ 3.11", ok, True,
        hint=f"running {version}; run ./run.sh to create a Python 3.12 virtualenv (setup.md #9)",
        detail=version,
    )


# ---- runner ---------------------------------------------------------------- #


def run_checks(settings: Settings) -> list[Check]:
    """Looked up by name at call time so tests can monkeypatch individual checks."""
    return [
        check_api_key(settings),
        check_accessibility(settings),
        check_screen_recording(settings),
        check_microphone(settings),
        check_say(settings),
        check_db_path(settings),
        check_python(settings),
    ]


def format_check(check: Check) -> str:
    mark = "✓" if check.ok else "✗"
    tag = "" if (check.ok or check.required) else " (optional)"
    text = check.detail if check.ok else check.hint
    return _redact(f"{mark} {check.name}{tag}: {text}")


def run_doctor(settings: Optional[Settings] = None, out: Optional[TextIO] = None) -> int:
    settings = settings or Settings.from_env()
    out = out or sys.stdout
    checks = run_checks(settings)
    for check in checks:
        print(format_check(check), file=out)
    failed = [c for c in checks if c.required and not c.ok]
    degraded = [c for c in checks if not c.required and not c.ok]
    if failed:
        print(_redact(f"{len(failed)} required check(s) failed — see setup.md"), file=out)
        return 1
    if degraded:
        print(f"All required checks passed; {len(degraded)} optional permission(s) missing (Anchor degrades).", file=out)
    else:
        print("All checks passed.", file=out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(run_doctor())
