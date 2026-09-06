"""Configuration: fixed product choices plus a few environment switches.

The OpenAI key is read from ``.env`` (or the environment) and is never logged,
printed, or stored in the database. Only ``api_key_present()`` is exposed.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Booty Globlin intensities ↔ the register ladder (dry | playful | spicy). Default "playful" == persona "pointed".
INTENSITY_TO_REGISTER = {"playful": "dry", "pointed": "playful", "savage": "spicy"}


def load_env() -> None:
    """Load .env from ANCHOR_ENV, the project root, or the current directory (first found)."""
    candidates = []
    if os.environ.get("ANCHOR_ENV"):
        candidates.append(Path(os.environ["ANCHOR_ENV"]))
    candidates += [PROJECT_ROOT / ".env", Path.cwd() / ".env"]
    for path in candidates:
        if path.is_file():
            load_dotenv(dotenv_path=path, override=False)
            return


def api_key_present() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def _truthy(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def default_db_path() -> Path:
    override = os.environ.get("ANCHOR_DB")
    if override:
        return Path(override).expanduser()
    return Path.home() / "Library" / "Application Support" / "Anchor" / "anchor.db"


@dataclass
class Settings:
    # Fixed product choices (from the owner)
    voice: str = "cedar"
    default_register: str = "playful"
    default_detour_minutes: int = 15

    # Models (architecture.md)
    judge_model: str = "gpt-5.6-luna"
    judge_fallback_model: str = "gpt-5.6-terra"
    tts_model: str = "gpt-4o-mini-tts"
    stt_model: str = "gpt-4o-mini-transcribe"

    # Behaviour constants
    max_confrontations_per_hour: int = 4
    roast_cooldown_s: int = 45           # minimum gap between two unsolicited roasts (persona pack default)
    cache_ttl_s: int = 1800
    hamming_threshold: int = 12          # dhash bits (of 64) that count as "pixels diverged"
    vision_cooldown_s: int = 60          # at most one forced "look" per context per minute (videos, animations)
    late_threshold_s: int = 120          # reminder later than this acknowledges the delay
    listen_window_s: float = 8.0
    tick_s: float = 1.0
    title_max_chars: int = 120
    image_long_edge: int = 768

    # Environment switches
    demo: bool = False                   # ANCHOR_DEMO=1: patience ÷ 3 (min 30 s), quicker pauses — for live demos
    silent: bool = False                 # ANCHOR_SILENT=1 starts muted
    exclude_titles: bool = False         # EXCLUDE_TITLES=1 sends app names only
    profanity_ok: bool = False           # ANCHOR_PROFANITY=1 (default off)
    db_path: Path = field(default_factory=default_db_path)

    @classmethod
    def from_env(cls) -> "Settings":
        load_env()
        s = cls()
        s.demo = _truthy("ANCHOR_DEMO")
        if s.demo:
            s.max_confrontations_per_hour = 12   # a rehearsal repeats the whole script several times an hour
            s.roast_cooldown_s = 0               # "still playing after the break" must be called within seconds
        # Persona intensity (playful | pointed | savage) maps onto the register ladder; it is selected, never escalated.
        intensity = os.environ.get("ANCHOR_INTENSITY", "").strip().lower()
        if intensity in INTENSITY_TO_REGISTER:
            s.default_register = INTENSITY_TO_REGISTER[intensity]
        s.silent = _truthy("ANCHOR_SILENT")
        s.exclude_titles = _truthy("EXCLUDE_TITLES")
        s.profanity_ok = _truthy("ANCHOR_PROFANITY")
        s.db_path = default_db_path()
        if os.environ.get("ANCHOR_VOICE"):
            s.voice = os.environ["ANCHOR_VOICE"]
        if os.environ.get("ANCHOR_JUDGE_MODEL"):
            s.judge_model = os.environ["ANCHOR_JUDGE_MODEL"]
        return s
