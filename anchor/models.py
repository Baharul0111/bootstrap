"""Shared types. Every module codes against these; keep them boring and stable."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol


class Register(str, Enum):
    DRY = "dry"
    PLAYFUL = "playful"
    SPICY = "spicy"


REGISTER_LADDER = [Register.DRY, Register.PLAYFUL, Register.SPICY]


def cool_register(register: Register) -> Register:
    """One step gentler. Floors at DRY. There is deliberately no ``warm_register``."""
    idx = REGISTER_LADDER.index(Register(register))
    return REGISTER_LADDER[max(0, idx - 1)]


class State(str, Enum):
    IDLE = "IDLE"
    ANCHORING = "ANCHORING"
    WATCHING = "WATCHING"
    DRIFTING = "DRIFTING"
    CONFRONTING = "CONFRONTING"
    PUSHED = "PUSHED"
    DETOUR = "DETOUR"
    SWITCH = "SWITCH"
    DONE = "DONE"


class Intent(str, Enum):
    DEFER = "DEFER"
    SWITCH = "SWITCH"
    DONE = "DONE"
    EVASIVE = "EVASIVE"


class Sentiment(str, Enum):
    AMUSED = "amused"
    NEUTRAL = "neutral"
    IRRITATED = "irritated"


@dataclass
class ContextFrame:
    """One sample of the desktop. ``image_jpeg`` is only filled on demand and never persisted."""

    ts: float
    app: str
    bundle_id: str = ""
    title: str = ""
    url_domain: str = ""
    idle_s: float = 0.0
    dhash: str = ""
    title_available: bool = True      # False when Accessibility is missing
    screen_available: bool = True     # False when Screen Recording is missing
    image_jpeg: Optional[bytes] = None


@dataclass
class Verdict:
    drift: float                 # 0..1, 1 = fully off task
    confidence: float            # 0..1
    reason: str = ""
    tolerated: bool = False      # expected surface (search, AI chat) — on-task for a while
    source: str = "llm"          # llm | llm+vision | cache | heuristic
    activity: str = ""           # short phrase of what the user is doing, for the roast

    @property
    def off_task(self) -> bool:
        return self.drift >= 0.5


@dataclass
class Policy:
    patience_seconds: int = 180
    pause_idle_s: Optional[int] = 4          # None => input idle is not a pause signal (watching)
    pause_stable_s: int = 10
    default_detour_minutes: int = 15
    humor_ok: bool = True
    register: Register = Register.PLAYFUL
    expected_surfaces: list[str] = field(default_factory=list)
    tolerance_seconds: int = 600             # how long an expected surface stays "part of the task"
    task_kind: str = "general"               # coding | writing | watching | reading | general
    language: str = "en"                     # en | hi


@dataclass
class Anchor:
    id: int
    verbatim: str
    clarified: str
    language: str
    status: str                              # active | retired | completed
    created_at: float
    ended_at: Optional[float] = None
    detour_until: Optional[float] = None     # absolute epoch seconds, survives sleep/restart
    detour_minutes: Optional[int] = None


@dataclass
class ReplyIntent:
    intent: Intent
    minutes: Optional[int] = None
    new_anchor: Optional[str] = None
    sentiment: Sentiment = Sentiment.NEUTRAL
    language: str = "en"


@dataclass
class PolicyDraft:
    clarified_anchor: str
    language: str
    policy: Policy
    needs_clarification: bool = False
    clarifying_question: str = ""


@dataclass
class Observed:
    """What the user is actually doing, phrased for the roast."""

    activity: str                # "researching whether crabs can swim on Google"
    app: str = ""
    title: str = ""
    url_domain: str = ""
    minutes_off_task: int = 0
    confidence: float = 1.0


@dataclass
class Composition:
    roast: str
    choice_line: str
    joke_used: bool


@dataclass
class Blockers:
    mic_in_use: bool = False
    do_not_disturb: bool = False
    screen_locked: bool = False
    muted: bool = False

    @property
    def any(self) -> bool:
        return self.mic_in_use or self.do_not_disturb or self.screen_locked or self.muted


class Speaker(Protocol):
    def speak(self, text: str, *, language: str = "en", tone: str = "neutral") -> None: ...


class Listener(Protocol):
    def listen(self, *, window_s: float = 8.0, language: str = "en") -> str: ...
