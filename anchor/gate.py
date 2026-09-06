"""Component 3 — Change gate.

Decides, for free and locally, whether a new ContextFrame is worth doing anything
about. Two signals are compared against the previous frame:

* the **fingerprint** — ``app|title_head|url_domain`` after normalisation, and
* the **pixel hash** — Hamming distance between consecutive dhashes.

| Fingerprint | Pixels    | Meaning                    | Decision                      |
|-------------|-----------|----------------------------|-------------------------------|
| same        | same      | nothing happened           | changed=False                 |
| changed     | any       | ordinary context switch    | changed=True, force_vision=False |
| same        | diverged  | in-page navigation / video | changed=True, force_vision=True  |

Pixel divergence on its own may only trigger a *look* (force_vision), never a
verdict — dhash is content-blind and a playing video changes it every frame.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from anchor.models import ContextFrame

# Trailing browser/editor decorations that titles carry. Matched case-insensitively
# after whitespace has been collapsed, so both " - " and " — " variants are listed.
APP_SUFFIXES = (
    " - google chrome",
    " - brave",
    " - microsoft edge",
    " - arc",
    " — mozilla firefox",
    " - mozilla firefox",
    " - safari",
    " - visual studio code",
    " — google chrome",
)

_WHITESPACE = re.compile(r"\s+")


def normalize_title(title: str, app: str, head: int = 60) -> str:
    """Lowercase, collapse whitespace, strip one trailing app suffix, keep the first ``head`` chars.

    The fixed ``APP_SUFFIXES`` table is tried first; then a dynamic ``" - <app>"`` /
    ``" — <app>"`` suffix built from the frontmost app's own name, so titles from
    apps not in the table still normalise cleanly.
    """
    text = _WHITESPACE.sub(" ", (title or "").lower()).strip()
    candidates: list[str] = list(APP_SUFFIXES)
    app_name = _WHITESPACE.sub(" ", (app or "").lower()).strip()
    if app_name:
        candidates += [f" - {app_name}", f" — {app_name}"]
    for suffix in candidates:
        if text.endswith(suffix):
            text = text[: -len(suffix)].rstrip()
            break
    return text[:head]


def normalize_fingerprint(app: str, title: str, url_domain: str) -> str:
    """``app|title_head|url_domain`` — the cache key and the change-gate identity of a context."""
    return f"{(app or '').lower().strip()}|{normalize_title(title, app)}|{(url_domain or '').lower().strip()}"


def hamming(hex_a: str, hex_b: str) -> int:
    """Bit distance between two hex-encoded dhashes.

    An empty or malformed hash (non-hex, or the two differ in length) yields 0:
    unknown counts as unchanged, so a missing Screen Recording grant can never
    manufacture a divergence.
    """
    a = (hex_a or "").strip()
    b = (hex_b or "").strip()
    if not a or not b or len(a) != len(b):
        return 0
    try:
        return (int(a, 16) ^ int(b, 16)).bit_count()
    except ValueError:
        return 0


@dataclass
class GateDecision:
    changed: bool               # anything to do at all
    fingerprint: str
    fingerprint_changed: bool
    pixels_diverged: bool       # hamming > threshold
    force_vision: bool          # fingerprint same AND pixels diverged (drift inside one app)
    distance: int


class ChangeGate:
    """Compares each frame against the previous one. The first frame always counts as a change."""

    def __init__(self, hamming_threshold: int = 12) -> None:
        self.hamming_threshold = hamming_threshold
        self.last_fingerprint: str = ""
        self.last_dhash: str = ""
        self._primed = False

    def update(self, frame: ContextFrame) -> GateDecision:
        fingerprint = normalize_fingerprint(frame.app, frame.title, frame.url_domain)
        dhash = frame.dhash or ""

        if not self._primed:
            self._primed = True
            self.last_fingerprint = fingerprint
            self.last_dhash = dhash
            return GateDecision(
                changed=True,
                fingerprint=fingerprint,
                fingerprint_changed=True,
                pixels_diverged=False,
                force_vision=False,
                distance=0,
            )

        distance = hamming(self.last_dhash, dhash)
        pixels_diverged = distance > self.hamming_threshold
        fingerprint_changed = fingerprint != self.last_fingerprint

        self.last_fingerprint = fingerprint
        self.last_dhash = dhash

        return GateDecision(
            changed=fingerprint_changed or pixels_diverged,
            fingerprint=fingerprint,
            fingerprint_changed=fingerprint_changed,
            pixels_diverged=pixels_diverged,
            force_vision=(not fingerprint_changed) and pixels_diverged,
            distance=distance,
        )

    def reset(self) -> None:
        """Forget the last frame; the next ``update`` behaves like a first frame."""
        self.last_fingerprint = ""
        self.last_dhash = ""
        self._primed = False
