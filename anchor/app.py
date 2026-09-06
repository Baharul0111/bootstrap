"""Component 1 — the menu-bar shell. One dot, three menu items, no window.

``rumps`` owns the NSApplication run loop on the main thread. Everything that can
block (network, audio, the judge) lives in the engine's worker thread. This module
only reads ``engine.status()`` snapshots twice a second, flips the title glyph and
the Mute label, and echoes new transcript lines to the terminal for the demo.
Nothing here ever opens a window or posts a notification.
"""

from __future__ import annotations

import signal
import sys
import time
from typing import Any, Optional, TextIO

import rumps

from .config import Settings

APP_NAME = "Anchor"
IDLE_DOT = "⚪"
FEEDBACK_DOT = "✓"           # shown briefly after "Wrong call" (no popup, just the glyph)
FEEDBACK_HOLD_S = 1.5
REFRESH_INTERVAL_S = 0.5

MENU_MUTE = "Mute"
MENU_UNMUTE = "Unmute"
MENU_WRONG_CALL = "Wrong call"
MENU_QUIT = "Quit"


def hide_from_dock() -> None:
    """Run as an accessory app: no Dock icon, no app menu, no window. Safe to call twice."""
    try:
        from AppKit import NSBundle

        NSBundle.mainBundle().infoDictionary()["LSUIElement"] = "1"
    except Exception:
        pass
    try:
        from AppKit import NSApplication, NSApplicationActivationPolicyAccessory

        NSApplication.sharedApplication().setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    except Exception:
        pass


class TranscriptTail:
    """Remembers what has already been printed from the engine's bounded transcript deque.

    The deque keeps only the last 30 lines, so once it wraps a plain index no longer
    works; instead the previous snapshot is aligned against the current one and only
    the lines after the overlap are returned.
    """

    def __init__(self) -> None:
        self._last: list[str] = []
        self.printed = 0

    def drain(self, transcript: Any) -> list[str]:
        snapshot = getattr(transcript, "snapshot", None)
        if callable(snapshot):
            # The engine's transcript counts every line ever appended: exact, even after the wrap.
            lines, total = snapshot()
            current = [str(line) for line in lines]
            fresh = min(len(current), max(0, total - self.printed))   # more than maxlen new: the oldest are gone
            self._last = current
            self.printed = total
            return current[len(current) - fresh:] if fresh else []
        try:
            current = [str(line) for line in (transcript or ())]
        except RuntimeError:            # a plain deque appended to mid-iteration: pick it up on the next tick
            return []
        prev = self._last
        if current[: len(prev)] == prev:
            new = current[len(prev):]
        else:
            overlap = 0
            for n in range(min(len(prev), len(current)), 0, -1):
                if prev[-n:] == current[:n]:
                    overlap = n
                    break
            new = current[overlap:]
        self._last = current
        self.printed += len(new)
        return new


class AnchorApp(rumps.App):
    """The dot. Title is the glyph only; the menu is exactly Mute / Wrong call / Quit."""

    def __init__(self, engine: Any, stream: Optional[TextIO] = None) -> None:
        super().__init__(APP_NAME, title=IDLE_DOT, quit_button=None)
        self.engine = engine
        self._out: TextIO = stream or sys.stdout
        self._tail = TranscriptTail()
        self._feedback_until = 0.0
        self._stopped = False
        self.mute_item = rumps.MenuItem(MENU_MUTE, callback=self.on_mute)
        self.wrong_call_item = rumps.MenuItem(MENU_WRONG_CALL, callback=self.on_wrong_call)
        self.quit_item = rumps.MenuItem(MENU_QUIT, callback=self.on_quit)
        self.menu = [self.mute_item, self.wrong_call_item, self.quit_item]
        self._refresh_timer = rumps.Timer(self._refresh, REFRESH_INTERVAL_S)

    # ---- menu actions (main thread, never block) -------------------------- #

    def on_mute(self, sender: Any = None) -> None:
        muted = bool(self.engine.toggle_mute())
        self._set_mute_label(muted)

    def on_wrong_call(self, sender: Any = None) -> None:
        self.engine.wrong_call()
        self.title = FEEDBACK_DOT
        self._feedback_until = time.monotonic() + FEEDBACK_HOLD_S

    def on_quit(self, sender: Any = None) -> None:
        self._shutdown()
        rumps.quit_application(sender)

    # ---- periodic refresh (main thread) ----------------------------------- #

    def _refresh(self, _timer: Any = None) -> None:
        try:
            status = self.engine.status()
        except Exception:
            return
        if time.monotonic() >= self._feedback_until:
            dot = getattr(status, "dot", None) or IDLE_DOT
            if dot != self.title:
                self.title = dot
        self._set_mute_label(bool(getattr(status, "muted", False)))
        for line in self._tail.drain(getattr(self.engine, "transcript", ())):
            print(line, file=self._out, flush=True)

    def _set_mute_label(self, muted: bool) -> None:
        label = MENU_UNMUTE if muted else MENU_MUTE
        if self.mute_item.title != label:
            self.mute_item.title = label

    # ---- lifecycle -------------------------------------------------------- #

    def _shutdown(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        try:
            self._refresh_timer.stop()
        except Exception:
            pass
        try:
            self.engine.stop()
        except Exception as exc:  # never let a bad stop keep the process alive
            print(f"anchor: engine.stop() failed: {exc}", file=sys.stderr)

    def _install_signal_handlers(self) -> None:
        def _handle(signum: int, _frame: Any) -> None:
            self._shutdown()
            rumps.quit_application()

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, _handle)
            except (ValueError, OSError):
                pass

    def run(self, **options: Any) -> None:
        """Hide from the Dock, start the engine's worker thread, then hand the main thread to rumps."""
        hide_from_dock()
        self._install_signal_handlers()
        self.engine.start()
        self._refresh_timer.start()
        try:
            super().run(**options)
        finally:
            self._shutdown()


def main(settings: Optional[Settings] = None) -> int:
    settings = settings or Settings.from_env()
    from .engine import build_engine  # lazy: the engine pulls in audio and the OpenAI client

    engine = build_engine(settings)
    AnchorApp(engine).run()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
