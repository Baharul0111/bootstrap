"""Entry point: ``python -m anchor [app|doctor|headless|version]``.

- ``app`` (default): the menu-bar dot.
- ``doctor``: one line per environment check, exit 0 when every required check passes.
- ``headless``: no menu bar; runs the engine in the terminal until Ctrl-C (for debugging).
- ``version``: print the version and exit.
"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
from typing import Optional, Sequence

from . import __version__
from .config import Settings

COMMANDS = ("app", "doctor", "headless", "version")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m anchor",
        description="Anchor — a quiet menu-bar helper that holds you to one spoken sentence.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="app",
        choices=COMMANDS,
        help="app (default, menu bar) | doctor (environment checks) | headless (terminal only) | version",
    )
    return parser


def cmd_app(settings: Settings) -> int:
    from .app import main as app_main

    return int(app_main(settings) or 0)


def cmd_doctor(settings: Settings) -> int:
    from .doctor import run_doctor

    return run_doctor(settings)


def cmd_headless(settings: Settings) -> int:
    """Run the engine without a menu bar and echo its transcript until Ctrl-C."""
    from .app import TranscriptTail
    from .engine import build_engine

    stop = threading.Event()

    def _request_stop(_signum: int, _frame: object) -> None:
        stop.set()

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _request_stop)

    engine = build_engine(settings)
    engine.start()
    print("anchor headless: running, Ctrl-C to stop", flush=True)
    tail = TranscriptTail()
    last_dot: Optional[str] = None
    try:
        while not stop.is_set():
            status = engine.status()
            dot = getattr(status, "dot", "")
            if dot != last_dot:
                state = getattr(getattr(status, "state", ""), "value", getattr(status, "state", ""))
                print(f"{dot} [{state}]", flush=True)
                last_dot = dot
            for line in tail.drain(getattr(engine, "transcript", ())):
                print(line, flush=True)
            stop.wait(0.5)
    finally:
        engine.stop()
        print("anchor headless: stopped", flush=True)
    return 0


def cmd_version(_settings: Optional[Settings] = None) -> int:
    print(f"anchor {__version__}")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    if args.command == "version":
        return cmd_version()
    settings = Settings.from_env()
    if args.command == "doctor":
        return cmd_doctor(settings)
    if args.command == "headless":
        return cmd_headless(settings)
    return cmd_app(settings)


if __name__ == "__main__":
    sys.exit(main())
