"""Hermetic tests for the menu-bar shell, the CLI entry point and the doctor.

No run loop is started, no network is touched: the engine is a fake, the doctor's
checks are monkeypatched, and the OpenAI call is replaced.
"""

from __future__ import annotations

import io
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import pytest
import rumps

from anchor import __version__
from anchor.config import Settings
from anchor.models import State


class FakeEngine:
    def __init__(self, dot: str = "🟢", muted: bool = False) -> None:
        self.transcript: deque[str] = deque(maxlen=30)
        self.dot = dot
        self.muted = muted
        self.calls: list[str] = []

    def status(self):
        return SimpleNamespace(
            state=State.WATCHING,
            dot="🔇" if self.muted else self.dot,
            last_line=self.transcript[-1] if self.transcript else "",
            anchor_text="write the report",
            muted=self.muted,
            missing_permissions=[],
        )

    def start(self) -> None:
        self.calls.append("start")

    def stop(self) -> None:
        self.calls.append("stop")

    def set_muted(self, muted: bool) -> None:
        self.muted = muted

    def toggle_mute(self) -> bool:
        self.calls.append("toggle_mute")
        self.muted = not self.muted
        return self.muted

    def wrong_call(self) -> None:
        self.calls.append("wrong_call")


@pytest.fixture
def app():
    from anchor.app import AnchorApp

    engine = FakeEngine()
    out = io.StringIO()
    return AnchorApp(engine, stream=out), engine, out


def _titles(app) -> list[str]:
    return [app.menu[key].title for key in app.menu.keys()]


# ---- shell ------------------------------------------------------------------ #


def test_import_does_not_start_run_loop():
    import anchor.app  # noqa: F401

    assert getattr(rumps.App, "*app_instance", None) is None


def test_exactly_three_menu_items(app):
    shell, _, _ = app
    assert len(shell.menu) == 3
    assert _titles(shell) == ["Mute", "Wrong call", "Quit"]
    assert shell.quit_button is None
    assert shell.title == "⚪"


def test_menu_items_are_wired(app):
    shell, _, _ = app
    assert shell.mute_item.callback == shell.on_mute
    assert shell.wrong_call_item.callback == shell.on_wrong_call
    assert shell.quit_item.callback == shell.on_quit


def test_mute_click_flips_label_and_toggles_engine(app):
    shell, engine, _ = app
    shell.on_mute(shell.mute_item)
    assert shell.mute_item.title == "Unmute"
    assert engine.calls == ["toggle_mute"]
    shell.on_mute(shell.mute_item)
    assert shell.mute_item.title == "Mute"
    assert engine.calls == ["toggle_mute", "toggle_mute"]
    assert len(shell.menu) == 3


def test_wrong_call_calls_engine_without_notification(app, monkeypatch):
    shell, engine, _ = app

    def _boom(*_a, **_k):
        raise AssertionError("nothing must pop up")

    monkeypatch.setattr(rumps, "notification", _boom)
    shell.on_wrong_call(shell.wrong_call_item)
    assert engine.calls == ["wrong_call"]
    assert shell.title == "✓"


def test_refresh_updates_title_to_status_dot(app):
    shell, engine, _ = app
    shell._refresh()
    assert shell.title == "🟢"
    engine.dot = "🟠"
    shell._refresh()
    assert shell.title == "🟠"


def test_refresh_syncs_mute_label_from_status(app):
    shell, engine, _ = app
    engine.muted = True  # e.g. ANCHOR_SILENT=1 started the engine muted
    shell._refresh()
    assert shell.mute_item.title == "Unmute"
    assert shell.title == "🔇"
    engine.muted = False
    shell._refresh()
    assert shell.mute_item.title == "Mute"


def test_refresh_prints_only_new_transcript_lines(app):
    shell, engine, out = app
    engine.transcript.append("anchor: what are you working on?")
    engine.transcript.append("you: writing the report")
    shell._refresh()
    assert out.getvalue().splitlines() == [
        "anchor: what are you working on?",
        "you: writing the report",
    ]
    shell._refresh()
    assert len(out.getvalue().splitlines()) == 2
    engine.transcript.append("anchor: noted.")
    shell._refresh()
    assert out.getvalue().splitlines()[-1] == "anchor: noted."
    assert len(out.getvalue().splitlines()) == 3


def test_refresh_keeps_printing_after_deque_wraps(app):
    shell, engine, out = app
    for i in range(20):
        engine.transcript.append(f"line {i}")
    shell._refresh()
    for i in range(20, 45):  # deque maxlen is 30, so the front falls off
        engine.transcript.append(f"line {i}")
    shell._refresh()
    shell._refresh()
    assert out.getvalue().splitlines() == [f"line {i}" for i in range(45)]


def test_quit_stops_engine_then_quits(app, monkeypatch):
    shell, engine, _ = app
    order: list[str] = []
    engine.stop = lambda: order.append("engine.stop")  # type: ignore[assignment]
    monkeypatch.setattr(rumps, "quit_application", lambda sender=None: order.append("quit_application"))
    shell.on_quit(shell.quit_item)
    assert order == ["engine.stop", "quit_application"]


def test_transcript_tail_alignment():
    from anchor.app import TranscriptTail

    tail = TranscriptTail()
    assert tail.drain(["a", "b"]) == ["a", "b"]
    assert tail.drain(["a", "b"]) == []
    assert tail.drain(["b", "c", "d"]) == ["c", "d"]   # "a" fell off the front
    assert tail.drain(["x", "y"]) == ["x", "y"]        # no overlap at all: everything is new
    assert tail.printed == 6


# ---- CLI -------------------------------------------------------------------- #


def test_cli_parser_defaults_to_app():
    from anchor.__main__ import build_parser

    assert build_parser().parse_args([]).command == "app"
    assert build_parser().parse_args(["doctor"]).command == "doctor"
    assert build_parser().parse_args(["headless"]).command == "headless"


def test_cli_version(capsys):
    from anchor.__main__ import main

    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == f"anchor {__version__}"


# ---- doctor ----------------------------------------------------------------- #


def _settings(tmp_path: Path) -> Settings:
    return Settings(db_path=tmp_path / "anchor.db")


def _all_pass(monkeypatch):
    from anchor import doctor

    for name in (
        "check_api_key", "check_accessibility", "check_screen_recording",
        "check_microphone", "check_say", "check_db_path", "check_python",
    ):
        label = name.removeprefix("check_")
        monkeypatch.setattr(
            doctor, name,
            lambda _s, label=label: doctor.Check(label, True, True, detail="ok"),
        )


def test_doctor_all_pass_returns_zero(monkeypatch, tmp_path, capsys):
    from anchor import doctor

    _all_pass(monkeypatch)
    assert doctor.run_doctor(_settings(tmp_path)) == 0
    out = capsys.readouterr().out
    assert out.count("✓") == 7
    assert "✗" not in out


def test_doctor_key_failure_returns_one(monkeypatch, tmp_path, capsys):
    from anchor import doctor

    _all_pass(monkeypatch)
    monkeypatch.setattr(
        doctor, "check_api_key",
        lambda _s: doctor.Check("OpenAI API key", False, True, hint="put OPENAI_API_KEY=... in .env (setup.md #1)"),
    )
    assert doctor.run_doctor(_settings(tmp_path)) == 1
    out = capsys.readouterr().out
    assert "✗ OpenAI API key" in out
    assert "setup.md #1" in out


def test_doctor_optional_failure_still_returns_zero(monkeypatch, tmp_path, capsys):
    from anchor import doctor

    _all_pass(monkeypatch)
    monkeypatch.setattr(
        doctor, "check_accessibility",
        lambda _s: doctor.Check("Accessibility", False, False, hint="add your terminal (setup.md #2)"),
    )
    assert doctor.run_doctor(_settings(tmp_path)) == 0
    assert "✗ Accessibility (optional)" in capsys.readouterr().out


def test_doctor_never_prints_the_key(monkeypatch, tmp_path, capsys):
    from anchor import doctor

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-SECRET")
    monkeypatch.setattr(doctor, "load_env", lambda: None)

    def _leaky(timeout=15):
        raise RuntimeError("401 for key sk-test-SECRET")

    monkeypatch.setattr(doctor, "_list_model_ids", _leaky)
    rc = doctor.run_doctor(_settings(tmp_path))
    out = capsys.readouterr().out
    assert rc == 1
    assert "SECRET" not in out
    assert "✗ OpenAI API key" in out


def test_doctor_key_check_wants_all_four_models(monkeypatch, tmp_path):
    from anchor import doctor

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-SECRET")
    monkeypatch.setattr(doctor, "load_env", lambda: None)
    settings = _settings(tmp_path)
    monkeypatch.setattr(doctor, "_list_model_ids", lambda timeout=15: {
        settings.judge_model, settings.judge_fallback_model, settings.tts_model, settings.stt_model,
    })
    assert doctor.check_api_key(settings).ok
    monkeypatch.setattr(doctor, "_list_model_ids", lambda timeout=15: {settings.tts_model})
    check = doctor.check_api_key(settings)
    assert not check.ok and settings.judge_model in check.hint and "SECRET" not in check.hint


def test_doctor_real_local_checks_run(tmp_path):
    from anchor import doctor

    settings = _settings(tmp_path)
    assert doctor.check_python(settings).ok
    assert doctor.check_say(settings).ok
    assert doctor.check_db_path(settings).ok
    assert doctor.check_db_path(Settings(db_path=Path("/nonexistent-root-dir/anchor.db"))).ok is False
