"""Tests for anchor.sensors: FakeSensor, MacSensor degradation, permissions, no-disk guarantee."""

from __future__ import annotations

import builtins
import io
import re
import time

import pytest

from anchor import sensors
from anchor.config import Settings
from anchor.models import Blockers, ContextFrame
from anchor.sensors import MAC_AVAILABLE, FakeSensor, MacSensor

HEX16 = re.compile(r"^[0-9a-f]{16}$")

HELPERS = [
    "_frontmost",
    "_read_title",
    "_title_ax",
    "_title_cg",
    "_url_domain",
    "_browser_url",
    "_idle_s",
    "_screen_hash",
    "_capture",
    "_screen_locked",
    "_do_not_disturb",
    "_mic_in_use",
]


def _boom(*_args, **_kwargs):
    raise RuntimeError("helper exploded")


# --------------------------------------------------------------------------- #
# FakeSensor
# --------------------------------------------------------------------------- #


def test_fake_sensor_defaults():
    fake = FakeSensor()
    frame = fake.sample()
    assert isinstance(frame, ContextFrame)
    assert frame.app == "FakeApp"
    assert fake.grab_jpeg() == b"\xff\xd8fake"
    assert fake.grab_calls == 1
    assert fake.blockers() == Blockers()
    assert fake.missing_permissions() == []


def test_fake_sensor_set_mutates_current_frame():
    fake = FakeSensor()
    fake.set(app="Safari", title="Crabs - Wikipedia", url_domain="wikipedia.org", idle_s=3.5)
    frame = fake.sample()
    assert frame.app == "Safari"
    assert frame.title == "Crabs - Wikipedia"
    assert frame.url_domain == "wikipedia.org"
    assert frame.idle_s == 3.5
    # ts is returned as-is, never rewritten
    ts = fake.frame.ts
    assert fake.sample().ts == ts


def test_fake_sensor_frames_served_in_order_then_sticks():
    a = ContextFrame(ts=1.0, app="A")
    b = ContextFrame(ts=2.0, app="B")
    fake = FakeSensor([a, b])
    assert fake.frame is a
    assert fake.sample().app == "A"
    assert fake.sample().app == "B"
    assert fake.sample().app == "B"
    assert fake.sample_calls == 3


def test_fake_sensor_blockers_and_missing():
    fake = FakeSensor()
    fake.set_blockers(mic_in_use=True)
    assert fake.blockers().mic_in_use is True
    assert fake.blockers().muted is False
    assert fake.blockers().any is True
    fake.missing = ["Accessibility"]
    assert fake.missing_permissions() == ["Accessibility"]
    fake.jpeg = None
    assert fake.grab_jpeg() is None
    assert fake.grab_calls == 1


# --------------------------------------------------------------------------- #
# MacSensor degradation (runs on any platform via monkeypatching)
# --------------------------------------------------------------------------- #


def test_sample_degrades_to_app_only_when_titles_unreadable(monkeypatch):
    sensor = MacSensor(Settings())
    monkeypatch.setattr(sensor, "_frontmost", lambda: ("TestApp", "com.test.app", 4242))
    monkeypatch.setattr(sensors, "_ax_trusted", lambda: False)
    monkeypatch.setattr(sensor, "_title_ax", _boom)
    monkeypatch.setattr(sensor, "_title_cg", _boom)
    monkeypatch.setattr(sensor, "_screen_hash", _boom)
    monkeypatch.setattr(sensor, "_idle_s", lambda: 2.0)

    frame = sensor.sample()

    assert frame.app == "TestApp"
    assert frame.bundle_id == "com.test.app"
    assert frame.title == ""
    assert frame.title_available is False
    assert frame.dhash == ""
    assert frame.screen_available is False
    assert frame.idle_s == 2.0


def test_sample_never_raises_when_every_helper_raises(monkeypatch):
    sensor = MacSensor(Settings())
    for name in HELPERS:
        monkeypatch.setattr(sensor, name, _boom)
    monkeypatch.setattr(sensors, "_ax_trusted", _boom)
    monkeypatch.setattr(sensors, "_screen_capture_allowed", _boom)

    frame = sensor.sample()
    assert isinstance(frame, ContextFrame)
    assert frame.app == ""
    assert frame.title == ""
    assert frame.title_available is False
    assert frame.screen_available is False
    assert frame.idle_s == 0.0
    assert frame.image_jpeg is None

    assert sensor.grab_jpeg() is None
    assert sensor.blockers() == Blockers()
    assert sensor.missing_permissions() == ["Accessibility", "Screen Recording"]


def test_title_trimmed_and_exclude_titles_mode(monkeypatch):
    settings = Settings()
    settings.title_max_chars = 10
    sensor = MacSensor(settings)
    monkeypatch.setattr(sensor, "_frontmost", lambda: ("TestApp", "com.test.app", 1))
    monkeypatch.setattr(sensors, "_ax_trusted", lambda: True)
    monkeypatch.setattr(sensor, "_title_ax", lambda: "A very long window title indeed")
    monkeypatch.setattr(sensor, "_screen_hash", lambda: ("", False))
    monkeypatch.setattr(sensor, "_idle_s", lambda: 0.0)

    frame = sensor.sample()
    assert frame.title == "A very lon"
    assert frame.title_available is True

    settings.exclude_titles = True
    frame = sensor.sample()
    assert frame.title == ""
    assert frame.title_available is True


def test_cg_title_is_used_when_ax_missing(monkeypatch):
    sensor = MacSensor(Settings())
    monkeypatch.setattr(sensor, "_frontmost", lambda: ("TestApp", "com.test.app", 7))
    monkeypatch.setattr(sensors, "_ax_trusted", lambda: False)
    monkeypatch.setattr(sensor, "_title_ax", _boom)
    monkeypatch.setattr(sensor, "_title_cg", lambda pid: f"Window of {pid}")
    monkeypatch.setattr(sensor, "_screen_hash", lambda: ("", False))
    monkeypatch.setattr(sensor, "_idle_s", lambda: 0.0)

    frame = sensor.sample()
    assert frame.title == "Window of 7"
    assert frame.title_available is True


def test_url_domain_parsing_and_failure_cache(monkeypatch):
    assert sensors._domain_from_url("https://www.youtube.com/watch?v=x") == "youtube.com"
    assert sensors._domain_from_url("http://docs.python.org/3/") == "docs.python.org"
    assert sensors._domain_from_url("") == ""
    assert sensors._domain_from_url("not a url") == ""

    sensor = MacSensor(Settings())
    calls = []

    def fake_browser_url(bundle_id, expr):
        calls.append(bundle_id)
        raise TimeoutError("osascript timed out")

    monkeypatch.setattr(sensor, "_browser_url", fake_browser_url)
    assert sensor._url_domain("com.google.Chrome") == ""
    assert sensor._url_domain("com.google.Chrome") == ""
    assert calls == ["com.google.Chrome"], "failure must be cached, not retried every second"
    assert sensor._url_domain("com.apple.TextEdit") == ""
    assert calls == ["com.google.Chrome"], "non-browsers never run osascript"

    monkeypatch.setattr(sensor, "_browser_url", lambda b, e: "https://www.github.com/x")
    assert sensor._url_domain("com.apple.Safari") == "github.com"


def test_missing_permissions_reports_accessibility(monkeypatch):
    sensor = MacSensor(Settings())
    monkeypatch.setattr(sensors, "_ax_trusted", lambda: False)
    monkeypatch.setattr(sensors, "_screen_capture_allowed", lambda: True)
    assert sensor.missing_permissions() == ["Accessibility"]

    monkeypatch.setattr(sensors, "_ax_trusted", lambda: True)
    monkeypatch.setattr(sensors, "_screen_capture_allowed", lambda: False)
    assert sensor.missing_permissions() == ["Screen Recording"]


@pytest.mark.skipif(not MAC_AVAILABLE, reason="needs PyObjC ApplicationServices")
def test_missing_permissions_uses_real_ax_call(monkeypatch):
    monkeypatch.setattr(sensors.ApplicationServices, "AXIsProcessTrusted", lambda: False)
    assert "Accessibility" in MacSensor(Settings()).missing_permissions()


def test_do_not_disturb_reads_assertions(monkeypatch, tmp_path):
    sensor = MacSensor(Settings())
    path = tmp_path / "Assertions.json"
    monkeypatch.setattr(sensors, "DND_ASSERTIONS_PATH", str(path))
    assert sensor._do_not_disturb() is False  # missing file
    path.write_text('{"data": [{"storeAssertionRecords": []}]}')
    assert sensor._do_not_disturb() is False
    path.write_text('{"data": [{"storeAssertionRecords": [{"assertionDetails": {}}]}]}')
    assert sensor._do_not_disturb() is True
    path.write_text("not json")
    assert sensor._do_not_disturb() is False


# --------------------------------------------------------------------------- #
# No-disk guarantee and live smoke (mac only)
# --------------------------------------------------------------------------- #


@pytest.mark.skipif(not MAC_AVAILABLE, reason="needs the real macOS sensor")
def test_no_disk_writes_during_sample_and_grab(monkeypatch):
    from PIL import Image

    real_save = Image.Image.save
    real_open = builtins.open
    save_targets = []

    def guarded_save(self, fp, *args, **kwargs):
        save_targets.append(fp)
        assert isinstance(fp, io.BytesIO), f"image saved to non-memory target: {fp!r}"
        return real_save(self, fp, *args, **kwargs)

    def guarded_open(file, mode="r", *args, **kwargs):
        if any(flag in str(mode) for flag in ("w", "a", "x", "+")):
            raise AssertionError(f"write open attempted during sensing: {file!r} mode={mode!r}")
        return real_open(file, mode, *args, **kwargs)

    monkeypatch.setattr(Image.Image, "save", guarded_save)
    monkeypatch.setattr(builtins, "open", guarded_open)

    sensor = MacSensor(Settings())
    frame = sensor.sample()
    jpeg = sensor.grab_jpeg()
    sensor.blockers()

    assert isinstance(frame, ContextFrame)
    if frame.screen_available:
        assert jpeg is not None and jpeg[:2] == b"\xff\xd8"
        assert len(save_targets) == 1
    else:
        assert jpeg is None


@pytest.mark.skipif(not MAC_AVAILABLE, reason="live macOS smoke")
def test_live_smoke():
    sensor = MacSensor(Settings())
    sensor.sample()  # warm up lazy state
    t0 = time.perf_counter()
    frame = sensor.sample()
    elapsed = time.perf_counter() - t0

    assert frame.app
    assert frame.idle_s >= 0
    assert frame.dhash == "" or HEX16.match(frame.dhash)
    # Budget is 1 Hz; allow headroom for a loaded machine (parallel test runs, a live Anchor process).
    # Two AppleScript reads can each time out at 1.5 s when another Anchor instance holds the same permissions.
    assert elapsed < 5.0, f"sample() took {elapsed:.3f}s"

    blockers = sensor.blockers()
    assert isinstance(blockers, Blockers)
    assert blockers.muted is False
    assert set(sensor.missing_permissions()) <= {"Accessibility", "Screen Recording"}
