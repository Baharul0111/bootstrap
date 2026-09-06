"""macOS sensors: one ``ContextFrame`` per second, degrading to app-name-only.

Every external read (frontmost app, window title, browser URL, idle seconds,
screen hash, blockers) is wrapped in try/except. ``MacSensor.sample()`` never
raises. Nothing is ever written to disk: screenshots live in memory only.

The module stays importable on non-mac machines; ``MAC_AVAILABLE`` says whether
the PyObjC frameworks loaded.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import io
import json
import os
import struct
import subprocess
import time
from dataclasses import replace
from typing import Optional
from urllib.parse import urlparse

from anchor.config import Settings
from anchor.models import Blockers, ContextFrame

try:  # PyObjC frameworks (mac only)
    import AppKit
    import ApplicationServices
    import Quartz

    MAC_AVAILABLE = True
except Exception:  # pragma: no cover - exercised only off-mac
    AppKit = None  # type: ignore[assignment]
    ApplicationServices = None  # type: ignore[assignment]
    Quartz = None  # type: ignore[assignment]
    MAC_AVAILABLE = False

try:  # imaging stack for the screen hash
    import dhash
    import numpy as np
    from PIL import Image

    IMAGING_AVAILABLE = True
except Exception:  # pragma: no cover
    dhash = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    Image = None  # type: ignore[assignment]
    IMAGING_AVAILABLE = False


OSASCRIPT_TIMEOUT_S = 1.5
URL_FAILURE_TTL_S = 60.0

# Bundle id -> AppleScript expression that yields the front tab's URL.
CHROMIUM_URL = "URL of active tab of front window"
SAFARI_URL = "URL of front document"
BROWSER_URL_SCRIPTS: dict[str, str] = {
    "com.google.Chrome": CHROMIUM_URL,
    "com.brave.Browser": CHROMIUM_URL,
    "com.microsoft.edgemac": CHROMIUM_URL,
    "company.thebrowser.Browser": CHROMIUM_URL,  # Arc
    "com.vivaldi.Vivaldi": CHROMIUM_URL,
    "org.chromium.Chromium": CHROMIUM_URL,
    "com.apple.Safari": SAFARI_URL,
}

TITLE_SCRIPT = (
    'tell application "System Events" to tell '
    "(first application process whose frontmost is true) to get name of front window"
)

DND_ASSERTIONS_PATH = os.path.join(
    os.path.expanduser("~"), "Library", "DoNotDisturb", "DB", "Assertions.json"
)


# --------------------------------------------------------------------------- #
# Small module-level helpers (kept separate so tests can monkeypatch them)
# --------------------------------------------------------------------------- #


def _ax_trusted() -> bool:
    """True when this process holds the Accessibility grant."""
    if not MAC_AVAILABLE:
        return False
    return bool(ApplicationServices.AXIsProcessTrusted())


def _screen_capture_allowed() -> bool:
    """True when this process holds the Screen Recording grant."""
    if not MAC_AVAILABLE:
        return False
    return bool(Quartz.CGPreflightScreenCaptureAccess())


def _run_osascript(script: str, timeout: float = OSASCRIPT_TIMEOUT_S) -> str:
    """Run one AppleScript line and return stdout stripped. Raises on any failure."""
    proc = subprocess.run(
        ["osascript", "-e", script],
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"osascript failed ({proc.returncode}): {proc.stderr.strip()[:120]}")
    return proc.stdout.strip()


def _domain_from_url(url: str) -> str:
    """Hostname of ``url`` with a leading ``www.`` stripped; "" if none."""
    if not url:
        return ""
    host = urlparse(url.strip()).hostname or ""
    host = host.lower()
    if host.startswith("www."):
        host = host[4:]
    return host


def _fourcc(code: str) -> int:
    return struct.unpack(">I", code.encode("ascii"))[0]


class _AudioObjectPropertyAddress(ctypes.Structure):
    _fields_ = [
        ("mSelector", ctypes.c_uint32),
        ("mScope", ctypes.c_uint32),
        ("mElement", ctypes.c_uint32),
    ]


class _CoreAudio:
    """Thin ctypes binding over the three CoreAudio calls we need for mic detection."""

    SYSTEM_OBJECT = 1

    def __init__(self) -> None:
        path = ctypes.util.find_library("CoreAudio")
        if not path:
            raise OSError("CoreAudio framework not found")
        lib = ctypes.cdll.LoadLibrary(path)
        addr_p = ctypes.POINTER(_AudioObjectPropertyAddress)
        u32_p = ctypes.POINTER(ctypes.c_uint32)
        lib.AudioObjectGetPropertyDataSize.restype = ctypes.c_int32
        lib.AudioObjectGetPropertyDataSize.argtypes = [
            ctypes.c_uint32, addr_p, ctypes.c_uint32, ctypes.c_void_p, u32_p,
        ]
        lib.AudioObjectGetPropertyData.restype = ctypes.c_int32
        lib.AudioObjectGetPropertyData.argtypes = [
            ctypes.c_uint32, addr_p, ctypes.c_uint32, ctypes.c_void_p, u32_p, ctypes.c_void_p,
        ]
        self.lib = lib
        self.sel_devices = _fourcc("dev#")
        self.sel_streams = _fourcc("stm#")
        self.sel_running = _fourcc("goin")
        self.scope_global = _fourcc("glob")
        self.scope_input = _fourcc("inpt")

    def _size(self, obj: int, selector: int, scope: int) -> int:
        addr = _AudioObjectPropertyAddress(selector, scope, 0)
        size = ctypes.c_uint32(0)
        status = self.lib.AudioObjectGetPropertyDataSize(obj, ctypes.byref(addr), 0, None, ctypes.byref(size))
        if status != 0:
            raise OSError(f"AudioObjectGetPropertyDataSize status {status}")
        return size.value

    def devices(self) -> list[int]:
        size = self._size(self.SYSTEM_OBJECT, self.sel_devices, self.scope_global)
        count = size // ctypes.sizeof(ctypes.c_uint32)
        if count == 0:
            return []
        buf = (ctypes.c_uint32 * count)()
        addr = _AudioObjectPropertyAddress(self.sel_devices, self.scope_global, 0)
        sz = ctypes.c_uint32(size)
        status = self.lib.AudioObjectGetPropertyData(
            self.SYSTEM_OBJECT, ctypes.byref(addr), 0, None, ctypes.byref(sz), buf
        )
        if status != 0:
            raise OSError(f"AudioObjectGetPropertyData status {status}")
        return list(buf)

    def has_input(self, device: int) -> bool:
        return self._size(device, self.sel_streams, self.scope_input) > 0

    def is_running_somewhere(self, device: int) -> bool:
        addr = _AudioObjectPropertyAddress(self.sel_running, self.scope_global, 0)
        value = ctypes.c_uint32(0)
        sz = ctypes.c_uint32(ctypes.sizeof(value))
        status = self.lib.AudioObjectGetPropertyData(
            device, ctypes.byref(addr), 0, None, ctypes.byref(sz), ctypes.byref(value)
        )
        if status != 0:
            raise OSError(f"AudioObjectGetPropertyData status {status}")
        return value.value == 1

    def any_input_running(self) -> bool:
        for device in self.devices():
            try:
                if self.has_input(device) and self.is_running_somewhere(device):
                    return True
            except Exception:
                continue
        return False


# --------------------------------------------------------------------------- #
# MacSensor
# --------------------------------------------------------------------------- #


class MacSensor:
    """Reads the desktop once per call. Every helper degrades on its own; nothing raises."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._last_image = None  # PIL image of the last successful capture, memory only
        self._url_failures: dict[str, float] = {}  # bundle id -> monotonic time of last failure
        self._coreaudio: Optional[_CoreAudio] = None
        self._coreaudio_failed = False

    # ---- public API ------------------------------------------------------- #

    def sample(self) -> ContextFrame:
        """One frame of the desktop. Never raises; degrades to app-name-only."""
        ts = time.time()
        app, bundle_id, pid = "", "", -1
        try:
            app, bundle_id, pid = self._frontmost()
        except Exception:
            pass

        title, title_available = "", False
        try:
            title, title_available = self._read_title(pid)
        except Exception:
            title, title_available = "", bool(self.settings.exclude_titles)

        url_domain = ""
        try:
            url_domain = self._url_domain(bundle_id)
        except Exception:
            url_domain = ""

        idle_s = 0.0
        try:
            idle_s = float(self._idle_s())
        except Exception:
            idle_s = 0.0

        screen_hash, screen_available = "", False
        try:
            screen_hash, screen_available = self._screen_hash()
        except Exception:
            screen_hash, screen_available = "", False

        try:
            return ContextFrame(
                ts=ts,
                app=app or "",
                bundle_id=bundle_id or "",
                title=title or "",
                url_domain=url_domain or "",
                idle_s=max(0.0, idle_s),
                dhash=screen_hash or "",
                title_available=bool(title_available),
                screen_available=bool(screen_available),
            )
        except Exception:
            return ContextFrame(ts=ts, app="", title_available=False, screen_available=False)

    def grab_jpeg(self) -> Optional[bytes]:
        """Downscaled JPEG of the last captured screen, in memory. None if unavailable."""
        try:
            if self._last_image is None:
                self._screen_hash()
            image = self._last_image
            if image is None or Image is None:
                return None
            thumb = image.copy()
            edge = int(self.settings.image_long_edge)
            thumb.thumbnail((edge, edge))
            buf = io.BytesIO()
            thumb.save(buf, "JPEG", quality=70)
            return buf.getvalue()
        except Exception:
            return None

    def blockers(self) -> Blockers:
        """mic_in_use / do_not_disturb / screen_locked. ``muted`` is left False for the engine."""
        blockers = Blockers()
        try:
            blockers.mic_in_use = bool(self._mic_in_use())
        except Exception:
            blockers.mic_in_use = False
        try:
            blockers.do_not_disturb = bool(self._do_not_disturb())
        except Exception:
            blockers.do_not_disturb = False
        try:
            blockers.screen_locked = bool(self._screen_locked())
        except Exception:
            blockers.screen_locked = False
        return blockers

    def missing_permissions(self) -> list[str]:
        """Subset of ["Accessibility", "Screen Recording"] that this process lacks."""
        missing: list[str] = []
        try:
            if not _ax_trusted():
                missing.append("Accessibility")
        except Exception:
            missing.append("Accessibility")
        try:
            if not _screen_capture_allowed():
                missing.append("Screen Recording")
        except Exception:
            missing.append("Screen Recording")
        return missing

    # ---- frontmost app ---------------------------------------------------- #

    def _frontmost(self) -> tuple[str, str, int]:
        """(localized name, bundle id, pid) of the frontmost application."""
        if not MAC_AVAILABLE:
            raise RuntimeError("AppKit unavailable")
        running = AppKit.NSWorkspace.sharedWorkspace().frontmostApplication()
        if running is None:
            raise RuntimeError("no frontmost application")
        name = running.localizedName() or ""
        bundle = running.bundleIdentifier() or ""
        pid = int(running.processIdentifier())
        return str(name), str(bundle), pid

    # ---- window title ----------------------------------------------------- #

    def _read_title(self, pid: int) -> tuple[str, bool]:
        """(title, title_available). Empty title when nothing could be read."""
        if self.settings.exclude_titles:
            return "", True
        title = ""
        available = False
        if _ax_trusted():
            available = True
            try:
                title = self._title_ax()
            except Exception:
                title = ""
        if not title:
            try:
                title = self._title_cg(pid)
            except Exception:
                title = ""
        title = (title or "").strip()
        if title:
            available = True
        limit = int(self.settings.title_max_chars)
        if limit > 0:
            title = title[:limit]
        return title, available

    def _title_ax(self) -> str:
        """Front window name via System Events (needs Accessibility)."""
        return _run_osascript(TITLE_SCRIPT)

    def _title_cg(self, pid: int) -> str:
        """First layer-0 on-screen window owned by ``pid`` with a name (needs Screen Recording)."""
        if not MAC_AVAILABLE or pid < 0:
            return ""
        options = Quartz.kCGWindowListOptionOnScreenOnly | Quartz.kCGWindowListExcludeDesktopElements
        windows = Quartz.CGWindowListCopyWindowInfo(options, Quartz.kCGNullWindowID) or []
        for window in windows:
            if window.get("kCGWindowOwnerPID") != pid:
                continue
            if window.get("kCGWindowLayer") != 0:
                continue
            name = window.get("kCGWindowName")
            if name:
                return str(name)
        return ""

    # ---- browser URL ------------------------------------------------------ #

    def _url_domain(self, bundle_id: str) -> str:
        """Hostname of the front tab for known browsers; "" otherwise or on failure.

        ``EXCLUDE_TITLES=1`` means app-name-only: the browser is never asked for its URL either."""
        if self.settings.exclude_titles:
            return ""
        expr = BROWSER_URL_SCRIPTS.get(bundle_id or "")
        if not expr:
            return ""
        now = time.monotonic()
        failed_at = self._url_failures.get(bundle_id)
        if failed_at is not None and now - failed_at < URL_FAILURE_TTL_S:
            return ""
        try:
            url = self._browser_url(bundle_id, expr)
        except Exception:
            self._url_failures[bundle_id] = now
            return ""
        self._url_failures.pop(bundle_id, None)
        return _domain_from_url(url)

    def _browser_url(self, bundle_id: str, expr: str) -> str:
        script = f'tell application id "{bundle_id}" to get {expr}'
        return _run_osascript(script)

    # ---- idle ------------------------------------------------------------- #

    def _idle_s(self) -> float:
        if not MAC_AVAILABLE:
            raise RuntimeError("Quartz unavailable")
        return float(
            Quartz.CGEventSourceSecondsSinceLastEventType(
                Quartz.kCGEventSourceStateHIDSystemState, Quartz.kCGAnyInputEventType
            )
        )

    # ---- screen ----------------------------------------------------------- #

    def _screen_hash(self) -> tuple[str, bool]:
        """(dhash hex, screen_available). Stores the PIL image on self._last_image."""
        self._last_image = None   # dropped first, so a capture that raises can never leave a stale frame behind
        if not MAC_AVAILABLE or not IMAGING_AVAILABLE or not _screen_capture_allowed():
            return "", False
        image = self._capture()
        if image is None:
            return "", False
        self._last_image = image
        row, _col = dhash.dhash_row_col(image, size=8)
        # Classic 64-bit dhash (row gradients) => 16 hex chars, matching
        # Settings.hamming_threshold's "bits of 64". format_hex(row, col) would be 128 bits.
        return f"{row:016x}", True

    def _capture(self):
        """Grab the whole screen into a PIL image, in memory only. None on failure."""
        img = Quartz.CGWindowListCreateImage(
            Quartz.CGRectInfinite,
            Quartz.kCGWindowListOptionOnScreenOnly,
            Quartz.kCGNullWindowID,
            Quartz.kCGWindowImageNominalResolution,
        )
        if img is None:
            return None
        w = Quartz.CGImageGetWidth(img)
        h = Quartz.CGImageGetHeight(img)
        bpr = Quartz.CGImageGetBytesPerRow(img)
        if not w or not h or not bpr:
            return None
        data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(img))
        buf = np.frombuffer(data, dtype=np.uint8)
        arr = buf[: h * bpr].reshape(h, bpr)[:, : w * 4].reshape(h, w, 4)
        rgb = np.ascontiguousarray(arr[..., [2, 1, 0]])  # BGRA -> RGB
        return Image.fromarray(rgb)

    # ---- blockers --------------------------------------------------------- #

    def _screen_locked(self) -> bool:
        if not MAC_AVAILABLE:
            return False
        session = Quartz.CGSessionCopyCurrentDictionary()
        if not session:
            return False
        return bool(session.get("CGSSessionScreenIsLocked", False))

    def _do_not_disturb(self) -> bool:
        """Best effort: Focus assertions file. Unreadable (the common case) => False."""
        try:
            with open(DND_ASSERTIONS_PATH, "r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            return False
        try:
            if isinstance(data, dict):
                data = data.get("data", [])
            records = data[0]["storeAssertionRecords"]
            return isinstance(records, list) and len(records) > 0
        except Exception:
            return False

    def _mic_in_use(self) -> bool:
        if not MAC_AVAILABLE or self._coreaudio_failed:
            return False
        if self._coreaudio is None:
            try:
                self._coreaudio = _CoreAudio()
            except Exception:
                self._coreaudio_failed = True
                return False
        return self._coreaudio.any_input_running()


# --------------------------------------------------------------------------- #
# FakeSensor
# --------------------------------------------------------------------------- #


class FakeSensor:
    """Scriptable stand-in for tests and demos.

    ``sample()`` returns ``frame`` as-is (timestamps are not rewritten). When a
    list of frames is given they are served in order; the last one then sticks.
    ``set(**fields)`` mutates the current frame.
    """

    def __init__(self, frames: list[ContextFrame] | None = None) -> None:
        queue = list(frames or [])
        if queue:
            self.frame = queue.pop(0)
        else:
            self.frame = ContextFrame(ts=time.time(), app="FakeApp", bundle_id="com.example.fake")
        self._queue = queue
        self.jpeg: Optional[bytes] = b"\xff\xd8fake"
        self.grab_calls = 0
        self.sample_calls = 0
        self.missing: list[str] = []
        self._blockers = Blockers()

    def set(self, **fields) -> ContextFrame:
        self.frame = replace(self.frame, **fields)
        return self.frame

    def set_blockers(self, **fields) -> Blockers:
        self._blockers = replace(self._blockers, **fields)
        return self._blockers

    def sample(self) -> ContextFrame:
        self.sample_calls += 1
        current = self.frame
        if self._queue:
            self.frame = self._queue.pop(0)
        return current

    def grab_jpeg(self) -> Optional[bytes]:
        self.grab_calls += 1
        return self.jpeg

    def blockers(self) -> Blockers:
        return self._blockers

    def missing_permissions(self) -> list[str]:
        return list(self.missing)
