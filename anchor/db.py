"""One SQLite file holds everything. No table ever stores a screenshot.

Location: ~/Library/Application Support/Anchor/anchor.db (mode 0600).
A corrupt file is renamed aside and recreated rather than crashing the app.
"""

from __future__ import annotations

import json
import os
import sqlite3
import threading
from pathlib import Path
from typing import Optional

from .models import Anchor, Policy, Register, Verdict

SCHEMA = """
CREATE TABLE IF NOT EXISTS anchors(
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    verbatim TEXT NOT NULL,
    clarified TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'en',
    status TEXT NOT NULL DEFAULT 'active',
    created_at REAL NOT NULL,
    ended_at REAL,
    detour_until REAL,
    detour_minutes INTEGER
);
CREATE TABLE IF NOT EXISTS policies(
    anchor_id INTEGER PRIMARY KEY,
    patience_seconds INTEGER, pause_idle_s INTEGER, pause_stable_s INTEGER,
    default_detour_minutes INTEGER, humor_ok INTEGER, register TEXT,
    expected_surfaces TEXT, tolerance_seconds INTEGER, task_kind TEXT, language TEXT
);
CREATE TABLE IF NOT EXISTS verdicts(
    anchor_id INTEGER, fingerprint TEXT, drift REAL, confidence REAL, reason TEXT,
    tolerated INTEGER, activity TEXT, dhash TEXT, ts REAL,
    PRIMARY KEY(anchor_id, fingerprint)
);
CREATE TABLE IF NOT EXISTS events(
    id INTEGER PRIMARY KEY AUTOINCREMENT, anchor_id INTEGER, state TEXT, detail TEXT, ts REAL
);
CREATE TABLE IF NOT EXISTS refinements(
    id INTEGER PRIMARY KEY AUTOINCREMENT, anchor_id INTEGER, fingerprint TEXT, correction TEXT, ts REAL
);
CREATE TABLE IF NOT EXISTS settings(key TEXT PRIMARY KEY, value TEXT);
"""


class Store:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self._lock = threading.RLock()
        self._conn: Optional[sqlite3.Connection] = None

    # ---------- lifecycle ----------
    def open(self) -> "Store":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.chmod(self.path.parent, 0o700)
        except OSError:
            pass
        try:
            self._conn = self._connect()
            verdict = self._conn.execute("PRAGMA integrity_check").fetchone()
            if verdict is None or str(verdict[0]).strip().lower() != "ok":
                # Subtler damage (a bad freelist, a broken index) is reported as rows, not raised.
                raise sqlite3.DatabaseError(f"integrity check failed: {verdict[0] if verdict else 'no result'}")
            self._conn.executescript(SCHEMA)
        except sqlite3.DatabaseError:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
            aside = self.path.with_name(self.path.name + ".corrupt")
            i = 1
            while aside.exists():
                aside = self.path.with_name(f"{self.path.name}.corrupt{i}")
                i += 1
            self.path.rename(aside)
            self._conn = self._connect()
            self._conn.executescript(SCHEMA)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass
        return self

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.path), check_same_thread=False, isolation_level=None)
        conn.row_factory = sqlite3.Row
        return conn

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self.open()
        assert self._conn is not None
        return self._conn

    # ---------- anchors ----------
    def _row_to_anchor(self, r: sqlite3.Row) -> Anchor:
        return Anchor(
            id=r["id"], verbatim=r["verbatim"], clarified=r["clarified"], language=r["language"],
            status=r["status"], created_at=r["created_at"], ended_at=r["ended_at"],
            detour_until=r["detour_until"], detour_minutes=r["detour_minutes"],
        )

    def create_anchor(self, verbatim: str, clarified: str, language: str, ts: float) -> Anchor:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO anchors(verbatim, clarified, language, status, created_at) VALUES(?,?,?,?,?)",
                (verbatim, clarified or verbatim, language, "active", ts),
            )
            return self.get_anchor(cur.lastrowid)

    def get_anchor(self, anchor_id: int) -> Optional[Anchor]:
        with self._lock:
            r = self.conn.execute("SELECT * FROM anchors WHERE id=?", (anchor_id,)).fetchone()
            return self._row_to_anchor(r) if r else None

    def active_anchor(self) -> Optional[Anchor]:
        with self._lock:
            r = self.conn.execute(
                "SELECT * FROM anchors WHERE status='active' ORDER BY id DESC LIMIT 1"
            ).fetchone()
            return self._row_to_anchor(r) if r else None

    def set_status(self, anchor_id: int, status: str, ended_at: Optional[float] = None) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE anchors SET status=?, ended_at=COALESCE(?, ended_at) WHERE id=?",
                (status, ended_at, anchor_id),
            )

    def set_detour(self, anchor_id: int, until: Optional[float], minutes: Optional[int] = None) -> None:
        """Store the ABSOLUTE wall-clock deadline (epoch seconds). None clears it."""
        with self._lock:
            self.conn.execute(
                "UPDATE anchors SET detour_until=?, detour_minutes=? WHERE id=?", (until, minutes, anchor_id)
            )

    # ---------- policies ----------
    def save_policy(self, anchor_id: int, p: Policy) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO policies VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    anchor_id, p.patience_seconds, p.pause_idle_s, p.pause_stable_s,
                    p.default_detour_minutes, 1 if p.humor_ok else 0, Register(p.register).value,
                    json.dumps(list(p.expected_surfaces)), p.tolerance_seconds, p.task_kind, p.language,
                ),
            )

    def load_policy(self, anchor_id: int) -> Optional[Policy]:
        with self._lock:
            r = self.conn.execute("SELECT * FROM policies WHERE anchor_id=?", (anchor_id,)).fetchone()
        if not r:
            return None
        return Policy(
            patience_seconds=r["patience_seconds"], pause_idle_s=r["pause_idle_s"],
            pause_stable_s=r["pause_stable_s"], default_detour_minutes=r["default_detour_minutes"],
            humor_ok=bool(r["humor_ok"]), register=Register(r["register"] or "playful"),
            expected_surfaces=json.loads(r["expected_surfaces"] or "[]"),
            tolerance_seconds=r["tolerance_seconds"] or 600, task_kind=r["task_kind"] or "general",
            language=r["language"] or "en",
        )

    # ---------- verdict cache ----------
    def get_verdict(self, anchor_id: int, fingerprint: str, now: float, ttl_s: int):
        """Returns (Verdict, dhash) or None if missing/expired."""
        with self._lock:
            r = self.conn.execute(
                "SELECT * FROM verdicts WHERE anchor_id=? AND fingerprint=?", (anchor_id, fingerprint)
            ).fetchone()
        if not r or (now - r["ts"]) > ttl_s:
            return None
        v = Verdict(drift=r["drift"], confidence=r["confidence"], reason=r["reason"] or "",
                    tolerated=bool(r["tolerated"]), source="cache", activity=r["activity"] or "")
        return v, (r["dhash"] or "")

    def put_verdict(self, anchor_id: int, fingerprint: str, v: Verdict, dhash: str, ts: float) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT OR REPLACE INTO verdicts VALUES(?,?,?,?,?,?,?,?,?)",
                (anchor_id, fingerprint, v.drift, v.confidence, v.reason, 1 if v.tolerated else 0,
                 v.activity, dhash, ts),
            )

    def invalidate_verdict(self, anchor_id: int, fingerprint: str) -> None:
        with self._lock:
            self.conn.execute("DELETE FROM verdicts WHERE anchor_id=? AND fingerprint=?", (anchor_id, fingerprint))

    # ---------- events ----------
    def log_event(self, anchor_id: Optional[int], state: str, detail: str, ts: float) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO events(anchor_id, state, detail, ts) VALUES(?,?,?,?)", (anchor_id, state, detail, ts)
            )

    def count_events(self, state: str, since: float, anchor_id: Optional[int] = None) -> int:
        with self._lock:
            if anchor_id is None:
                r = self.conn.execute("SELECT COUNT(*) FROM events WHERE state=? AND ts>=?", (state, since)).fetchone()
            else:
                r = self.conn.execute(
                    "SELECT COUNT(*) FROM events WHERE state=? AND ts>=? AND anchor_id=?", (state, since, anchor_id)
                ).fetchone()
            return int(r[0])

    def recent_events(self, limit: int = 50) -> list[dict]:
        with self._lock:
            rows = self.conn.execute("SELECT * FROM events ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # ---------- refinements (the "wrong call" loop) ----------
    def add_refinement(self, anchor_id: Optional[int], fingerprint: str, correction: str, ts: float) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO refinements(anchor_id, fingerprint, correction, ts) VALUES(?,?,?,?)",
                (anchor_id, fingerprint, correction, ts),
            )

    def refinements_since(self, since: float, anchor_id: Optional[int] = None) -> list[str]:
        with self._lock:
            if anchor_id is None:
                rows = self.conn.execute(
                    "SELECT correction FROM refinements WHERE ts>=? ORDER BY id", (since,)
                ).fetchall()
            else:
                rows = self.conn.execute(
                    "SELECT correction FROM refinements WHERE ts>=? AND anchor_id=? ORDER BY id", (since, anchor_id)
                ).fetchall()
        return [r[0] for r in rows]

    # ---------- settings (key/value) ----------
    def get_setting(self, key: str, default: Optional[str] = None) -> Optional[str]:
        with self._lock:
            r = self.conn.execute("SELECT value FROM settings WHERE key=?", (key,)).fetchone()
        return r[0] if r else default

    def set_setting(self, key: str, value: str) -> None:
        with self._lock:
            self.conn.execute("INSERT OR REPLACE INTO settings(key, value) VALUES(?,?)", (key, value))
