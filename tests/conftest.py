"""Shared fixtures: fake clock, temp SQLite store, and scripted fakes for the model, voice and sensor."""

from __future__ import annotations

import pytest

from anchor.clock import FakeClock
from anchor.config import Settings
from anchor.db import Store
from anchor.models import ContextFrame


@pytest.fixture
def clock() -> FakeClock:
    # 2027-01-15 14:32:00 local-ish; only relative arithmetic matters in tests
    return FakeClock(1_800_000_000.0)


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings()
    s.db_path = tmp_path / "anchor.db"
    s.silent = False
    return s


@pytest.fixture
def store(settings) -> Store:
    st = Store(settings.db_path).open()
    yield st
    st.close()


def frame(app="Google Chrome", title="crabs can swim? - Google Search", domain="google.com",
          idle=0.0, dhash="0000000000000000", ts=0.0, **kw) -> ContextFrame:
    return ContextFrame(ts=ts, app=app, bundle_id="", title=title, url_domain=domain, idle_s=idle, dhash=dhash, **kw)
