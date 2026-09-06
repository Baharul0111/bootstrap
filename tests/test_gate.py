"""Component 3 — Change gate."""

from anchor.gate import ChangeGate, GateDecision, hamming, normalize_fingerprint, normalize_title
from anchor.models import ContextFrame

ZERO = "0" * 16
ONES = "f" * 16
BITS_12 = "0" * 13 + "fff"     # 12 bits set
BITS_13 = "0" * 12 + "1fff"    # 13 bits set


def frame(app="Google Chrome", title="Crabs - Wikipedia - Google Chrome", url_domain="en.wikipedia.org",
          dhash=ZERO, idle_s=0.0, ts=0.0):
    return ContextFrame(ts=ts, app=app, title=title, url_domain=url_domain, dhash=dhash, idle_s=idle_s)


# --- normalisation -----------------------------------------------------------

def test_normalize_title_lowercases_and_collapses_whitespace():
    assert normalize_title("  Hello   World  ", "X") == "hello world"


def test_normalize_title_strips_chrome_suffix():
    assert normalize_title("Crabs - Wikipedia - Google Chrome", "Google Chrome") == "crabs - wikipedia"


def test_normalize_title_strips_em_dash_firefox_suffix():
    assert normalize_title("Crabs — Mozilla Firefox", "Firefox") == "crabs"


def test_normalize_title_strips_vscode_suffix():
    assert normalize_title("gate.py - Anchor - Visual Studio Code", "Code") == "gate.py - anchor"


def test_normalize_title_strips_dynamic_app_name_suffix():
    assert normalize_title("Notes - Obsidian", "Obsidian") == "notes"


def test_normalize_title_only_strips_trailing_suffix():
    assert normalize_title("Google Chrome release notes", "Google Chrome") == "google chrome release notes"


def test_normalize_title_head_truncation():
    long = "a" * 100
    assert normalize_title(long, "X") == "a" * 60
    assert normalize_title(long, "X", head=10) == "a" * 10


def test_fingerprint_shape_and_case():
    assert normalize_fingerprint("Google Chrome", "Crabs - Google Chrome", "EN.Wikipedia.org") == \
        "google chrome|crabs|en.wikipedia.org"


def test_fingerprint_ignores_title_tail_beyond_head():
    base = "x" * 60
    assert normalize_fingerprint("A", base + "one", "d") == normalize_fingerprint("A", base + "two", "d")


# --- hamming -----------------------------------------------------------------

def test_hamming_identical_is_zero():
    assert hamming(ZERO, ZERO) == 0
    assert hamming("abcdef0123456789", "abcdef0123456789") == 0


def test_hamming_complementary_is_64():
    assert hamming(ZERO, ONES) == 64


def test_hamming_counts_bits():
    assert hamming(ZERO, BITS_12) == 12
    assert hamming(ZERO, BITS_13) == 13


def test_hamming_empty_or_malformed_is_zero():
    assert hamming("", ONES) == 0
    assert hamming(ONES, "") == 0
    assert hamming("", "") == 0
    assert hamming("zzzzzzzzzzzzzzzz", ONES) == 0
    assert hamming("ff", ONES) == 0  # length mismatch is treated as unknown


# --- gate table --------------------------------------------------------------

def test_first_frame_is_a_change():
    gate = ChangeGate()
    d = gate.update(frame())
    assert isinstance(d, GateDecision)
    assert d.changed is True
    assert d.fingerprint_changed is True
    assert d.pixels_diverged is False
    assert d.force_vision is False
    assert d.distance == 0
    assert gate.last_fingerprint == d.fingerprint
    assert gate.last_dhash == ZERO


def test_same_fingerprint_same_pixels_is_not_a_change():
    gate = ChangeGate()
    gate.update(frame())
    d = gate.update(frame())
    assert d.changed is False
    assert d.fingerprint_changed is False
    assert d.pixels_diverged is False
    assert d.force_vision is False


def test_changed_fingerprint_changed_pixels_is_ordinary_switch():
    gate = ChangeGate()
    gate.update(frame())
    d = gate.update(frame(app="Slack", title="general - Slack", url_domain="", dhash=ONES))
    assert d.changed is True
    assert d.fingerprint_changed is True
    assert d.pixels_diverged is True
    assert d.force_vision is False
    assert d.distance == 64


def test_same_fingerprint_diverged_pixels_forces_vision():
    gate = ChangeGate()
    gate.update(frame())
    d = gate.update(frame(dhash=ONES))
    assert d.changed is True
    assert d.fingerprint_changed is False
    assert d.pixels_diverged is True
    assert d.force_vision is True


def test_changed_fingerprint_same_pixels_is_cosmetic_change():
    gate = ChangeGate()
    gate.update(frame())
    d = gate.update(frame(title="(1) Crabs - Wikipedia - Google Chrome"))
    assert d.changed is True
    assert d.fingerprint_changed is True
    assert d.pixels_diverged is False
    assert d.force_vision is False


def test_force_vision_requires_same_fingerprint_and_divergence():
    gate = ChangeGate()
    gate.update(frame())
    assert gate.update(frame(app="Slack", dhash=ONES)).force_vision is False   # fp changed
    gate.reset()
    gate.update(frame())
    assert gate.update(frame(dhash=BITS_12)).force_vision is False            # not diverged
    assert gate.update(frame(dhash=ONES)).force_vision is True                # both


def test_threshold_boundary_12_is_not_diverged_13_is():
    gate = ChangeGate(hamming_threshold=12)
    gate.update(frame(dhash=ZERO))
    d12 = gate.update(frame(dhash=BITS_12))
    assert d12.distance == 12
    assert d12.pixels_diverged is False
    assert d12.changed is False

    gate.reset()
    gate.update(frame(dhash=ZERO))
    d13 = gate.update(frame(dhash=BITS_13))
    assert d13.distance == 13
    assert d13.pixels_diverged is True
    assert d13.changed is True


def test_app_suffix_does_not_change_fingerprint():
    gate = ChangeGate()
    gate.update(frame(title="Crabs - Wikipedia - Google Chrome"))
    d = gate.update(frame(title="Crabs - Wikipedia"))
    assert d.fingerprint_changed is False
    assert d.changed is False


def test_missing_dhash_counts_as_unchanged():
    gate = ChangeGate()
    gate.update(frame(dhash=ONES))
    d = gate.update(frame(dhash=""))
    assert d.pixels_diverged is False
    assert d.changed is False


def test_reset_makes_next_frame_a_first_frame():
    gate = ChangeGate()
    gate.update(frame())
    gate.update(frame())
    gate.reset()
    assert gate.last_fingerprint == ""
    assert gate.last_dhash == ""
    d = gate.update(frame())
    assert d.changed is True and d.fingerprint_changed is True and d.pixels_diverged is False
