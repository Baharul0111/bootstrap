"""Live judgement-quality eval for the drift judge (``anchor/llm.py::LLM.judge``).

Forty realistic desktop situations a hackathon judge might create, across five anchors, each with
an expected label and a one-line rationale. Every case is sent to the real judge once (text only,
no screenshot), and the script prints a table, a miss count and the token cost of the run.

Usage (from the project root, OPENAI_API_KEY in .env or the environment):
    .venv/bin/python tools/eval_judge.py                 # all 40 cases, 4 parallel calls
    .venv/bin/python tools/eval_judge.py --anchors A,D   # only some anchors
    .venv/bin/python tools/eval_judge.py --cases 4,14    # only some case numbers
    .venv/bin/python tools/eval_judge.py --runs 3        # repeat to separate real misses from noise
    .venv/bin/python tools/eval_judge.py --verbose       # show the judge's reason on every row

Expected labels and the MISS rules (they mirror the accumulator in ``anchor/drift.py``, where a
verdict with confidence < 0.6 holds the timer and drift >= 0.5 counts as off task):
    ON         the activity is the task itself     MISS if drift >= 0.5 and confidence >= 0.6
    OFF        clearly unrelated                   MISS if drift <  0.5 and confidence >= 0.6
    HELPER     an expected helper surface, on-topic MISS if tolerated is False and drift >= 0.5
    AMBIGUOUS  the evidence is thin either way     MISS if confidence >= 0.75 either way

Exit status is the number of misses (capped at 1) plus 1 more when any call failed outright.
The API key is loaded by ``anchor.config.load_env`` and is never printed.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from anchor.config import Settings, api_key_present, load_env  # noqa: E402
from anchor.llm import LLM  # noqa: E402
from anchor.models import Anchor, ContextFrame, Policy, Verdict  # noqa: E402

# --------------------------------------------------------------------------- #
# Prices (USD per million tokens). Defaults are the gpt-5.6-luna figures recorded in
# architecture.md ("Scaling plan & cost"); override with the environment when they change.
# --------------------------------------------------------------------------- #
USD_PER_M_INPUT = float(os.environ.get("ANCHOR_EVAL_USD_PER_M_INPUT", "0.20"))
USD_PER_M_CACHED = float(os.environ.get("ANCHOR_EVAL_USD_PER_M_CACHED", "0.02"))
USD_PER_M_OUTPUT = float(os.environ.get("ANCHOR_EVAL_USD_PER_M_OUTPUT", "1.20"))

LOW_CONFIDENCE = 0.6       # anchor/drift.py: below this the accumulator holds
OFF_TASK = 0.5             # anchor/models.py: Verdict.off_task
AMBIGUOUS_CAP = 0.75       # an unsure judge must not be this sure


# --------------------------------------------------------------------------- #
# Anchors. Policies mirror the reference policies in anchor/llm.py::_POLICY_SYSTEM
# (task kind plus expected surfaces), which is all the judge sees of the policy.
# --------------------------------------------------------------------------- #

CODING_SURFACES = ["chatgpt", "claude", "stackoverflow", "google", "github", "docs"]
WRITING_SURFACES = ["google", "docs", "chatgpt", "claude"]
WATCHING_SURFACES = ["youtube", "google"]

ANCHORS: dict[str, tuple[str, str, Policy]] = {
    "A": (
        "I am solving a DSA question on LeetCode",
        "en",
        Policy(task_kind="coding", expected_surfaces=CODING_SURFACES + ["leetcode"], patience_seconds=240),
    ),
    "B": (
        "fix the login bug in our Django backend",
        "en",
        Policy(task_kind="coding", expected_surfaces=CODING_SURFACES, patience_seconds=240),
    ),
    "C": (
        "write my resume for the internship",
        "en",
        Policy(task_kind="writing", expected_surfaces=WRITING_SURFACES, patience_seconds=150),
    ),
    "D": (
        "watch the operating systems lecture",
        "en",
        Policy(task_kind="watching", expected_surfaces=WATCHING_SURFACES, patience_seconds=420, pause_idle_s=None),
    ),
    "E": (
        "मुझे आज रात असाइनमेंट खत्म करना है",
        "hi",
        Policy(task_kind="writing", expected_surfaces=WRITING_SURFACES, patience_seconds=150, language="hi"),
    ),
}


@dataclass(frozen=True)
class Case:
    number: int
    anchor: str          # key into ANCHORS
    app: str             # localizedName as the sensor reports it
    title: str           # front window name (Chrome keeps its " - Google Chrome" suffix)
    domain: str          # hostname without "www.", "" for non-browsers
    idle_s: float
    expected: str        # ON | OFF | HELPER | AMBIGUOUS
    rationale: str


def _c(number: int, anchor: str, app: str, title: str, domain: str, idle_s: float, expected: str, why: str) -> Case:
    return Case(number, anchor, app, title, domain, idle_s, expected, why)


CHROME = "Google Chrome"
SFX = " - Google Chrome"

CASES: list[Case] = [
    # ---- A: solving a DSA question on LeetCode ------------------------------------------------
    _c(1, "A", CHROME, "Two Sum - LeetCode" + SFX, "leetcode.com", 2, "ON", "the problem page is the task itself"),
    _c(2, "A", "Code", "two_sum.py — leetcode-practice", "", 1, "ON", "editing the solution file for the problem"),
    _c(3, "A", "Code", "wedding_guest_list.csv — personal", "", 1, "OFF", "the editor is open on a personal spreadsheet"),
    _c(4, "A", CHROME, "ChatGPT - best pizza in Bangalore" + SFX, "chatgpt.com", 3, "OFF", "expected AI surface, clearly unrelated content"),
    _c(5, "A", CHROME, "python - How do I reverse a linked list in place? - Stack Overflow" + SFX, "stackoverflow.com", 5, "HELPER", "expected surface, on-topic DSA question"),
    _c(6, "A", CHROME, "Best places to visit in Goa in December - Google Search" + SFX, "google.com", 2, "OFF", "expected search engine, travel query"),
    _c(7, "A", CHROME, "Chess.com - Play Chess Online" + SFX, "chess.com", 1, "OFF", "game site"),
    _c(8, "A", CHROME, "", "", 4, "AMBIGUOUS", "bare browser with an empty title: no evidence either way"),
    # ---- B: fix the login bug in our Django backend -------------------------------------------
    _c(9, "B", "Code", "views.py — backend", "", 1, "ON", "editing the Django view code"),
    _c(10, "B", "Terminal", "python manage.py test accounts — zsh", "", 2, "ON", "running the accounts app tests"),
    _c(11, "B", "Postman", "POST /api/auth/login — Postman", "", 3, "ON", "exercising the login endpoint"),
    _c(12, "B", CHROME, "Claude - Django CSRF verification failed on login POST" + SFX, "claude.ai", 6, "HELPER", "expected AI surface, on-topic"),
    _c(13, "B", CHROME, "django/django: The Web framework for perfectionists with deadlines. - GitHub" + SFX, "github.com", 4, "HELPER", "expected surface, the framework's own repo"),
    _c(14, "B", CHROME, "Excel VBA - loop through all worksheets - Stack Overflow" + SFX, "stackoverflow.com", 5, "OFF", "expected surface, but the question has nothing to do with Django"),
    _c(15, "B", "Slack", "#weekend-plans - Acme - Slack", "", 2, "OFF", "social channel"),
    _c(16, "B", "Microsoft Teams", "Weekly standup | Microsoft Teams", "", 20, "AMBIGUOUS", "a call window says nothing about relevance"),
    # ---- C: write my resume for the internship ------------------------------------------------
    _c(17, "C", CHROME, "Resume - Aarav Mehta - Google Docs" + SFX, "docs.google.com", 2, "ON", "the resume document itself"),
    _c(18, "C", "Canva", "Modern Resume Template - Canva", "", 3, "ON", "building the resume in a design tool"),
    _c(19, "C", CHROME, "ChatGPT - rewrite my resume bullet points for a backend internship" + SFX, "chatgpt.com", 5, "HELPER", "expected AI surface, on-topic"),
    _c(20, "C", CHROME, "How to write a resume with no experience - Google Search" + SFX, "google.com", 2, "HELPER", "expected search engine, on-topic query"),
    _c(21, "C", "Figma", "College fest poster — Figma", "", 2, "OFF", "designing an unrelated poster"),
    _c(22, "C", "Spotify", "Spotify Premium", "", 1, "OFF", "music app in front instead of the resume"),
    _c(23, "C", "Xcode", "MyApp — ContentView.swift", "", 1, "OFF", "coding an iOS app instead of writing"),
    _c(24, "C", CHROME, "Inbox (23) - Outlook" + SFX, "outlook.live.com", 3, "AMBIGUOUS", "the inbox could hold the internship mail or anything else"),
    # ---- D: watch the operating systems lecture -----------------------------------------------
    _c(25, "D", CHROME, "Lecture 7: Virtual Memory and Paging - CS 162 - YouTube" + SFX, "youtube.com", 45, "ON", "the lecture itself"),
    _c(26, "D", CHROME, "MrBeast - I Survived 7 Days In The Desert - YouTube" + SFX, "youtube.com", 30, "OFF", "entertainment on the expected video site"),
    _c(27, "D", CHROME, "Arijit Singh - Tum Hi Ho (Official Video) - YouTube" + SFX, "youtube.com", 25, "OFF", "a music video on the expected video site"),
    _c(28, "D", CHROME, "Operating system - Wikipedia" + SFX, "en.wikipedia.org", 8, "ON", "same subject as the lecture"),
    _c(29, "D", CHROME, "List of Bollywood films of 2024 - Wikipedia" + SFX, "en.wikipedia.org", 6, "OFF", "Wikipedia, unrelated article"),
    _c(30, "D", "Preview", "Silberschatz - Operating System Concepts - ch09 Virtual Memory.pdf", "", 10, "ON", "the course textbook chapter"),
    _c(31, "D", CHROME, "New Tab" + SFX, "", 2, "AMBIGUOUS", "no content yet"),
    _c(32, "D", "zoom.us", "Zoom Meeting", "", 30, "AMBIGUOUS", "could be the live lecture or a chat with friends"),
    # ---- E: मुझे आज रात असाइनमेंट खत्म करना है (finish the assignment tonight) ---------------------
    _c(33, "E", "Microsoft Word", "DBMS Assignment 3 - Word", "", 2, "ON", "the assignment document"),
    _c(34, "E", CHROME, "ChatGPT - explain 3NF with an example" + SFX, "chatgpt.com", 4, "HELPER", "expected AI surface, coursework question"),
    _c(35, "E", CHROME, "Flipkart - Apple iPhone 16 Pro" + SFX, "flipkart.com", 3, "OFF", "shopping"),
    _c(36, "E", CHROME, "The Times of India: Latest News, Breaking News" + SFX, "timesofindia.indiatimes.com", 3, "OFF", "news"),
    _c(37, "E", CHROME, "Hotstar - Koffee with Karan Season 8" + SFX, "hotstar.com", 20, "OFF", "streaming a talk show"),
    _c(38, "E", "WhatsApp", "WhatsApp", "", 1, "OFF", "messaging app in front"),
    _c(39, "E", "Finder", "Downloads", "", 2, "AMBIGUOUS", "could be locating the assignment file"),
    _c(40, "E", CHROME, "[1706.03762] Attention Is All You Need - arXiv" + SFX, "arxiv.org", 8, "AMBIGUOUS", "a paper could be for the assignment; the subject is unknown"),
]


# --------------------------------------------------------------------------- #
# Metered client: LLM._parse discards ``response.usage``, so wrap the real client and
# add up tokens per call. Thread-safe because the eval fans calls out.
# --------------------------------------------------------------------------- #


class MeteredClient:
    """Quacks like ``OpenAI()`` for ``client.responses.parse`` and records usage."""

    def __init__(self, real: Any) -> None:
        self._real = real
        self.responses = self
        self.calls = 0
        self.input_tokens = 0
        self.cached_tokens = 0
        self.output_tokens = 0
        self.reasoning_tokens = 0
        self.models: dict[str, int] = {}
        self._lock = threading.Lock()

    def parse(self, **kwargs: Any) -> Any:
        response = self._real.responses.parse(**kwargs)
        usage = getattr(response, "usage", None)
        with self._lock:
            self.calls += 1
            model = str(kwargs.get("model", "?"))
            self.models[model] = self.models.get(model, 0) + 1
            if usage is not None:
                self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
                in_details = getattr(usage, "input_tokens_details", None)
                out_details = getattr(usage, "output_tokens_details", None)
                self.cached_tokens += int(getattr(in_details, "cached_tokens", 0) or 0)
                self.reasoning_tokens += int(getattr(out_details, "reasoning_tokens", 0) or 0)
        return response

    @property
    def usd(self) -> float:
        fresh = max(0, self.input_tokens - self.cached_tokens)
        return (
            fresh * USD_PER_M_INPUT + self.cached_tokens * USD_PER_M_CACHED + self.output_tokens * USD_PER_M_OUTPUT
        ) / 1_000_000


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #


def is_miss(expected: str, verdict: Verdict) -> bool:
    drift, conf = verdict.drift, verdict.confidence
    if expected == "ON":
        return drift >= OFF_TASK and conf >= LOW_CONFIDENCE
    if expected == "OFF":
        return drift < OFF_TASK and conf >= LOW_CONFIDENCE
    if expected == "AMBIGUOUS":
        return conf >= AMBIGUOUS_CAP
    if expected == "HELPER":
        return (not verdict.tolerated) and drift >= OFF_TASK
    raise ValueError(f"unknown expected label {expected!r}")


def make_anchor(key: str) -> Anchor:
    verbatim, language, _ = ANCHORS[key]
    return Anchor(id=ord(key), verbatim=verbatim, clarified=verbatim, language=language, status="active", created_at=0.0)


def make_frame(case: Case) -> ContextFrame:
    return ContextFrame(ts=time.time(), app=case.app, title=case.title, url_domain=case.domain, idle_s=case.idle_s)


def run_case(llm: LLM, case: Case) -> Optional[Verdict]:
    anchor = make_anchor(case.anchor)
    policy = ANCHORS[case.anchor][2]
    return llm.judge(anchor, policy, make_frame(case), refinements=[], image_jpeg=None)


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #


def _context(case: Case, width: int) -> str:
    title = case.title.replace(SFX, "") if case.title else "(empty title)"
    text = f"{case.app} | {title}"
    if case.domain:
        text += f" | {case.domain}"
    return text if len(text) <= width else text[: width - 1] + "…"


def _got(verdict: Optional[Verdict]) -> str:
    if verdict is None:
        return "call failed"
    return f"{verdict.drift:.2f}/{verdict.confidence:.2f}/{'T' if verdict.tolerated else 'F'}"


def print_table(results: dict[int, list[Optional[Verdict]]], cases: list[Case], runs: int, verbose: bool) -> tuple[int, int]:
    """Print the table; return (misses in the last run, cases that missed in any run)."""
    width = 66
    header = f"{'#':>2} {'A':1} {'context':<{width}} {'expected':<9} {'got drift/conf/tol':<20} verdict"
    if runs > 1:
        header += f"  miss runs (of {runs})"
    print(header)
    print("-" * len(header))
    last_misses = 0
    any_misses = 0
    failed = 0
    for case in cases:
        verdicts = results[case.number]
        last = verdicts[-1]
        miss_runs = sum(1 for v in verdicts if v is None or is_miss(case.expected, v))
        if last is None:
            failed += 1
        missed_last = last is None or is_miss(case.expected, last)
        last_misses += int(missed_last)
        any_misses += int(miss_runs > 0)
        verdict = "MISS" if missed_last else "OK"
        line = f"{case.number:>2} {case.anchor} {_context(case, width):<{width}} {case.expected:<9} {_got(last):<20} {verdict}"
        if runs > 1:
            line += f"  {miss_runs}/{runs}"
        print(line)
        if verbose or missed_last:
            print(f"     expected because: {case.rationale}")
            if last is not None:
                print(f"     judge: [{last.activity}] {last.reason}")
    print("-" * len(header))
    if failed:
        print(f"{failed} call(s) failed (judge returned None); each counts as a miss")
    return last_misses, any_misses


def print_usage(meter: MeteredClient, elapsed: float) -> None:
    fresh = max(0, meter.input_tokens - meter.cached_tokens)
    models = ", ".join(f"{m} x{n}" for m, n in sorted(meter.models.items()))
    print(
        f"usage: {meter.calls} calls ({models}) in {elapsed:.1f}s | input {meter.input_tokens} "
        f"(cached {meter.cached_tokens}, fresh {fresh}) | output {meter.output_tokens} "
        f"(reasoning {meter.reasoning_tokens})"
    )
    print(
        f"cost estimate: ${meter.usd:.4f} at ${USD_PER_M_INPUT}/M input, ${USD_PER_M_CACHED}/M cached, "
        f"${USD_PER_M_OUTPUT}/M output (override with ANCHOR_EVAL_USD_PER_M_*)"
    )


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #


def select_cases(anchors: Optional[str], numbers: Optional[str]) -> list[Case]:
    chosen = list(CASES)
    if anchors:
        keys = {k.strip().upper() for k in anchors.split(",") if k.strip()}
        unknown = keys - set(ANCHORS)
        if unknown:
            raise SystemExit(f"unknown anchor key(s): {', '.join(sorted(unknown))}; known: {', '.join(ANCHORS)}")
        chosen = [c for c in chosen if c.anchor in keys]
    if numbers:
        wanted = {int(n) for n in numbers.split(",") if n.strip()}
        chosen = [c for c in chosen if c.number in wanted]
    if not chosen:
        raise SystemExit("no cases selected")
    return chosen


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--anchors", help="comma-separated anchor keys, e.g. A,D (default: all)")
    parser.add_argument("--cases", help="comma-separated case numbers (default: all)")
    parser.add_argument("--runs", type=int, default=1, help="repeat every case this many times (default 1)")
    parser.add_argument("--workers", type=int, default=4, help="parallel judge calls (default 4)")
    parser.add_argument("--verbose", action="store_true", help="print rationale and judge reason on every row")
    parser.add_argument("--debug", action="store_true", help="show anchor.llm debug logs (failure reasons)")
    args = parser.parse_args(argv)

    if args.debug:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s %(message)s")
        logging.getLogger("httpx").setLevel(logging.WARNING)

    load_env()
    if not api_key_present():
        print("OPENAI_API_KEY is not set (put it in .env); nothing to evaluate.", file=sys.stderr)
        return 2

    from openai import OpenAI  # imported late so --help works without the SDK installed

    settings = Settings.from_env()
    meter = MeteredClient(OpenAI(timeout=30, max_retries=1))
    llm = LLM(settings, client=meter)
    cases = select_cases(args.anchors, args.cases)
    runs = max(1, args.runs)

    for key, (verbatim, language, policy) in ANCHORS.items():
        if any(c.anchor == key for c in cases):
            print(f"anchor {key} [{language}, {policy.task_kind}; surfaces: {', '.join(policy.expected_surfaces)}]: {verbatim}")
    print(f"judge model: {settings.judge_model} (fallback {settings.judge_fallback_model}); {len(cases)} cases x {runs} run(s)")
    print()

    results: dict[int, list[Optional[Verdict]]] = {c.number: [] for c in cases}
    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for _ in range(runs):
            for case, verdict in zip(cases, pool.map(lambda c: run_case(llm, c), cases)):
                results[case.number].append(verdict)
    elapsed = time.monotonic() - started

    last_misses, any_misses = print_table(results, cases, runs, args.verbose)
    if runs > 1:
        print(f"misses: {last_misses} of {len(cases)} in the last run; {any_misses} case(s) missed in at least one of {runs} runs")
    else:
        print(f"misses: {last_misses} of {len(cases)}")
    print_usage(meter, elapsed)
    failed = sum(1 for vs in results.values() if vs[-1] is None)
    return min(1, last_misses) + (1 if failed else 0)


if __name__ == "__main__":
    sys.exit(main())
