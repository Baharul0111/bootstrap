"""Drive a REAL desktop end-to-end test: Anchor runs headless with a text reply file, this script opens
real sites/apps, waits, replies as the user would, and checks the transcript at each step.

Usage (from the project root):
    ANCHOR_DEMO=1 ANCHOR_DB=/tmp/anchor-drive.db ANCHOR_REPLY_FILE=/tmp/anchor-replies.txt \
        .venv/bin/python -m anchor headless > /tmp/anchor-drive.log 2>&1 &
    .venv/bin/python tools/drive_desktop.py /tmp/anchor-drive.log /tmp/anchor-replies.txt

It never types into other apps; it only opens URLs/apps with `open` and brings them to the front.
"""

from __future__ import annotations

import subprocess
import sys
import time

LOG, REPLIES = sys.argv[1], sys.argv[2]
T0 = time.time()
results: list[tuple[str, bool, str]] = []


def log_lines() -> list[str]:
    try:
        with open(LOG, encoding="utf-8") as fh:
            return fh.read().splitlines()
    except FileNotFoundError:
        return []


def say(text: str) -> None:
    with open(REPLIES, "a", encoding="utf-8") as fh:
        fh.write(text + "\n")
    print(f"[{time.time()-T0:6.1f}s] >>> reply: {text}", flush=True)


def open_url(url: str) -> None:
    subprocess.run(["open", "-a", "Google Chrome", url], check=False)
    print(f"[{time.time()-T0:6.1f}s] opened {url}", flush=True)


def open_app(name: str) -> None:
    subprocess.run(["open", "-a", name], check=False)
    print(f"[{time.time()-T0:6.1f}s] front: {name}", flush=True)


def wait_for(pred, timeout: float, label: str) -> str | None:
    """Wait until a NEW transcript line satisfies pred. Returns the line or None."""
    seen = len(log_lines())
    end = time.time() + timeout
    while time.time() < end:
        lines = log_lines()
        for line in lines[seen:]:
            if pred(line):
                print(f"[{time.time()-T0:6.1f}s] <<< {line}", flush=True)
                return line
        seen = len(lines)
        time.sleep(0.5)
    print(f"[{time.time()-T0:6.1f}s] !!! timeout waiting for: {label}", flush=True)
    return None


def expect_quiet(seconds: float, label: str) -> bool:
    seen = len(log_lines())
    time.sleep(seconds)
    new = [l for l in log_lines()[seen:] if l.startswith("anchor:")]
    ok = not new
    print(f"[{time.time()-T0:6.1f}s] quiet check '{label}': {'OK' if ok else 'SPOKE: ' + ' | '.join(new)}", flush=True)
    return ok


def check(label: str, ok: bool, detail: str = "") -> None:
    results.append((label, ok, detail))
    print(f"[{time.time()-T0:6.1f}s] {'PASS' if ok else 'FAIL'}: {label} {detail}", flush=True)


is_roast = lambda l: l.startswith("anchor:") and ("main thing" in l or "मुख्य" in l)
is_pushback = lambda l: l.startswith("anchor:") and "Still true?" in l
is_howlong = lambda l: l.startswith("anchor:") and "How long" in l
is_confirm = lambda l: l.startswith("anchor:") and "come back at" in l
is_reminder = lambda l: l.startswith("anchor:") and "Time's up" in l
is_ack = lambda l: l.startswith("anchor:") and "stay out of your way" in l
is_intake = lambda l: l.startswith("anchor:") and "working on right now" in l

# ---------------------------------------------------------------- 1. intake
line = wait_for(is_intake, 60, "intake question")
check("asks for the goal", line is not None)
open_url("https://leetcode.com/problems/two-sum/")
time.sleep(2)
say("I am solving a DSA question on LeetCode")
check("adopts the goal", wait_for(is_ack, 40, "ack") is not None)

# ---------------------------------------------------------------- 2. on task: silence
check("silent while on LeetCode (45 s)", expect_quiet(45, "leetcode"))

# ---------------------------------------------------------------- 3. helper surface: silence within tolerance
open_url("https://stackoverflow.com/questions/509211/understanding-slicing")
check("silent on Stack Overflow helper page (40 s)", expect_quiet(40, "stackoverflow"))

# ---------------------------------------------------------------- 4. drift to a game → roast within ~30 s
open_url("https://poki.com/en/g/drive-mad")
line = wait_for(is_roast, 45, "roast about the game")
check("roasts the game within 45 s", line is not None, line or "")
if line:
    lower = line.lower()
    check("roast names the game", "drive" in lower or "poki" in lower or "game" in lower, line)
    check("roast is short (≤ 2 sentences + question)", line.count(". ") <= 3, line)

# ---------------------------------------------------------------- 5. break without amount → how long → 2 minutes
say("I just need a short break")
check("asks how long", wait_for(is_howlong, 20, "how long") is not None)
say("two minutes")
line = wait_for(is_confirm, 20, "confirmation")
check("confirms 2 minutes", line is not None and "2 minutes" in line, line or "")

# ---------------------------------------------------------------- 6. during the break: total silence even on the game
check("silent during the break (100 s)", expect_quiet(100, "break"))

# ---------------------------------------------------------------- 7. reminder, still playing → roast again
line = wait_for(is_reminder, 60, "reminder")
check("reminder quotes the goal", line is not None and "DSA question on LeetCode" in line, line or "")
say("okay")
line = wait_for(is_roast, 45, "second roast (still playing)")
check("roasts again while still playing", line is not None, line or "")

# ---------------------------------------------------------------- 8. evasive → one push-back → going back
say("it's replying back")
line = wait_for(is_pushback, 20, "push-back")
check("flat push-back quoting the sentence", line is not None and "I am solving a DSA question on LeetCode" in line, line or "")
say("I will go back to LeetCode")
open_url("https://leetcode.com/problems/two-sum/")
check("accepts going back (says Okay)", wait_for(lambda l: l.strip() == "anchor: Okay.", 15, "okay") is not None)
check("silent once back on LeetCode (40 s)", expect_quiet(40, "back on leetcode"))

# ---------------------------------------------------------------- 9. other real apps while on task-ish: Terminal (work) then Notes
open_app("Terminal")
check("silent in Terminal (35 s, ambiguous → careful)", expect_quiet(35, "terminal"))
open_url("https://www.youtube.com/watch?v=dQw4w9WgXcQ")
line = wait_for(is_roast, 45, "roast about YouTube")
check("roasts YouTube", line is not None, line or "")

# ---------------------------------------------------------------- 10. switch goal, then reverse drift
say("this is the new main thing, I'm watching YouTube now")
check("adopts the new goal", wait_for(is_ack, 30, "ack") is not None)
check("silent on YouTube under the new goal (40 s)", expect_quiet(40, "youtube as goal"))
open_url("https://leetcode.com/problems/two-sum/")
line = wait_for(is_roast, 45, "reverse roast (leetcode vs youtube)")
check("roasts LeetCode against the YouTube goal", line is not None, line or "")

# ---------------------------------------------------------------- 11. done → what's next → new goal in Hindi
say("it's done")
line = wait_for(lambda l: l.startswith("anchor:") and ("next" in l.lower()), 20, "what's next")
check("congratulates and asks what's next", line is not None, line or "")
say("मुझे अपना रिज़्यूमे लिखना है")
check("adopts a Hindi goal", wait_for(lambda l: l.startswith("anchor:") and "ठीक है" in l, 30, "hindi ack") is not None)
open_url("https://www.instagram.com/")
line = wait_for(lambda l: l.startswith("anchor:") and any("ऀ" <= ch <= "ॿ" for ch in l), 45, "Hindi roast")
check("roasts Instagram in Hindi", line is not None, line or "")
say("बीस मिनट")
line = wait_for(is_confirm if False else (lambda l: l.startswith("anchor:") and "20" in l), 20, "Hindi confirm")
check("confirms 20 minutes in Hindi", line is not None, line or "")

print("\n================ RESULTS ================")
for label, ok, detail in results:
    print(f"{'PASS' if ok else 'FAIL'}  {label}  {detail[:100]}")
print(f"{sum(ok for _, ok, _ in results)}/{len(results)} passed")
