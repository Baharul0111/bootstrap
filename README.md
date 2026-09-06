# Anchor

A quiet macOS menu-bar helper that holds you to the one sentence you said you were going to do.
It asks out loud what you are working on, watches the whole desktop once a second, and only speaks
when you have really drifted: one short roast about the situation, then a plain question.

## Run it

```sh
./run.sh
```

That is the whole install. The first run creates `.venv` and installs the dependencies; every run
after that starts the dot. You need a `.env` file with `OPENAI_API_KEY=...` first — see
[setup.md](setup.md) for that and for the macOS permissions.

Useful variants:

```sh
./run.sh doctor     # one line per environment check, exit 1 if something required is missing
./run.sh headless   # no menu bar; prints the transcript in the terminal until Ctrl-C
./run.sh version
```

## The dot

Anchor is one glyph in the menu bar and nothing else — no window, no Dock icon.

| Glyph | Meaning |
|-------|---------|
| ⚪ | idle, or asking for your goal |
| 🟢 | watching — you are on task |
| 🟠 | drifting — wander is adding up, nothing said yet |
| 🔴 | confronting — it has spoken and is waiting for your answer |
| 🔵 | detour — you asked for time; it will come back when the time is up |
| 🔇 | muted |
| ✓ | flashes briefly after "Wrong call" |

The menu has exactly three items:

- **Mute / Unmute** — stops all speech immediately (`ANCHOR_SILENT=1` starts muted).
- **Wrong call** — tells it the last judgement was wrong; it writes a refinement and is less likely
  to make the same mistake again today.
- **Quit** — stops cleanly. The active goal stays in the database and is picked up next launch.

## How the demo goes

1. Start it. It speaks: "What are you working on?" (the first time ever, it also says once that
   its voice is computer-generated.)
2. Say the goal in one sentence, in English or Hindi. If it is vague it asks one follow-up, then
   confirms the sentence and goes quiet. From the sentence it decides how patient to be.
3. Work. Switch to something unrelated — a video, a shopping tab, a chat — and stay there.
   Drift adds up in the background; nothing happens on a single glance.
4. When wander has added up and you pause, it speaks: a two-sentence roast about the situation
   (never about you), then the question: "Do you want more time, is this the new goal, or is the
   old one done?"
5. Answer out loud. Three branches:
   - **"Give me 10 minutes"** — a detour. The dot turns blue and it comes back at the deadline,
     even across a closed lid, with a plain reminder.
   - **"This is the new goal"** — it lets the old sentence go and anchors the new one.
   - **"That's done"** — it marks the goal finished, says one short line, and asks what is next.
   Dodge the question and it holds its ground exactly once, without a joke, then lets it go.

It will never interrupt more than four times an hour, never while your mic is in use, the screen
is locked, or a Focus mode is on.

## Privacy

- Everything lives in one SQLite file: `~/Library/Application Support/Anchor/anchor.db` (mode 0600).
  No account, no server of ours, no sync.
- Screenshots are never saved. When a screenshot is needed it exists in memory only and is discarded.
- The only outside party is OpenAI: the goal sentence, app names, trimmed window titles, the rare
  escalation image, and your voice audio. `EXCLUDE_TITLES=1` sends app names only.
- The API key is read from `.env` and never logged, printed, or stored.

## Environment switches

| Variable | Effect |
|----------|--------|
| `ANCHOR_DEMO=1` | demo tempo: asks for the goal on every launch, 10 s of patience, 2 s idle / 3 s still-screen pauses, so a drift is called 10–20 s after it starts |
| `ANCHOR_SILENT=1` | start muted |
| `EXCLUDE_TITLES=1` | send app names only, never window titles |
| `ANCHOR_DB=path` | put the database somewhere else |
| `ANCHOR_PROFANITY=1` | allow mild profanity in roasts (default off) |
| `ANCHOR_VOICE=name` | override the TTS voice (default `cedar`) |
| `ANCHOR_ENV=path` | load a different `.env` file |

## Development

```sh
.venv/bin/python -m pytest              # all tests
.venv/bin/python -m pytest tests/test_app.py -q
.venv/bin/python -m anchor doctor       # same as ./run.sh doctor
```

The menu-bar shell is `anchor/app.py`; `anchor/engine.py` owns the worker thread where every
network and audio call happens — the main thread only ever reads a status snapshot.
