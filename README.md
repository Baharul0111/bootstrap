# Booty Goblin (Team Bootstrap)

Formerly "Anchor". The Python package is still called `anchor`; the product and the voice are **Booty Goblin**.

Booty Goblin is a quiet macOS menu-bar helper that holds you to the one sentence you said you were going to do.
It asks out loud what you are working on, watches the whole desktop once a second, and only speaks
when you have really drifted: one short roast about the situation, then a plain question.

## Videos

- **Demo:** [booty-goblin-demo-video.mp4](booty-goblin-demo-video.mp4) — the product in use: goal spoken aloud, a drift caught within seconds, the roast, the forced choice, the break and the reminder.
- **Process:** [booty-goblin-process-video.mp4](booty-goblin-process-video.mp4) — how it was built.

<video src="booty-goblin-demo-video.mp4" controls width="720"></video>

<video src="booty-goblin-process-video.mp4" controls width="720"></video>

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
| (default) | real use: every launch starts by asking for the goal (a previous goal or break is dropped); 12 s of patience for every task (a drift is called 10–15 s after it starts, 24 s at most while you keep typing), 20 s cooldown, 12 interruptions an hour at most |
| `ANCHOR_RESUME=1` | resume the previous goal and any pending break across a quit, sleep or lid-close (the original spec behaviour) |
| `ANCHOR_DEMO=1` | demo tempo: asks for the goal on every launch, 10 s of patience, no cooldown |
| `ANCHOR_PATIENCE_S=30` | any other patience in seconds |
| `ANCHOR_PACE=calm` | the architecture's original pace: minutes of patience derived from the task (coding 4 min, writing 2.5, lecture 7), 4 interruptions an hour, 45 s cooldown |
| `ANCHOR_INTENSITY=playful\|pointed\|savage` | Booty Globlin intensity (selected, never escalated). Default `pointed`, which is the `playful` register on the ladder; irritation only ever cools it |
| `ANCHOR_PERSONALITY=path` | use another personality JSON instead of `personality/booty_globlin.json` |
| `ANCHOR_SILENT=1` | start muted |
| `ANCHOR_REPLY_FILE=path` | replace the microphone with a text file: each line appended to it is one spoken reply (used by `tools/drive_desktop.py`, which opens real sites and apps on this Mac and checks every response end to end) |
| `EXCLUDE_TITLES=1` | send app names only, never window titles |
| `ANCHOR_DB=path` | put the database somewhere else |
| `ANCHOR_PROFANITY=1` | allow mild profanity in roasts (default off) |
| `ANCHOR_VOICE=name` | override the TTS voice (default `marin`, a female voice; `sage` and `coral` are the other two auditioned). The Indian-English accent comes from the delivery instructions, not the voice |
| `ANCHOR_ENV=path` | load a different `.env` file |

## Development

```sh
.venv/bin/python -m pytest              # all tests
.venv/bin/python -m pytest tests/test_app.py -q
.venv/bin/python -m anchor doctor       # same as ./run.sh doctor
```

The menu-bar shell is `anchor/app.py`; `anchor/engine.py` owns the worker thread where every
network and audio call happens — the main thread only ever reads a status snapshot.

## Personality: Booty Globlin (Team Bootstrap)

The voice is **Booty Globlin**: a tiny, excessively confident work companion. Everything about the character
lives in one editable file, `personality/booty_globlin.json`:

- `system_prompt` — the persona the roast-writing model is given (from `personality/booty_globlin_personality.md`).
- `library` — the ten creator-approved lines R01–R10 with the conditions under which each may be used. An approved
  line is used verbatim as the punchline after a short grounded premise ("One minute into Drive Mad on Poki.
  You're the minimum in MVP."), because the first roast of a goal must name the actual activity and the elapsed time.
  A line is never delivered twice in a session; lines whose facts are unknown (paid plans, imposter syndrome) stay unused.
- `intensity_by_register`, `word_caps`, `sentence_caps` — how sharp and how short. Intensity is chosen with
  `ANCHOR_INTENSITY`; it is never raised by the number of distractions, and an irritated reply cools it for the day.
- `comebacks`, `stop_words`, `jab_words` — "you're just a bot" gets one short comeback and then the question stands;
  "stop" / "shut up" / "chup" ends the exchange immediately with no farewell joke.
- `delivery_briefs` — deadpan / mock_respect / disbelief, sent to the speech model as performance direction, never spoken.
- `fixtures` — demo presets (intensity, teasing material, exclusions, known facts). Labelled as fixtures; they are not evidence.

The hard safety gates stay outside the personality file and cannot be edited away: never about the person, the
banned topics, no joke on serious goals or on low confidence, one push-back only, and the 45-second cooldown
between unsolicited roasts (0 in demo mode). Hindi goals get native Hindi lines; the English library is not translated.
