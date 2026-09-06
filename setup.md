# Anchor — setup

`./run.sh doctor` checks each item below and prints one line per check with the number to fix.
Only #1, #4 and #7–#9 are required; the rest make Anchor sharper but it works without them.

Grant permissions to **the app that runs Anchor** — your terminal (Terminal.app, iTerm2, Warp,
the VS Code terminal) — because a plain Python process inherits its host's grants.

1. **OpenAI account and API key.** Create a key at <https://platform.openai.com/api-keys>, then in
   the project folder create a file named `.env` containing one line:
   `OPENAI_API_KEY=sk-...`
   The key is read at startup and never logged, printed or stored in the database. Without it
   Anchor cannot judge drift, speak, or listen, so the doctor treats this as required.

2. **Accessibility** (window titles).
   System Settings → Privacy & Security → Accessibility → `+` → add your terminal → toggle on.
   Without it Anchor sees only the front app's name, not the window title, and tells you which
   permission is missing. Judgements get coarser; nothing breaks.

3. **Screen Recording** (screen-change hash and rare escalation screenshots).
   System Settings → Privacy & Security → Screen & System Audio Recording → `+` → add your terminal.
   macOS may ask you to quit and reopen the terminal. Without it Anchor cannot tell that the
   screen changed while the title stayed the same, and it cannot look at the screen when the
   title is useless. Screenshots are never written to disk either way.

4. **Microphone.** macOS prompts the first time Anchor listens; if you dismissed it:
   System Settings → Privacy & Security → Microphone → enable your terminal.
   Without it Anchor cannot hear your goal or your answers, so this is required.

5. **Automation for your browser** (front-tab URL). The first time Anchor asks Safari or Chrome
   for the current URL, macOS shows "Terminal wants to control Safari" — click OK. To fix later:
   System Settings → Privacy & Security → Automation → your terminal → enable Safari / Chrome.
   Without it Anchor sees the browser's window title but not the site's domain.

6. **Full Disk Access** (optional; detects Focus / Do Not Disturb).
   System Settings → Privacy & Security → Full Disk Access → `+` → add your terminal.
   Without it Anchor cannot read the Focus state and may speak while a Focus mode is on.

Also checked by the doctor:

7. **`/usr/bin/say`** — the built-in macOS voice, used when the network or the OpenAI voice
   fails. Present on every Mac; if it is missing something is very wrong with the OS install.

8. **A writable database folder** — `~/Library/Application Support/Anchor/` by default. Set
   `ANCHOR_DB=/some/path/anchor.db` to put the single database file elsewhere.

9. **Python 3.11 or newer** — `./run.sh` creates a `.venv` with Python 3.12 (via `uv` if it is
   installed, otherwise `python3 -m venv`).
