"""Free, local fallbacks. No network anywhere in this module.

Used directly for language detection and duration parsing, and as the safety net when
the model is unavailable: policy from keywords, reply intent from keywords, drift from
known distraction surfaces.
"""

from __future__ import annotations

import datetime as dt
import math
import re
import time
from typing import Optional

from .models import ContextFrame, Intent, Policy, PolicyDraft, Register, ReplyIntent, Sentiment, Verdict

DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_WORD = re.compile(r"[a-zA-Zऀ-ॿ][\w'ऀ-ॿ]*")

HINGLISH_WORDS = {
    "hai", "hain", "hoon", "hu", "karna", "karni", "kar", "karo", "raha", "rahi", "rahe", "mujhe", "main", "mai",
    "kaam", "abhi", "thoda", "thodi", "der", "baad", "nahi", "nahin", "haan", "khatam", "gaya", "gayi", "padhai",
    "padhna", "padh", "likhna", "likh", "banana", "bana", "dekhna", "dekh", "ghanta", "ghante", "baje", "tak",
    "aaj", "kal", "wala", "wali", "chahiye", "band", "chup", "mera", "meri", "mere", "yeh", "ye", "woh", "kya",
    "bhi", "aur", "liye", "ke", "ki", "ka", "ko", "se", "mein", "par", "ab", "phir", "bas", "accha", "theek",
    "ho", "hua", "hogaya", "kuch", "sab", "abhi", "pata", "bata", "matlab", "yaar",
    # Hindi number words and time words (Roman script)
    "ek", "teen", "char", "chaar", "paanch", "panch", "chhe", "saat", "aath", "nau", "das", "dus", "gyarah",
    "barah", "pandrah", "bees", "pachees", "tees", "chalis", "pachas", "aadha", "aadhe", "sade", "dedh", "dhai",
}

FILLERS = {
    "i", "need", "to", "want", "will", "am", "going", "gonna", "some", "do", "the", "a", "an", "my", "me", "im",
    "i'm", "just", "now", "tonight", "today", "on", "of", "for", "and", "is", "it", "this", "that", "have", "has",
    "be", "with", "at", "in", "up", "get", "got", "gotta", "should", "must", "then", "so", "really", "bit",
    "mujhe", "hai", "hain", "hoon", "karna", "karni", "kar", "ab", "aaj", "main", "mai", "mera", "meri", "mere",
    "ko", "ka", "ki", "ke", "se", "mein", "yeh", "ye", "woh", "kya", "bhi", "chahiye", "raha", "rahi", "hu",
    "है", "हूँ", "हूं", "करना", "करनी", "मुझे", "मैं", "अब", "आज", "को", "का", "की", "के", "से", "में", "यह", "ये",
    "चाहिए", "रहा", "रही", "है।",
}

VAGUE_CATEGORIES = {
    "study", "studying", "studies", "work", "working", "padhai", "padhna", "kaam", "reading", "read", "assignment",
    "homework", "project", "stuff", "things", "learn", "learning", "revise", "revision", "practice", "coding",
    "code", "writing", "write", "पढ़ाई", "काम", "पढ़ना", "लिखना",
}

SENSITIVE_ROMAN = [
    "doctor", "hospital", "diagnosis", "diagnosed", "cancer", "surgery", "therapy", "therapist", "medicine",
    "medication", "sick", "illness", "chemo", "depression", "anxiety", "panic", "funeral", "death", "died", "dying",
    "passed away", "grief", "grieving", "loss", "tax", "taxes", "loan", "debt", "emi", "mortgage", "bankruptcy",
    "bankrupt", "bills", "insurance claim", "rent overdue", "overdue", "court", "lawyer", "lawsuit", "visa",
    "layoff", "laid off", "fired", "divorce", "custody", "eviction", "medical", "clinic", "prescription",
    "mental health", "suicide", "self harm", "bimar", "aspataal", "karz", "karza", "ilaj", "maut", "dawai",
]
SENSITIVE_DEVANAGARI = [
    "बीमार", "अस्पताल", "डॉक्टर", "इलाज", "कर्ज़", "कर्ज", "टैक्स", "मौत", "अंतिम संस्कार", "दवाई", "दवा", "कैंसर",
    "सर्जरी", "तलाक", "अदालत", "वकील", "लोन", "क़र्ज़", "मृत्यु", "शोक",
]

TASK_KEYWORDS = {
    "coding": ["code", "coding", "bug", "debug", "implement", "program", "programming", "app", "api", "python",
               "javascript", "typescript", "java", "rust", "repo", "fix", "feature", "deploy", "script", "backend",
               "frontend", "compile", "refactor", "function", "endpoint", "database", "sql", "test", "tests",
               "build", "ship", "commit", "pull request", "unit test", "kotlin", "swift", "react"],
    "writing": ["write", "writing", "essay", "assignment", "report", "draft", "thesis", "blog", "email", "paper",
                "likhna", "likh", "notes", "article", "resume", "cover letter", "proposal", "documentation",
                "लिखना", "लिख", "निबंध", "असाइनमेंट", "रिपोर्ट"],
    "watching": ["watch", "watching", "lecture", "video", "course", "tutorial", "dekhna", "dekh", "recording",
                 "webinar", "देखना", "देख", "लेक्चर", "वीडियो"],
    "reading": ["read", "reading", "paper", "book", "chapter", "padhna", "padh", "padhai", "textbook", "article",
                "पढ़ना", "पढ़", "पढ़ाई", "किताब", "अध्याय"],
}
TASK_PATIENCE = {"coding": 240, "writing": 150, "watching": 420, "reading": 300, "general": 180}
TASK_SURFACES = {
    "coding": ["chatgpt", "claude", "stackoverflow", "google", "github", "docs", "localhost"],
    "writing": ["google", "docs", "chatgpt", "claude", "wikipedia", "scholar", "grammarly"],
    "watching": ["youtube", "coursera", "udemy", "nptel", "drive", "vimeo"],
    "reading": ["google", "scholar", "wikipedia", "pdf", "arxiv"],
    "general": ["google", "chatgpt", "claude"],
}

CLARIFY_QUESTION = {
    "en": "What exactly are you working on, and how will you know it's done?",
    "hi": "आप ठीक-ठीक किस पर काम कर रहे हैं, और आपको कैसे पता चलेगा कि यह पूरा हो गया?",
}

NUMBER_WORDS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30, "forty": 40,
    "fifty": 50, "sixty": 60, "ninety": 90,
    "ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5, "chhe": 6, "che": 6, "saat": 7,
    "aath": 8, "nau": 9, "das": 10, "dus": 10, "gyarah": 11, "barah": 12, "pandrah": 15, "bees": 20, "pachees": 25,
    "tees": 30, "chalis": 40, "pachas": 50, "saath": 60,
    "एक": 1, "दो": 2, "तीन": 3, "चार": 4, "पाँच": 5, "पांच": 5, "छह": 6, "छे": 6, "सात": 7, "आठ": 8, "नौ": 9, "दस": 10,
    "ग्यारह": 11, "बारह": 12, "पंद्रह": 15, "बीस": 20, "पच्चीस": 25, "तीस": 30, "चालीस": 40, "पचास": 50, "साठ": 60,
}
_NUM = r"(\d+(?:\.\d+)?|" + "|".join(re.escape(w) for w in NUMBER_WORDS) + r")"
_COMPOUND = r"(?:twenty|thirty|forty|fifty)[ -](?:one|two|three|four|five|six|seven|eight|nine)"
# A number token must stand alone: "yeah" must not read as "a" + "h" (an hour).
_L, _R = r"(?<![a-z0-9])", r"(?![a-z0-9])"
_NUMTOK = _L + r"(" + _COMPOUND + r"|" + _NUM[1:-1] + r")" + _R

VAGUE_DURATION = [
    "a bit", "a while", "some time", "sometime", "a few minutes", "few minutes", "a little", "a moment", "a sec",
    "later", "thoda der", "thodi der", "thodi si der", "kuch der", "baad mein", "baad me", "थोड़ी देर", "थोड़ा",
    "कुछ देर", "बाद में", "थोड़ी", "ek minute", "एक मिनट", "a minute", "one minute", "just a minute",
]

DISTRACTION_DOMAINS = [
    "youtube", "netflix", "primevideo", "hotstar", "instagram", "reddit", "twitter", "x.com", "tiktok",
    "facebook", "twitch", "steam", "9gag", "pinterest", "discord", "snapchat", "epicgames", "crazygames",
    "poki", "chess.com", "lichess", "imdb", "hulu", "disneyplus", "jiocinema", "sonyliv",
]
DISTRACTION_APPS = ["steam", "discord", "epic games", "telegram", "whatsapp", "tv", "music", "photos", "chess",
                    "netflix", "spotify"]
UNINFORMATIVE = {"", "new tab", "untitled", "window", "home", "start page", "blank", "new window", "loading",
                 "about:blank", "google chrome", "safari"}
APP_SUFFIX = re.compile(
    r"\s*[-—–|]\s*(google chrome|brave|microsoft edge|arc|mozilla firefox|safari|visual studio code|chromium|vivaldi)\s*$",
    re.IGNORECASE,
)


# --------------------------------------------------------------------------- language & vagueness
def _tokens(text: str) -> list[str]:
    return [t for t in _WORD.findall((text or "").lower())]


def detect_language(text: str) -> str:
    if DEVANAGARI.search(text or ""):
        return "hi"
    toks = _tokens(text)
    if not toks:
        return "en"
    hits = sum(1 for t in toks if t in HINGLISH_WORDS)
    if hits >= 2 or hits / len(toks) >= 0.4:
        return "hi"
    return "en"


def _content_words(text: str) -> list[str]:
    return [t for t in _tokens(text) if t not in FILLERS and len(t) > 1]


def is_vague(sentence: str) -> bool:
    words = _content_words(sentence)
    if len(words) <= 1:
        return True
    return all(w in VAGUE_CATEGORIES for w in words)


def is_sensitive(sentence: str) -> bool:
    low = (sentence or "").lower()
    for term in SENSITIVE_ROMAN:
        if re.search(r"\b" + re.escape(term) + r"\b", low):
            return True
    return any(term in low for term in SENSITIVE_DEVANAGARI)


# --------------------------------------------------------------------------- policy
def task_kind_of(sentence: str) -> str:
    low = (sentence or "").lower()
    toks = set(_tokens(sentence))
    best, best_hits = "general", 0
    for kind, words in TASK_KEYWORDS.items():
        hits = 0
        for w in words:
            if " " in w or DEVANAGARI.search(w):
                hits += 1 if w in low else 0
            else:
                hits += 1 if w in toks else 0
        if hits > best_hits:
            best, best_hits = kind, hits
    return best


def derive_policy_heuristic(
    sentence: str, default_register: Register = Register.PLAYFUL, default_detour_minutes: int = 15
) -> PolicyDraft:
    kind = task_kind_of(sentence)
    language = detect_language(sentence)
    policy = Policy(
        patience_seconds=TASK_PATIENCE[kind],
        pause_idle_s=None if kind == "watching" else 4,
        pause_stable_s=10,
        default_detour_minutes=default_detour_minutes,
        humor_ok=not is_sensitive(sentence),
        register=Register(default_register),
        expected_surfaces=list(TASK_SURFACES[kind]),
        tolerance_seconds=600,
        task_kind=kind,
        language=language,
    )
    vague = is_vague(sentence)
    return PolicyDraft(
        clarified_anchor=(sentence or "").strip(),
        language=language,
        policy=policy,
        needs_clarification=vague,
        clarifying_question=CLARIFY_QUESTION["hi" if language == "hi" else "en"] if vague else "",
    )


def merge_clarification(sentence: str, answer: str) -> str:
    sentence, answer = (sentence or "").strip().rstrip("."), (answer or "").strip().rstrip(".")
    if not answer:
        return sentence
    return f"{sentence} — {answer}"


# --------------------------------------------------------------------------- durations
def _num(token: str) -> Optional[float]:
    token = token.strip().lower()
    if re.fullmatch(r"\d+(?:\.\d+)?", token):
        return float(token)
    m = re.fullmatch(_COMPOUND, token)
    if m:
        tens, ones = re.split(r"[ -]", token)
        return NUMBER_WORDS[tens] + NUMBER_WORDS[ones]
    return float(NUMBER_WORDS[token]) if token in NUMBER_WORDS else None


def _minutes_until(hour: int, minute: int, now: float, explicit_period: bool) -> int:
    base = dt.datetime.fromtimestamp(now)
    candidates = []
    day = base.replace(second=0, microsecond=0)
    for offset in (0, 1):
        d = day + dt.timedelta(days=offset)
        candidates.append(d.replace(hour=hour % 24, minute=minute))
        if not explicit_period and hour <= 12:
            candidates.append(d.replace(hour=(hour + 12) % 24, minute=minute))
    future = sorted(c for c in candidates if c.timestamp() > now + 60)
    target = future[0]
    return max(1, int(math.ceil((target.timestamp() - now) / 60)))


def parse_duration_minutes(text: str, default_minutes: int, now: Optional[float] = None) -> Optional[int]:
    """Minutes implied by free-form text, ``default_minutes`` for vague phrasing, None when nothing is there."""
    low = (text or "").lower().strip()
    if not low:
        return None
    now = time.time() if now is None else now

    # --- explicit clock times: "till nine", "until 9:30 pm", "by 10", "9 baje tak", "sade nau tak", "नौ बजे तक"
    m = re.search(
        r"\b(?:till|until|by|upto|up to)\s+(noon|midnight|" + _NUM + r")" + _R + r"(?::(\d{2}))?\s*(am|pm|a\.m\.|p\.m\.)?\b",
        low,
    )
    if m:
        word, mins, period = m.group(1), m.group(3), m.group(4)
        if word == "noon":
            return _minutes_until(12, 0, now, True)
        if word == "midnight":
            return _minutes_until(0, 0, now, True)
        n = _num(word)
        if n is not None and 0 <= n <= 24:
            hour = int(n)
            if period and period.startswith("p") and hour < 12:
                hour += 12
            if period and period.startswith("a") and hour == 12:
                hour = 0
            return _minutes_until(hour, int(mins or 0), now, bool(period))
    m = re.search(
        r"(sade|साढ़े|साढे)?\s*" + _L + _NUM + _R + r"(?::(\d{2}))?\s*(?:(?:baje|बजे)\s*(?:tak|तक)?|(?:tak|तक)\b)", low
    )
    if m:
        n = _num(m.group(2))
        if n is not None and 0 <= n <= 24:
            minute = 30 if m.group(1) else int(m.group(3) or 0)
            return _minutes_until(int(n), minute, now, False)

    # --- hours
    if re.search(r"\b(half an hour|half hour|half-hour|aadha ghanta|aadhe ghante|adha ghanta)\b", low) or \
            "आधा घंटा" in low or "आधे घंटे" in low:
        return 30
    if re.search(r"\b(quarter of an hour|quarter hour|quarter-hour)\b", low):
        return 15
    if re.search(r"\b(dedh ghant[ae]|डेढ़ घंट[ाे])", low) or "डेढ़ घंटा" in low:
        return 90
    if re.search(r"\b(dhai ghant[ae])", low) or "ढाई घंटे" in low or "ढाई घंटा" in low:
        return 150
    m = re.search(_NUMTOK + r"\s*(?:and a half\s*)?(hours?|hrs?|ghant[ae]|घंट[ाे])", low) or \
        re.search(r"(?<![a-z0-9])(\d+(?:\.\d+)?)\s*h\b", low)
    if m:
        n = _num(m.group(1))
        if n is not None:
            extra = 30 if "and a half" in m.group(0) else 0
            return max(1, int(round(n * 60 + extra)))

    # --- "a minute" is a figure of speech, not sixty seconds
    for phrase in ("just a minute", "a minute", "one minute", "ek minute", "एक मिनट", "a sec", "one sec"):
        if phrase in low:
            return max(1, int(default_minutes))

    # --- minutes
    m = re.search(_NUMTOK + r"\s*(?:more\s+)?(minutes?|mins?|मिनट)", low) or \
        re.search(r"(?<![a-z0-9])(\d+)\s*m(?:in)?\b", low)
    if m:
        n = _num(m.group(1))
        if n is not None:
            return max(1, int(round(n)))

    # --- vague
    for phrase in VAGUE_DURATION:
        if phrase in low:
            return max(1, int(default_minutes))
    return None


# --------------------------------------------------------------------------- reply intent
DONE_WORDS = ["done", "finished", "already did", "completed", "submitted", "it's over", "its over", "ho gaya",
              "hogaya", "khatam", "kar liya", "हो गया", "पूरा", "खत्म", "ख़त्म", "finish kar", "complete"]
SWITCH_PHRASES = [
    "this is the new main thing", "this is now the main thing", "the new main thing", "new main thing",
    "this is the main thing", "the new thing", "new thing", "new goal", "switch", "actually i'm", "actually im",
    "actually, i'm", "now i'm working on", "now im working on", "now working on", "i'm now", "im now", "instead",
    "moved on", "ab main", "ab ye", "ab yeh", "अब मैं", "अब यह", "अब ये", "नया काम", "naya kaam", "main thing now",
]
DEFER_WORDS = ["break", "later", "remind", "few minutes", "minutes", "minute", "give me", "wait", "thoda", "thodi",
               "baad", "break lena", "बाद में", "ब्रेक", "रुको", "hold on", "pause", "some time", "a bit", "मिनट",
               "देर", "after"]
EVASIVE_FILLERS = ["hmm", "hm", "uh", "um", "whatever", "leave me", "idk", "kya", "not sure", "pata nahi",
                   "dunno", "meh", "eh", "huh", "what"]
IRRITATED_WORDS = ["stop", "shut up", "annoying", "leave me alone", "go away", "ugh", "seriously", "chup",
                   "band karo", "bakwas", "hatt", "dimag mat", "chhod do", "चुप", "बंद करो", "बकवास", "irritating",
                   "piss off", "get lost", "enough", "bas karo", "बस करो"]
AMUSED_WORDS = ["haha", "lol", "lmao", "😂", "hahaha", "hasi", "मज़ाक", "funny", "rofl", "😆", "🤣"]
_SWITCH_STRIP = re.compile(
    r"^(?:(?:ok|okay|actually|no|nah|well|hmm|umm|so|yeah|yes|fine|,|\.|:|-|\s)+)?"
    r"(?:this is (?:now )?the (?:new )?main thing|(?:the )?new (?:main )?thing|new goal|switch(?:ing)? to|"
    r"now i'?m (?:working on |doing )?|i'?m now (?:working on |doing )?|actually,? i'?m (?:working on |doing )?|"
    r"instead,? (?:i'?m )?|ab main|ab ye[h]?|अब मैं|अब यह|अब ये|main thing now)?[,:\-\s]*",
    re.IGNORECASE,
)


def _sentiment(low: str) -> Sentiment:
    if any(w in low for w in IRRITATED_WORDS):
        return Sentiment.IRRITATED
    if any(w in low for w in AMUSED_WORDS):
        return Sentiment.AMUSED
    return Sentiment.NEUTRAL


def _extract_new_anchor(text: str) -> str:
    stripped = text.strip()
    for _ in range(3):
        new = _SWITCH_STRIP.sub("", stripped, count=1).strip()
        if new == stripped:
            break
        stripped = new
    stripped = re.sub(r"^(?:i'?m |i am |main )", "", stripped, flags=re.IGNORECASE).strip() or stripped
    return stripped.strip(" ,.:-") or text.strip()


def classify_reply_keywords(text: str, default_minutes: int, now: Optional[float] = None) -> ReplyIntent:
    raw = (text or "").strip()
    low = raw.lower()
    language = detect_language(raw)
    sentiment = _sentiment(low)
    if not low:
        return ReplyIntent(Intent.EVASIVE, sentiment=sentiment, language=language)
    if any(re.search(r"(?<![a-z])" + re.escape(w) + r"(?![a-z])", low) if not DEVANAGARI.search(w) else w in low
           for w in DONE_WORDS):
        return ReplyIntent(Intent.DONE, sentiment=sentiment, language=language)
    if any(p in low for p in SWITCH_PHRASES):
        return ReplyIntent(Intent.SWITCH, new_anchor=_extract_new_anchor(raw), sentiment=sentiment, language=language)
    minutes = parse_duration_minutes(raw, default_minutes, now)
    if minutes is not None or any(w in low for w in DEFER_WORDS):
        return ReplyIntent(Intent.DEFER, minutes=minutes or max(1, int(default_minutes)), sentiment=sentiment,
                           language=language)
    return ReplyIntent(Intent.EVASIVE, sentiment=sentiment, language=language)


# --------------------------------------------------------------------------- judging
def clean_title(title: str) -> str:
    return APP_SUFFIX.sub("", (title or "").strip()).strip()


def is_uninformative_title(title: str, app: str) -> bool:
    t = clean_title(title).lower().strip(" -—|")
    if t in UNINFORMATIVE or len(t) < 3:
        return True
    return t == (app or "").lower().strip()


def _short(text: str, words: int = 6) -> str:
    parts = text.split()
    return " ".join(parts[:words]) + ("…" if len(parts) > words else "")


def describe_activity(frame: ContextFrame) -> str:
    title = clean_title(frame.title)
    domain = (frame.url_domain or "").lower()
    if domain:
        verb = "watching" if any(d in domain for d in ("youtube", "netflix", "hotstar", "primevideo", "twitch")) else \
            "scrolling" if any(d in domain for d in ("reddit", "instagram", "twitter", "x.com", "tiktok", "facebook")) else "reading"
        if title and not is_uninformative_title(title, frame.app):
            return f"{verb} '{_short(title)}' on {domain}"
        return f"{verb} {domain}"
    if title and not is_uninformative_title(title, frame.app):
        return f"'{_short(title)}' in {frame.app}"
    return f"using {frame.app or 'an app'}"


def judge_heuristic(anchor_text: str, policy: Policy, frame: ContextFrame) -> Verdict:
    domain = (frame.url_domain or "").lower()
    app = (frame.app or "").lower()
    title = clean_title(frame.title).lower()
    activity = describe_activity(frame)
    surfaces = [s.lower() for s in (policy.expected_surfaces or [])]
    haystack = f"{domain} {title} {app}"

    distraction = any(d in domain for d in DISTRACTION_DOMAINS) or app in DISTRACTION_APPS or \
        any(f"- {d}" in title or f"| {d}" in title or title.endswith(d) for d in ("youtube", "netflix", "reddit"))
    if distraction and not (policy.task_kind == "watching" and any(s in haystack for s in surfaces)):
        return Verdict(0.9, 0.7, "known distraction surface", source="heuristic", activity=activity)
    if surfaces and any(s in haystack for s in surfaces):
        return Verdict(0.2, 0.6, "expected surface for this task", tolerated=True, source="heuristic", activity=activity)
    anchor_words = [w for w in _content_words(anchor_text) if len(w) >= 4]
    if title and any(w in title for w in anchor_words):
        return Verdict(0.1, 0.7, "title matches the anchor", source="heuristic", activity=activity)
    if is_uninformative_title(frame.title, frame.app):
        return Verdict(0.5, 0.3, "title uninformative", source="heuristic", activity=activity)
    return Verdict(0.5, 0.4, "no local signal", source="heuristic", activity=activity)


def elapsed_phrase(minutes: int, language: str) -> str:
    if language == "hi":
        if minutes < 1:
            return "एक मिनट से कम"
        return f"{minutes} मिनट"
    if minutes < 1:
        return "under a minute"
    return "1 minute" if minutes == 1 else f"{minutes} minutes"
