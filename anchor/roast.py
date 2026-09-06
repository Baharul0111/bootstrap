"""The roast layer: the one place Anchor is allowed to be funny, and the place
that makes sure the joke is about the GAP (what was said vs. what is happening),
never about the person.

This module does no network I/O. It receives an ``llm`` object with a
``compose(...)`` method and only ever calls that; everything else here is pure
text: banned-term lists, validators, caps, native English/Hindi templates, the
``Composer`` (gates -> generate -> validate -> regenerate once -> fall back) and
the ``RegisterLadder`` (which can only ever cool, never warm).

Safety properties enforced here (architecture.md, Component 8; goal.md 13-18):
- Banned entirely: appearance, intelligence, discipline-as-character-flaw,
  relationships, family, money, weight, profanity (profanity opt-in via config).
- A roast must name the actual activity and (first two times) the elapsed time.
- Caps: confrontation 1 -> 2 sentences / 30 words, 2 -> 1 / 16, 3+ -> 1 / 8.
  Caps never escalate.
- Four gates suppress the joke entirely: serious anchor, low confidence,
  cooled-to-dry on a repeat, mute.
- The push-back and the detour reminder never carry a joke.
"""

from __future__ import annotations

import re
from typing import Any, Optional

from .config import Settings
from .models import Anchor, Composition, Observed, Policy, Register, cool_register

__all__ = [
    "BANNED",
    "find_banned",
    "caps_for",
    "count_sentences",
    "count_words",
    "is_specific",
    "validate_roast",
    "fallback_roast",
    "plain_statement",
    "hedged_statement",
    "choice_line",
    "pushback_line",
    "detour_confirm",
    "detour_reminder",
    "congrats",
    "whats_next",
    "intake_question",
    "disclosure",
    "permission_missing",
    "couldnt_hear",
    "switch_ack",
    "accept_line",
    "Composer",
    "RegisterLadder",
]

# ---------------------------------------------------------------------------
# 1. Banned terms (lowercase). Roman terms match on word boundaries,
#    case-insensitively; Devanagari terms match as a substring that starts at a
#    word start (Hindi inflects with suffixes, so the tail is left open).
# ---------------------------------------------------------------------------

BANNED: dict[str, list[str]] = {
    "APPEARANCE": [
        # English
        "ugly", "hideous", "unattractive", "gross", "disgusting", "fat", "skinny",
        "bald", "balding", "face", "faces", "looks", "hairline", "nose", "teeth",
        "wrinkles", "pimples", "acne", "your body",
        # Hinglish
        "badsurat", "badsoorat", "bhadda", "bhaddi", "bhadde", "ganja", "ganje",
        "shakal", "shakl", "chehra", "chehre",
        # Hindi
        "बदसूरत", "बदशक्ल", "भद्दा", "भद्दी", "भद्दे", "गंजा", "गंजे", "कुरूप", "शक्ल", "चेहर",
    ],
    "INTELLIGENCE": [
        # English
        "stupid", "stupidity", "dumb", "dumbass", "idiot", "idiots", "idiotic", "moron",
        "moronic", "brainless", "dimwit", "dim-witted", "imbecile", "fool", "foolish",
        "clueless", "retard", "retarded", "halfwit", "half-wit", "airhead", "numbskull",
        "nitwit", "dunce", "bonehead", "birdbrain", "no brain", "no brains", "pea brain",
        "pea-brain", "smooth brain", "smooth-brain", "iq",
        # Hinglish
        "bewakoof", "bewakuf", "bewkoof", "bevkoof", "bevakoof", "pagal", "paagal",
        "gadha", "gadhe", "gadhi", "ullu", "murkh", "moorkh", "buddhu", "budhu",
        "nasamajh", "anpadh", "jahil",
        # Hindi
        "बेवकूफ", "बेवकूफ़", "बेवक़ूफ़", "पागल", "गधा", "गधे", "गधी", "उल्लू", "मूर्ख", "बुद्धू",
        "नासमझ", "अनपढ़", "जाहिल", "बुद्धिहीन",
    ],
    "DISCIPLINE_AS_FLAW": [
        # English
        "lazy", "laziness", "lazybones", "useless", "pathetic", "hopeless", "loser",
        "losers", "failure", "failures", "worthless", "undisciplined", "weak-willed",
        "weak", "weakling", "spineless", "slacker", "procrastinator", "willpower",
        "self-control", "no discipline", "zero discipline", "irresponsible",
        "incompetent", "waste of space", "good-for-nothing", "good for nothing",
        "quitter", "flake", "flaky", "disgrace", "disgraceful", "shameful",
        "shame on you", "ashamed of yourself", "you never finish", "never finish anything",
        "can't finish anything", "can't focus", "cannot focus", "no focus",
        "attention span",
        # Hinglish
        "nikamma", "nikammi", "nikamme", "aalsi", "alsi", "nalayak", "naalayak",
        "nalaayak", "bekaar", "bekar", "kaamchor", "kamchor", "nakara", "nakaara",
        "nakaam", "bekaam", "sharam", "sharm", "besharam", "besharm", "aawara", "awara",
        # Hindi
        "निकम्मा", "निकम्मी", "निकम्मे", "आलसी", "नालायक", "बेकार", "कामचोर", "नाकारा", "नाकाम",
        "बेकाम", "शर्म", "बेशर्म", "आवारा", "कमज़ोर", "कमजोर",
    ],
    "RELATIONSHIPS": [
        # English
        "girlfriend", "girlfriends", "boyfriend", "boyfriends", "wife", "husband",
        "your ex", "an ex", "the ex", "ex-girlfriend", "ex-boyfriend", "ex-wife",
        "ex-husband", "single", "dating", "date night", "divorce", "divorced", "breakup",
        "break up", "broke up", "dumped", "crush", "tinder", "bumble", "love life",
        "lonely", "loneliness", "virgin", "married", "marriage", "spouse", "relationship",
        "relationships", "friend zone", "friendzone", "situationship",
        # Hinglish
        "shaadi", "shadi", "biwi", "bibi", "pati", "patni", "premika", "premi", "aashiq",
        "ashiq", "talaak", "talaq", "akela", "akeli", "akele", "kunwara", "kunwari",
        "kuwara", "rishta", "rishte",
        # Hindi
        "शादी", "बीवी", "पत्नी", "पति", "प्रेमिका", "प्रेमी", "आशिक़", "आशिक", "तलाक़", "तलाक",
        "अकेला", "अकेली", "अकेले", "कुंवारा", "कुंवारी", "कुँवारा", "ब्रेकअप", "गर्लफ्रेंड",
        "गर्लफ़्रेंड", "बॉयफ्रेंड", "बॉयफ़्रेंड", "रिश्ता", "रिश्ते", "रिश्तों",
    ],
    "FAMILY": [
        # English
        "mother", "mothers", "father", "fathers", "mom", "moms", "mum", "mums", "dad",
        "dads", "daddy", "mommy", "mummy", "papa", "parents", "parent", "family",
        "families", "brother", "brothers", "sister", "sisters", "son", "daughter",
        "uncle", "aunt", "aunty", "auntie", "grandma", "grandpa", "grandmother",
        "grandfather", "your kids", "your children", "sibling", "siblings", "orphan",
        "in-laws", "mother-in-law", "father-in-law",
        # Hinglish
        "maa", "baap", "mata", "pita", "pitaji", "behen", "behan", "bahan", "bhabhi",
        "beti", "dadi", "dada", "nani", "nana", "chacha", "chachi", "mama", "mausi",
        "bua", "khandaan", "khandan", "gharwale", "ghar wale", "ghar walon", "parivaar",
        "parivar",
        # Hindi
        "माँ", "मां", "माता", "बाप", "पिता", "पापा", "मम्मी", "परिवार", "बहन", "बहनें", "बेटा",
        "बेटी", "दादा", "दादी", "नाना", "नानी", "चाचा", "चाची", "मामा", "मौसी", "बुआ",
        "खानदान", "ख़ानदान", "घरवाले", "घरवालों", "अनाथ",
    ],
    "MONEY": [
        # English
        "poor", "broke", "rich", "salary", "salaries", "money", "income", "debt", "debts",
        "loan", "loans", "bank balance", "bankrupt", "afford", "cheap", "cheapskate",
        "wealthy", "wealth", "paycheck", "paycheque", "wages", "unemployed", "jobless",
        "rupees", "rupaye", "rupaiya", "dollars", "credit card", "emi",
        # Hinglish
        "paisa", "paise", "gareeb", "garib", "ghareeb", "kangaal", "kangal", "ameer",
        "berozgar", "berozgaar",
        # Hindi
        "पैसा", "पैसे", "पैसों", "गरीब", "ग़रीब", "कंगाल", "अमीर", "तनख्वाह", "तनख़्वाह",
        "बेरोज़गार", "बेरोजगार", "रुपय", "रुपए", "क़र्ज़", "कर्ज़", "कर्ज", "उधार", "दिवालिया",
        "सैलरी", "दौलत",
    ],
    "WEIGHT": [
        # English
        "weight", "obese", "obesity", "thin", "chubby", "overweight", "underweight",
        "plump", "belly", "beer belly", "kilos", "calories", "body fat", "waistline",
        "love handles",
        # Hinglish
        "mota", "moti", "mote", "patla", "patli", "patle", "tond", "dubla", "dubli",
        # Hindi
        "मोटा", "मोटी", "मोटे", "मोटापा", "पतला", "पतली", "पतले", "तोंद", "दुबला", "दुबली",
        "वज़न", "वजन", "कैलोरी",
    ],
    "PROFANITY": [
        # English
        "fuck", "fucking", "fucked", "fucker", "motherfucker", "shit", "shitty",
        "bullshit", "damn", "dammit", "goddamn", "ass", "asshole", "arse", "bitch",
        "bitches", "bastard", "crap", "dick", "dickhead", "piss", "pissed", "hell",
        "wtf", "prick", "cunt", "slut", "whore", "douche", "douchebag", "screw you",
        "jackass", "twat", "wanker", "bollocks", "sod off",
        # Hinglish
        "chutiya", "chutiye", "chutiyapa", "chutia", "bhosdike", "bhosdi", "bhosadi",
        "bhosad", "madarchod", "maderchod", "behenchod", "bhenchod", "benchod", "bc",
        "mc", "gandu", "gaandu", "gaand", "gand", "harami", "haraami", "haramkhor",
        "haramzada", "haramzaada", "haraamzaada", "kamina", "kamine", "kameena",
        "kameene", "kutta", "kutte", "kutti", "kutiya", "saala", "saale", "saali", "sala",
        "randi", "lauda", "laude", "lawda", "lund", "jhaat", "jhant", "jhaant", "tatti",
        "suar", "soor", "suwar", "bakchod", "bakchodi", "chodu", "chod", "chinaal",
        "chinal", "rakhail", "hijra", "ullu ka pattha", "ullu ke pathe", "bhadwa",
        "bhadwe", "dalla", "dalle",
        # Hindi
        "चूतिया", "चुतिया", "चूतिये", "भोसड़ी", "भोसड़ि", "मादरचोद", "बहनचोद", "भेनचोद", "बहेनचोद",
        "गांडू", "गाण्डू", "गांड", "गाण्ड", "हरामी", "हरामखोर", "हरामज़ादा", "हरामजादा", "हरामज़ादे",
        "कमीना", "कमीने", "कमीनी", "कुत्ता", "कुत्ते", "कुत्ती", "कुतिया", "साला", "साले", "साली",
        "रंडी", "लौड़ा", "लौड़े", "लंड", "झाट", "झांट", "टट्टी", "सुअर", "सूअर", "बकचोद", "बकचोदी",
        "चोदू", "चोद", "छिनाल", "रखैल", "भड़वा", "भड़वे", "दल्ला", "दल्ले", "हिजड़ा",
    ],
}

_DEVA_CLASS = "ऀ-ॿ"
_DEVA_RE = re.compile(f"[{_DEVA_CLASS}]")
_HAS_WORD_RE = re.compile(rf"[\w{_DEVA_CLASS}]")
_EDGE_PUNCT_RE = re.compile(rf"^[^\w{_DEVA_CLASS}]+|[^\w{_DEVA_CLASS}]+$")


def _has_devanagari(text: str) -> bool:
    return bool(_DEVA_RE.search(text or ""))


def _compile_term(term: str) -> re.Pattern[str]:
    if _has_devanagari(term):
        # Substring that begins at a word start; the tail is open for suffixes.
        return re.compile(rf"(?<![{_DEVA_CLASS}])" + re.escape(term))
    parts = [re.escape(p) for p in term.split()]
    return re.compile(r"\b" + r"\s+".join(parts) + r"\b", re.IGNORECASE)


_COMPILED: list[tuple[str, str, re.Pattern[str]]] = [
    (category, term, _compile_term(term))
    for category, terms in BANNED.items()
    for term in terms
]


def find_banned(text: str, profanity_ok: bool = False) -> list[tuple[str, str]]:
    """Return ``[(category, term), ...]`` for every banned term found in ``text``.

    Roman terms: word-boundary, case-insensitive. Devanagari terms: substring
    anchored at a word start. PROFANITY is skipped when ``profanity_ok``.
    """
    if not text:
        return []
    hits: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for category, term, pattern in _COMPILED:
        if profanity_ok and category == "PROFANITY":
            continue
        if pattern.search(text):
            key = (category, term)
            if key not in seen:
                seen.add(key)
                hits.append(key)
    return hits


# ---------------------------------------------------------------------------
# 2. Caps, counting, specificity, validation
# ---------------------------------------------------------------------------

def caps_for(repeat_index: int) -> tuple[int, int]:
    """(max_sentences, max_words) for the n-th confrontation on the same anchor.

    0 -> (2, 30); 1 -> (1, 16); 2+ -> (1, 8). Monotonically non-increasing:
    the joke only ever gets shorter, never sharper or longer.
    """
    idx = max(0, int(repeat_index or 0))
    if idx == 0:
        return (2, 30)
    if idx == 1:
        return (1, 16)
    return (1, 8)


_SENTENCE_SPLIT_RE = re.compile(r"[.!?।]+[\"'”’)\]]*(?:\s+|$)")


def count_sentences(text: str) -> int:
    """Sentences end at '.', '!', '?' or the Devanagari danda '।' followed by
    whitespace or the end of the text (so '2.5' and 'e.g.' do not split)."""
    if not text or not text.strip():
        return 0
    parts = _SENTENCE_SPLIT_RE.split(text.strip())
    return sum(1 for p in parts if _HAS_WORD_RE.search(p))


def count_words(text: str) -> int:
    """Whitespace-separated tokens that contain at least one letter or digit
    (a lone dash or quote mark is not a word)."""
    if not text:
        return 0
    return sum(1 for tok in text.split() if _HAS_WORD_RE.search(tok))


_STOPWORDS = {
    "with", "from", "that", "this", "your", "about", "then", "than", "into", "onto",
    "over", "some", "what", "when", "where", "which", "while", "have", "been", "being",
    "were", "just", "like", "there", "their", "they", "them", "very", "much", "more",
    "again", "still", "something", "someone", "thing", "things", "instead",
}

_TIME_WORDS_ROMAN = (
    "minute", "minutes", "min", "mins", "hour", "hours", "hr", "hrs", "second", "seconds",
    "sec", "secs", "ghanta", "ghante", "ghanton", "minit",
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen",
    "eighteen", "nineteen", "twenty", "thirty", "forty", "fifty", "sixty", "ninety",
    "hundred", "half", "quarter", "dozen", "couple",
    "ek", "paanch", "panch", "pandrah", "bees", "tees", "chalis", "pachas", "aadha",
    "aadhe", "aadhi",
)
_TIME_RE_ROMAN = re.compile(
    r"\d|\b(?:" + "|".join(re.escape(w) for w in _TIME_WORDS_ROMAN) + r")\b", re.IGNORECASE
)
_TIME_WORDS_DEVA = (
    "मिनट", "घंटा", "घंटे", "घण्टा", "घण्टे", "सेकंड", "सेकण्ड", "पल",
    "एक", "दो", "तीन", "चार", "पाँच", "पांच", "छह", "छः", "सात", "आठ", "नौ", "दस", "ग्यारह",
    "बारह", "पंद्रह", "पन्द्रह", "बीस", "तीस", "चालीस", "पचास", "साठ", "आधा", "आधे", "आधी",
    "सवा", "डेढ़", "ढाई",
)


def _tokens(text: str) -> list[str]:
    """Whitespace tokens with leading/trailing punctuation trimmed (Devanagari safe)."""
    out = []
    for raw in (text or "").split():
        tok = _EDGE_PUNCT_RE.sub("", raw)
        if tok:
            out.append(tok)
    return out


def _content_words(observed: Observed) -> tuple[list[str], list[str]]:
    """(roman_words, devanagari_tokens) worth naming from the observation."""
    sources = [observed.activity, observed.app, observed.url_domain, observed.title]
    roman: list[str] = []
    deva: list[str] = []
    short_roman: list[str] = []
    for src in sources:
        if not src:
            continue
        # url_domain "youtube.com" -> "youtube", "com"
        for tok in _tokens(re.sub(r"[./_-]+", " ", src)):
            if _has_devanagari(tok):
                if len(tok) >= 3:
                    deva.append(tok)
            else:
                low = tok.lower()
                if len(low) >= 4 and low not in _STOPWORDS:
                    roman.append(low)
                elif len(low) >= 2 and low not in _STOPWORDS:
                    short_roman.append(low)
    if not roman and not deva:
        roman = short_roman
    return roman, deva


def _mentions_activity(text: str, observed: Observed) -> bool:
    roman, deva = _content_words(observed)
    if not roman and not deva:
        return False
    low = text.lower()
    if any(w in low for w in roman):
        return True
    return any(t in text for t in deva)


def _mentions_time(text: str) -> bool:
    if _TIME_RE_ROMAN.search(text):
        return True
    return any(w in text for w in _TIME_WORDS_DEVA)


def is_specific(text: str, observed: Observed, repeat_index: int) -> bool:
    """True when the text names the observed activity (a content word of 4+
    chars from activity/app/url_domain/title, or a 3+ char Devanagari token)
    and, for the first two confrontations, refers to elapsed time (a digit, a
    minute/hour/second word, or an English/Hindi number word)."""
    if not text:
        return False
    if not _mentions_activity(text, observed):
        return False
    if max(0, int(repeat_index or 0)) <= 1 and not _mentions_time(text):
        return False
    return True


def validate_roast(
    text: str, observed: Observed, repeat_index: int, profanity_ok: bool = False
) -> tuple[bool, str]:
    """(ok, reason). Checks, in order: non-empty, no banned terms, within the
    caps for this repeat index, specific. All failures are listed in the reason."""
    if not text or not text.strip():
        return False, "empty"
    problems: list[str] = []
    hits = find_banned(text, profanity_ok=profanity_ok)
    if hits:
        problems.append(
            "banned terms: " + ", ".join(f"{cat} '{term}'" for cat, term in hits)
        )
    max_sentences, max_words = caps_for(repeat_index)
    n_sent = count_sentences(text)
    n_words = count_words(text)
    if n_sent > max_sentences:
        problems.append(f"too many sentences: {n_sent} > {max_sentences}")
    if n_words > max_words:
        problems.append(f"too many words: {n_words} > {max_words}")
    if not is_specific(text, observed, repeat_index):
        need = "the activity" if max(0, int(repeat_index or 0)) > 1 else "the activity and the elapsed time"
        problems.append(f"not specific: must name {need}")
    if problems:
        return False, "; ".join(problems)
    return True, "ok"


# ---------------------------------------------------------------------------
# 3. Templates. English and Hindi are written natively, not translated.
# ---------------------------------------------------------------------------

def _lang(language: Optional[str]) -> str:
    return "hi" if (language or "").strip().lower().startswith("hi") else "en"


def _clip_words(text: str, limit: int, ellipsis: str = "") -> str:
    toks = (text or "").split()
    if len(toks) <= limit:
        return " ".join(toks)
    return " ".join(toks[:limit]) + ellipsis


def _strip_end_punct(text: str) -> str:
    return (text or "").strip().rstrip(".!?।,;: ")


def _anchor_text(anchor: Anchor) -> str:
    """The phrase used after "you said you'd ..."; clarified first, else verbatim."""
    return _strip_end_punct(getattr(anchor, "clarified", "") or getattr(anchor, "verbatim", "") or "")


def _anchor_quote(anchor: Anchor, profanity_ok: bool) -> Optional[str]:
    """A <= 6 word quote of the anchor for the fallback roast, or None when the
    quote itself would carry a banned term (the joke must not repeat it)."""
    raw = _strip_end_punct(getattr(anchor, "verbatim", "") or getattr(anchor, "clarified", "") or "")
    if not raw:
        return None
    snippet = _clip_words(raw, 6, "…")
    if find_banned(snippet, profanity_ok=profanity_ok):
        return None
    return snippet


def _subject(observed: Observed, profanity_ok: bool, limit: int) -> str:
    """What to name: the activity, or the first of app/url_domain/title that is
    free of banned terms when the activity phrase itself is not."""
    candidates = [observed.activity, observed.app, observed.url_domain, observed.title]
    for cand in candidates:
        cand = (cand or "").strip()
        if cand and not find_banned(cand, profanity_ok=profanity_ok):
            return _clip_words(_strip_end_punct(cand), limit)
    fallback = (observed.activity or "").strip()
    return _clip_words(_strip_end_punct(fallback), limit)


def _minutes(observed: Observed) -> int:
    try:
        return max(0, int(observed.minutes_off_task or 0))
    except (TypeError, ValueError):
        return 0


def fallback_roast(
    anchor: Anchor, observed: Observed, repeat_index: int, language: str, profanity_ok: bool = False
) -> str:
    """Templated roast used when the model's line fails twice. Names the activity
    and the minutes, stays inside caps_for(repeat_index), and avoids banned terms
    (the anchor quote is dropped if it contains one)."""
    lang = _lang(language)
    idx = max(0, int(repeat_index or 0))
    m = _minutes(observed)
    if idx == 0:
        act = _subject(observed, profanity_ok, 9 if lang == "en" else 8)
        quote = _anchor_quote(anchor, profanity_ok)
        if lang == "hi":
            if quote:
                return f"{m} मिनट पहले आप '{quote}' कहने वाले इंसान थे। अब {act} चल रहा है — ज़बरदस्त मोड़।"
            return f"{m} मिनट पहले आपने कुछ और कहा था। अब {act} चल रहा है — ज़बरदस्त मोड़।"
        if quote:
            return f"{m} minutes ago you were a person who said '{quote}'. Now it's {act} — bold pivot."
        return f"{m} minutes ago you said one thing. Now it's {act} — bold pivot."
    if idx == 1:
        act = _subject(observed, profanity_ok, 9 if lang == "en" else 8)
        if lang == "hi":
            return f"अब भी {act}, {m} मिनट हो गए।"
        return f"Still {act}, {m} minutes in."
    act = _subject(observed, profanity_ok, 6 if lang == "en" else 5)
    if lang == "hi":
        return f"फिर से {act}।"
    return f"{act[:1].upper()}{act[1:]}, again."


def plain_statement(anchor: Anchor, observed: Observed, language: str) -> str:
    """No joke. States what it sees."""
    lang = _lang(language)
    m = _minutes(observed)
    act = _strip_end_punct(observed.activity or "")
    goal = _anchor_text(anchor)
    if lang == "hi":
        return f"आपने कहा था: '{goal}'। पिछले {m} मिनट से {act} चल रहा है।"
    return f"You said you'd {goal}. For the last {m} minutes it's been {act}."


def hedged_statement(anchor: Anchor, observed: Observed, language: str) -> str:
    """Careful, not accusing; used when confidence is low."""
    lang = _lang(language)
    m = _minutes(observed)
    act = _strip_end_punct(observed.activity or "")
    if lang == "hi":
        return f"शायद यह काम का ही हिस्सा हो, पर लगभग {m} मिनट से {act} दिख रहा है।"
    return f"This might be part of it, but it seems to be {act} for about {m} minutes."


def choice_line(language: str) -> str:
    if _lang(language) == "hi":
        return "एक छोटा सवाल: यह थोड़ी देर का ब्रेक है, या अब यही मुख्य काम है?"
    return "Quick one: is this a short break, or is this the new main thing?"


def pushback_line(verbatim: str, language: str) -> str:
    """The one push-back. The person's own sentence, unchanged, then the question.
    Never carries a joke."""
    v = (verbatim or "").strip()
    if _lang(language) == "hi":
        end = "" if v and v[-1] in ".!?।" else "।"
        return f"आपने कहा था: {v}{end} क्या यह अब भी सच है?"
    end = "" if v and v[-1] in ".!?।" else "."
    return f"You said: {v}{end} Still true?"


def detour_confirm(minutes: int, back_at: str, language: str) -> str:
    if _lang(language) == "hi":
        return f"ठीक है, {minutes} मिनट। {back_at} बजे फिर मिलते हैं।"
    return f"Okay, {minutes} minutes. I'll come back at {back_at}."


def detour_reminder(verbatim: str, late_minutes: Optional[int], language: str) -> str:
    """Plain, friendly pull-back after a detour. No joke. Acknowledges lateness."""
    v = (verbatim or "").strip()
    late = late_minutes if isinstance(late_minutes, int) and late_minutes > 0 else 0
    if _lang(language) == "hi":
        end = "" if v and v[-1] in ".!?।" else "।"
        prefix = f"माफ़ कीजिए, इसमें मुझे {late} मिनट की देर हो गई। " if late else ""
        return f"{prefix}समय पूरा हुआ। आपने कहा था: {v}{end} वापस उसी पर चलें?"
    end = "" if v and v[-1] in ".!?।" else "."
    prefix = f"Sorry, I'm {late} minutes late with this. " if late else ""
    return f"{prefix}Time's up. You said: {v}{end} Heading back to it?"


def congrats(language: str) -> str:
    if _lang(language) == "hi":
        return "बढ़िया, वह काम पूरा हुआ।"
    return "Nice, that one's done."


def whats_next(language: str) -> str:
    if _lang(language) == "hi":
        return "अब आगे क्या?"
    return "What's next?"


def intake_question(language: str) -> str:
    if _lang(language) == "hi":
        return "आप अभी किस काम पर हैं? एक वाक्य काफ़ी है।"
    return "What are you working on right now? One sentence is enough."


def disclosure(language: str) -> str:
    if _lang(language) == "hi":
        return "एक छोटी-सी बात: मेरी आवाज़ कंप्यूटर से बनी है।"
    return "Quick note: my voice is computer-generated."


def permission_missing(name: str, language: str) -> str:
    if _lang(language) == "hi":
        return f"मुझे अभी {name} की अनुमति नहीं मिली है। कृपया System Settings में Privacy & Security के अंदर इसे चालू करें।"
    return f"I can't see {name} yet. Please allow it in System Settings, under Privacy & Security."


def couldnt_hear(language: str) -> str:
    if _lang(language) == "hi":
        return "माफ़ कीजिए, मुझे सुनाई नहीं दिया।"
    return "Sorry, I didn't catch that."


def switch_ack(new_anchor: str, language: str) -> str:
    goal = _strip_end_punct(new_anchor or "")
    if _lang(language) == "hi":
        return f"ठीक है। नया लक्ष्य: {goal}।"
    return f"Got it. New anchor: {goal}."


def accept_line(language: str) -> str:
    if _lang(language) == "hi":
        return "ठीक है।"
    return "Okay."


# ---------------------------------------------------------------------------
# 4. Composer: gates -> generate -> validate -> regenerate once -> fall back
# ---------------------------------------------------------------------------

_MIN_CONFIDENCE = 0.75
_CHOICE_MAX_WORDS = 20


class Composer:
    """Turns an anchor + observation into what gets spoken. Never raises."""

    def __init__(self, llm: Any, settings: Settings):
        self.llm = llm
        self.settings = settings

    @property
    def _profanity_ok(self) -> bool:
        return bool(getattr(self.settings, "profanity_ok", False))

    def compose(
        self,
        anchor: Anchor,
        observed: Observed,
        policy: Policy,
        register: Register,
        repeat_index: int,
        language: str,
        muted: bool = False,
    ) -> Composition:
        try:
            return self._compose(anchor, observed, policy, register, repeat_index, language, muted)
        except Exception:
            # Last resort: something plain, never silence and never an exception.
            try:
                return Composition(
                    roast=plain_statement(anchor, observed, language),
                    choice_line=choice_line(language),
                    joke_used=False,
                )
            except Exception:
                lang = _lang(language)
                return Composition(
                    roast="यह काम से अलग दिख रहा है।" if lang == "hi" else "This looks different from the plan.",
                    choice_line=choice_line(language),
                    joke_used=False,
                )

    def _compose(
        self,
        anchor: Anchor,
        observed: Observed,
        policy: Policy,
        register: Register,
        repeat_index: int,
        language: str,
        muted: bool,
    ) -> Composition:
        lang = _lang(language)
        idx = max(0, int(repeat_index or 0))
        template_choice = choice_line(lang)
        try:
            reg = Register(register)
        except ValueError:
            reg = Register.PLAYFUL

        # --- The four gates. Any one of them: no joke at all. ---
        if not getattr(policy, "humor_ok", True):
            return Composition(plain_statement(anchor, observed, lang), template_choice, False)
        confidence = float(getattr(observed, "confidence", 1.0))
        if confidence < _MIN_CONFIDENCE:
            return Composition(hedged_statement(anchor, observed, lang), template_choice, False)
        if reg == Register.DRY and idx > 0:
            return Composition(plain_statement(anchor, observed, lang), template_choice, False)
        if muted:
            return Composition(plain_statement(anchor, observed, lang), template_choice, False)

        # --- Joke allowed: generate, validate, regenerate once, fall back. ---
        sentence_cap, word_cap = caps_for(idx)
        profanity_ok = self._profanity_ok
        roast: Optional[str] = None
        choice = template_choice
        for _attempt in range(2):
            result = self._call_llm(anchor, observed, policy, reg, idx, lang, word_cap, sentence_cap, profanity_ok)
            if result is None:
                continue
            candidate, candidate_choice = result
            ok, _reason = validate_roast(candidate, observed, idx, profanity_ok=profanity_ok)
            if ok:
                roast = candidate.strip()
                if self._choice_acceptable(candidate_choice, profanity_ok):
                    choice = candidate_choice.strip()
                break

        if roast is None:
            candidate = fallback_roast(anchor, observed, idx, lang, profanity_ok=profanity_ok)
            ok, _reason = validate_roast(candidate, observed, idx, profanity_ok=profanity_ok)
            if not ok:
                # The observation itself carries something the joke must not repeat.
                return Composition(plain_statement(anchor, observed, lang), template_choice, False)
            roast = candidate

        return Composition(roast, choice, True)

    def _call_llm(
        self,
        anchor: Anchor,
        observed: Observed,
        policy: Policy,
        register: Register,
        repeat_index: int,
        language: str,
        word_cap: int,
        sentence_cap: int,
        profanity_ok: bool,
    ) -> Optional[tuple[str, str]]:
        """One guarded call to ``llm.compose``. Any exception or malformed result -> None."""
        try:
            result = self.llm.compose(
                anchor=anchor,
                observed=observed,
                policy=policy,
                register=register,
                repeat_index=repeat_index,
                language=language,
                word_cap=word_cap,
                sentence_cap=sentence_cap,
                profanity_ok=profanity_ok,
            )
        except Exception:
            return None
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            return None
        roast, choice = result
        if not isinstance(roast, str):
            return None
        if not isinstance(choice, str):
            choice = ""
        return roast, choice

    @staticmethod
    def _choice_acceptable(line: str, profanity_ok: bool) -> bool:
        line = (line or "").strip()
        if not line or not line.endswith("?"):
            return False
        if count_words(line) > _CHOICE_MAX_WORDS:
            return False
        return not find_banned(line, profanity_ok=profanity_ok)


# ---------------------------------------------------------------------------
# 5. RegisterLadder: dry | playful | spicy. Can only cool. Never warms.
# ---------------------------------------------------------------------------

class RegisterLadder:
    """Persists a cooled register for the day through a duck-typed store with
    ``get_setting(key, default)`` / ``set_setting(key, value)``.

    There is deliberately NO method that raises the register (no warm/raise/
    escalate). Once cooled for a day it stays cooled; the next day starts from
    the default again (goal.md 15).
    """

    KEY_REGISTER = "register_override"
    KEY_DAY = "register_override_day"

    def __init__(self, store: Any):
        self.store = store

    def _saved_for(self, today: str) -> Optional[Register]:
        try:
            day = self.store.get_setting(self.KEY_DAY, None)
            if day != today:
                return None
            value = self.store.get_setting(self.KEY_REGISTER, None)
            return Register(value) if value else None
        except Exception:
            return None

    def current(self, default: Register, today: str) -> Register:
        """The register to use today: the cooled one saved for ``today``, else ``default``
        (never anything warmer than what was saved for today)."""
        try:
            default_reg = Register(default)
        except ValueError:
            default_reg = Register.PLAYFUL
        saved = self._saved_for(today)
        if saved is None:
            return default_reg
        return _gentler(saved, default_reg)

    def cool(self, today: str, current: Register) -> Register:
        """Save one step gentler than the effective register for today and return it."""
        try:
            cur = Register(current)
        except ValueError:
            cur = Register.PLAYFUL
        saved = self._saved_for(today)
        effective = cur if saved is None else _gentler(saved, cur)
        cooled = cool_register(effective)
        try:
            self.store.set_setting(self.KEY_REGISTER, cooled.value)
            self.store.set_setting(self.KEY_DAY, today)
        except Exception:
            pass
        return cooled


def _gentler(a: Register, b: Register) -> Register:
    order = [Register.DRY, Register.PLAYFUL, Register.SPICY]
    return a if order.index(a) <= order.index(b) else b
