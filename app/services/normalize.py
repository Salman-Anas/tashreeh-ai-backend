"""Urdu text normalization.

Two levels:

* ``normalize_urdu`` — canonical *characters* (Arabic Yeh/Kaf/Heh → Urdu forms,
  Arabic-Indic digits → Urdu digits, ZWNJ clean-up, whitespace collapse). Keeps
  diacritics. Safe to store (e.g. TM rows) but never applied to text the user
  typed and sees on screen.
* ``normalize_for_matching`` — everything above plus diacritics, tatweel and all
  zero-width characters removed, and Heh variants folded. Used only for glossary
  matching, term checking and embeddings. ``normalize_with_map`` returns the same
  string together with an index map back into the original text so matches can be
  highlighted in the user's original (un-normalized) input.

A note on Heh: Urdu text typed on Arabic keyboards uses Arabic Heh ``ه`` (U+0647)
where Urdu orthography has Heh Goal ``ہ`` (U+06C1). U+0647 is not part of the
standard Urdu alphabet, so we always map it to U+06C1. Do-chashmi Heh ``ھ``
(U+06BE, used for aspiration as in ``کھ``) is a distinct letter and is left alone.
"""

from __future__ import annotations

import re
import unicodedata

ARABIC_YEH = "ي"
ALEF_MAKSURA = "ى"
URDU_YEH = "ی"
ARABIC_KAF = "ك"
URDU_KEHEH = "ک"
ARABIC_HEH = "ه"
HEH_GOAL = "ہ"
HEH_DOACHASHMEE = "ھ"
ALEF = "ا"
ALEF_MADDA = "آ"
MADDA_ABOVE = "ٓ"
ZWNJ = "‌"

_CHAR_MAP: dict[str, str] = {
    ARABIC_YEH: URDU_YEH,
    ALEF_MAKSURA: URDU_YEH,
    ARABIC_KAF: URDU_KEHEH,
    ARABIC_HEH: HEH_GOAL,
    # Arabic-Indic digits → Extended Arabic-Indic (Urdu) digits
    **{chr(0x0660 + i): chr(0x06F0 + i) for i in range(10)},
}

# Extra folding applied for matching only.
_MATCH_MAP: dict[str, str] = {
    **_CHAR_MAP,
    "ۂ": HEH_GOAL,  # ۂ heh goal + hamza (izafat)  → ہ
    "ۀ": HEH_GOAL,  # ۀ heh + yeh above             → ہ
    "ە": HEH_GOAL,  # ە ae, sometimes used for final heh → ہ
    "أ": ALEF,      # أ alef + hamza above          → ا
    "إ": ALEF,      # إ alef + hamza below          → ا
}

# Harakat U+064B–U+0652, superscript alef U+0670, hamza above/below marks,
# and tatweel U+0640. (Madda U+0653 is handled separately.)
_DIACRITICS: frozenset[str] = frozenset(
    [chr(c) for c in range(0x064B, 0x0653)] + ["ٰ", "ٔ", "ٕ", "ٖ", "ـ"]
)
_ZERO_WIDTH: frozenset[str] = frozenset(["​", "‌", "‍", "⁠", "﻿", "­"])

_ZWNJ_RUN = re.compile(ZWNJ + "{2,}")
_ZWNJ_NEAR_SPACE = re.compile(rf"[^\S\n]*{ZWNJ}+[^\S\n]+|[^\S\n]+{ZWNJ}+[^\S\n]*")
_ZWNJ_AT_LINE_EDGE = re.compile(rf"{ZWNJ}+(?=\n)|(?<=\n){ZWNJ}+")
_STRAY_ZW = re.compile("[​⁠﻿]")
_HSPACE = re.compile(r"[^\S\n]+")
_MANY_NEWLINES = re.compile(r"\n{3,}")
_ANY_SPACE = re.compile(r"\s+")


def _map_chars(text: str, table: dict[str, str]) -> str:
    return "".join(table.get(ch, ch) for ch in text)


def normalize_urdu(text: str, *, preserve_newlines: bool = True) -> str:
    """Canonicalize Urdu characters and whitespace. Keeps diacritics."""
    if not text:
        return ""
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = _map_chars(text, _CHAR_MAP)
    text = _STRAY_ZW.sub("", text)
    text = _ZWNJ_RUN.sub(ZWNJ, text)
    text = _ZWNJ_NEAR_SPACE.sub(" ", text)
    text = _ZWNJ_AT_LINE_EDGE.sub("", text)
    if preserve_newlines:
        text = _HSPACE.sub(" ", text)
        text = "\n".join(line.strip() for line in text.split("\n"))
        text = _MANY_NEWLINES.sub("\n\n", text)
    else:
        text = _ANY_SPACE.sub(" ", text)
    return text.strip(" \n" + ZWNJ)


def normalize_with_map(text: str) -> tuple[str, list[int]]:
    """Matching-normalize ``text`` and return ``(normalized, index_map)``.

    ``index_map[i]`` is the index in ``text`` of the character that produced
    ``normalized[i]``. Diacritics, tatweel and zero-width characters are dropped,
    whitespace runs collapse to a single space, and letters are folded to their
    canonical Urdu forms.
    """
    out: list[str] = []
    idx: list[int] = []
    for i, ch in enumerate(text):
        if ch == MADDA_ABOVE:
            # Decomposed alef + madda → precomposed alef madda.
            if out and out[-1] == ALEF:
                out[-1] = ALEF_MADDA
            continue
        if ch in _DIACRITICS or ch in _ZERO_WIDTH:
            continue
        if ch.isspace():
            if out and out[-1] != " ":
                out.append(" ")
                idx.append(i)
            continue
        out.append(_MATCH_MAP.get(ch, ch))
        idx.append(i)
    if out and out[-1] == " ":
        out.pop()
        idx.pop()
    return "".join(out), idx


def normalize_for_matching(text: str) -> str:
    """Aggressive normalization for comparisons only — never for display."""
    if not text:
        return ""
    return normalize_with_map(unicodedata.normalize("NFC", text))[0]


def strip_diacritics(text: str) -> str:
    return "".join(ch for ch in text if ch not in _DIACRITICS)


_URDU_RANGE = re.compile(r"[؀-ۿݐ-ݿﭐ-﷿ﹰ-﻿]")
_LATIN_RANGE = re.compile(r"[A-Za-z]")


def urdu_ratio(text: str) -> float:
    """Share of letters in ``text`` that are Arabic-script."""
    ur = len(_URDU_RANGE.findall(text))
    en = len(_LATIN_RANGE.findall(text))
    total = ur + en
    return ur / total if total else 0.0
