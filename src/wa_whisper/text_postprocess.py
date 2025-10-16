"""Lightweight text post-processing utilities."""

from __future__ import annotations

import re
from typing import Callable, Iterable

from word2number import w2n


_NUMBER_WORD_PATTERN = re.compile(
    r"""
    (?P<prefix>\b|^)
    (?P<body>(?:zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|
              thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|
              thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|million|
              billion|trillion|and|point)(?:[\s-]+(?:zero|one|two|three|four|five|six|
              seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|
              seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|
              eighty|ninety|hundred|thousand|million|billion|trillion|and|point))*)(?P<suffix>\b|$)
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _word_to_number(fragment: str) -> str:
    normalized = fragment.replace("-", " ").lower()
    try:
        value = w2n.word_to_num(normalized)
    except ValueError:
        return fragment
    return str(value)


def normalize_numbers(text: str) -> str:
    """Convert number words to digits while preserving punctuation."""

    def replacer(match: re.Match[str]) -> str:
        prefix = match.group("prefix") or ""
        body = match.group("body") or ""
        suffix = match.group("suffix") or ""
        replacement = _word_to_number(body)
        return f"{prefix}{replacement}{suffix}"

    return _NUMBER_WORD_PATTERN.sub(replacer, text)


def normalize_acronyms(text: str) -> str:
    """Uppercase common acronym patterns (e.g., 'f b i' -> 'FBI')."""
    tokens = text.split()
    result: list[str] = []
    i = 0
    while i < len(tokens):
        window: list[str] = []
        j = i
        while j < len(tokens) and re.fullmatch(r"[A-Za-z]", tokens[j]):
            window.append(tokens[j].upper())
            j += 1
        if len(window) >= 2:
            result.append("".join(window))
            i = j
        else:
            result.append(tokens[i])
            i += 1
    return " ".join(result)


def clean_whitespace(text: str) -> str:
    """Collapse repeated whitespace and trim edges."""
    return re.sub(r"\s+", " ", text).strip()


def ensure_sentence_final_punctuation(text: str) -> str:
    """Append a period if the sentence lacks terminal punctuation."""
    if not text:
        return text
    if text[-1] in ".!?":
        return text
    return f"{text}."


def postprocess_text(
    text: str,
    *,
    normalize_numbers_enabled: bool = True,
    normalize_acronyms_enabled: bool = True,
    ensure_punctuation: bool = True,
    append_space: bool = False,
) -> str:
    """Run the configured post-processing passes."""
    working = text
    working = clean_whitespace(working)
    if normalize_numbers_enabled:
        working = normalize_numbers(working)
    if normalize_acronyms_enabled:
        working = normalize_acronyms(working)
    if ensure_punctuation:
        working = ensure_sentence_final_punctuation(working)
    if append_space:
        working = f"{working} "
    return working


def apply_pipeline(text: str, transforms: Iterable[Callable[[str], str]]) -> str:
    """Apply an ordered collection of callables to the text."""
    result = text
    for transform in transforms:
        result = transform(result)
    return result
