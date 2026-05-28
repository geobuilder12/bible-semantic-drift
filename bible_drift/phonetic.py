"""
Phonetic drift between translations.

Measures how much the sound of the text changed, independent of meaning.
Uses Metaphone (better than Soundex for English) via the jellyfish library.

Example: KJV "thou art" vs WEB "you are"
  - "thou"/"you":  different Metaphone codes → high phonetic distance
  - "art"/"are":   similar codes → low phonetic distance
"""

import re
import numpy as np
import pandas as pd
import jellyfish

from .corpus import BOOKS


def metaphone_distance(word_a: str, word_b: str) -> float:
    """
    0.0 = identical Metaphone code (same sound)
    1.0 = completely different (no shared phoneme pattern)

    Uses simple binary match on Metaphone codes. For a graded score,
    we fall back to normalised Levenshtein on the codes themselves.
    """
    code_a = jellyfish.metaphone(word_a)
    code_b = jellyfish.metaphone(word_b)
    if code_a == code_b:
        return 0.0
    if not code_a or not code_b:
        return 1.0
    dist = jellyfish.levenshtein_distance(code_a, code_b)
    max_len = max(len(code_a), len(code_b))
    return dist / max_len


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z]+", text.lower())  # phonetic stays ASCII-only (English only)


def verse_phonetic_drift(verse_a: str, verse_b: str) -> float | None:
    """
    Mean phonetic distance across word pairs aligned by position.

    Aligns by position (not meaning) — shorter verse length is the limit.
    Returns None if either verse is empty.
    """
    tokens_a = _tokenize(verse_a)
    tokens_b = _tokenize(verse_b)
    pairs = list(zip(tokens_a, tokens_b))
    if not pairs:
        return None
    scores = [metaphone_distance(a, b) for a, b in pairs]
    return float(np.mean(scores))


def compute_phonetic_drift(
    translation_a: dict[tuple[str, int, int], str],
    translation_b: dict[tuple[str, int, int], str],
    books: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute per-verse phonetic drift across all (or specified) books.

    Returns DataFrame: book, chapter, verse, phonetic_drift
    """
    target_books = books or BOOKS
    rows = []

    for book in target_books:
        shared_keys = sorted(
            (k for k in translation_a if k[0] == book and k in translation_b),
            key=lambda k: (k[1], k[2]),
        )
        for key in shared_keys:
            score = verse_phonetic_drift(translation_a[key], translation_b[key])
            if score is not None:
                rows.append({
                    "book": book,
                    "chapter": key[1],
                    "verse": key[2],
                    "phonetic_drift": score,
                })

    return pd.DataFrame(rows)
