"""
Compute semantic drift scores between two aligned embedding spaces.

Score per word:  cosine_distance(vec_a[w], vec_b_aligned[w])  ∈ [0, 2]
  0 = identical context distribution
  2 = maximally opposite context distribution

Score per verse: mean over all words present in both aligned spaces.
Score per book:  mean ± std over all verse scores in that book.
"""

import re
import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from .alignment import aligned_word_vectors
from .corpus import BOOKS, verses_for_book


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(1.0 - np.dot(a, b) / denom)


def word_drift_scores(
    model_a: Word2Vec,
    model_b: Word2Vec,
) -> dict[str, float]:
    """
    Return a dict of word → cosine distance for all shared vocabulary.
    Higher = more drift between translations.
    """
    vecs_a, vecs_b = aligned_word_vectors(model_a, model_b)
    return {
        word: cosine_distance(vecs_a[word], vecs_b[word])
        for word in vecs_a
    }


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def verse_drift_score(
    verse_a: str,
    verse_b: str,
    vecs_a: dict[str, np.ndarray],
    vecs_b: dict[str, np.ndarray],
) -> float | None:
    """
    Mean semantic drift across word-pairs in a verse.
    Returns None if no shared words are in the aligned vocab.
    """
    tokens_a = set(_tokenize(verse_a))
    tokens_b = set(_tokenize(verse_b))
    shared = (tokens_a | tokens_b) & set(vecs_a.keys())

    scores = [cosine_distance(vecs_a[w], vecs_b[w]) for w in shared if w in vecs_b]
    return float(np.mean(scores)) if scores else None


def compute_drift(
    translation_a: dict[tuple[str, int, int], str],
    translation_b: dict[tuple[str, int, int], str],
    model_a: Word2Vec,
    model_b: Word2Vec,
    books: list[str] | None = None,
) -> pd.DataFrame:
    """
    Compute per-verse semantic drift across all (or specified) books.

    Returns a DataFrame with columns:
        book, chapter, verse, drift_score
    """
    vecs_a, vecs_b = aligned_word_vectors(model_a, model_b)
    target_books = books or BOOKS
    rows = []

    for book in target_books:
        shared_keys = sorted(
            k for k in translation_a if k[0] == book and k in translation_b,
            key=lambda k: (k[1], k[2]),
        )
        for key in shared_keys:
            score = verse_drift_score(
                translation_a[key], translation_b[key], vecs_a, vecs_b
            )
            if score is not None:
                rows.append({
                    "book": book,
                    "chapter": key[1],
                    "verse": key[2],
                    "drift_score": score,
                })

    return pd.DataFrame(rows)


def book_summary(drift_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate per-verse drift to per-book mean ± std.
    Returns DataFrame sorted by mean drift (descending).
    """
    summary = (
        drift_df.groupby("book")["drift_score"]
        .agg(mean_drift="mean", std_drift="std", verse_count="count")
        .reset_index()
        .sort_values("mean_drift", ascending=False)
    )
    # Preserve canonical book order as a secondary sort column
    order = {b: i for i, b in enumerate(BOOKS)}
    summary["book_order"] = summary["book"].map(order)
    return summary
