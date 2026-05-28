"""
Cross-lingual source-fidelity analysis using multilingual sentence embeddings.

Uses sentence-transformers `paraphrase-multilingual-MiniLM-L12-v2` to embed
verses in any language into a shared 384-d space, then measures cosine distance
between the source-language verse and each English translation.

Limitation: the model was trained on modern languages. Koine Greek and Biblical
Hebrew are related to but distinct from Modern Greek and Modern Hebrew, so
embeddings carry an anachronism error. Results are directionally meaningful but
absolute distance values should not be over-interpreted.
"""

import numpy as np
import pandas as pd
from tqdm import tqdm

from .corpus import BOOKS

OT_BOOKS = BOOKS[:39]
NT_BOOKS = BOOKS[39:]


def _get_model(model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def embed_verses(
    translation: dict[tuple[str, int, int], str],
    books: list[str] | None = None,
    model=None,
    batch_size: int = 256,
    show_progress: bool = True,
) -> tuple[list[tuple[str, int, int]], np.ndarray]:
    """
    Embed all verses for the specified books.

    Returns:
        keys: list of (book, chapter, verse) tuples in order
        embeddings: (N, d) float32 array
    """
    if model is None:
        model = _get_model()

    target_books = set(books) if books else set(BOOKS)
    keys = sorted(
        (k for k in translation if k[0] in target_books),
        key=lambda k: (BOOKS.index(k[0]) if k[0] in BOOKS else 999, k[1], k[2]),
    )
    texts = [translation[k] for k in keys]

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    return keys, embeddings


def source_fidelity(
    source: dict[tuple[str, int, int], str],
    target: dict[tuple[str, int, int], str],
    books: list[str] | None = None,
    model=None,
    batch_size: int = 256,
) -> pd.DataFrame:
    """
    Compute per-verse cosine distance between source-language verse and
    target-language (English) translation.

    Lower = target is semantically closer to the source.
    Higher = target diverges from the source.

    Returns DataFrame: book, chapter, verse, fidelity_distance
    """
    if model is None:
        model = _get_model()

    shared_keys = sorted(
        (k for k in source if k in target and (books is None or k[0] in books)),
        key=lambda k: (BOOKS.index(k[0]) if k[0] in BOOKS else 999, k[1], k[2]),
    )

    src_texts = [source[k] for k in shared_keys]
    tgt_texts = [target[k] for k in shared_keys]

    print(f"  Embedding {len(shared_keys):,} source verses...")
    src_emb = model.encode(src_texts, batch_size=batch_size,
                           show_progress_bar=True, normalize_embeddings=True)
    print(f"  Embedding {len(shared_keys):,} target verses...")
    tgt_emb = model.encode(tgt_texts, batch_size=batch_size,
                           show_progress_bar=True, normalize_embeddings=True)

    # Cosine distance = 1 - cosine_similarity; since vecs are normalized, dot product = similarity
    similarities = (src_emb * tgt_emb).sum(axis=1)
    distances = 1.0 - similarities

    rows = [
        {
            "book": k[0],
            "chapter": k[1],
            "verse": k[2],
            "fidelity_distance": float(d),
        }
        for k, d in zip(shared_keys, distances)
    ]
    return pd.DataFrame(rows)


def compare_translations_to_source(
    source: dict[tuple[str, int, int], str],
    translations: dict[str, dict[tuple[str, int, int], str]],
    books: list[str] | None = None,
    model=None,
    batch_size: int = 256,
) -> pd.DataFrame:
    """
    Compare multiple English translations against a single source language.

    Returns DataFrame: book, chapter, verse, <translation_name>, ...
    One fidelity_distance column per translation.
    """
    if model is None:
        model = _get_model()

    shared_keys = sorted(
        (
            k for k in source
            if all(k in t for t in translations.values())
            and (books is None or k[0] in books)
        ),
        key=lambda k: (BOOKS.index(k[0]) if k[0] in BOOKS else 999, k[1], k[2]),
    )

    print(f"Embedding {len(shared_keys):,} source verses...")
    src_emb = model.encode(
        [source[k] for k in shared_keys],
        batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True,
    )

    result_df = pd.DataFrame({
        "book": [k[0] for k in shared_keys],
        "chapter": [k[1] for k in shared_keys],
        "verse": [k[2] for k in shared_keys],
    })

    for name, translation in translations.items():
        print(f"Embedding {name}...")
        tgt_emb = model.encode(
            [translation[k] for k in shared_keys],
            batch_size=batch_size, show_progress_bar=True, normalize_embeddings=True,
        )
        result_df[f"dist_{name}"] = 1.0 - (src_emb * tgt_emb).sum(axis=1)

    return result_df


def book_fidelity_summary(fidelity_df: pd.DataFrame, dist_cols: list[str] | None = None) -> pd.DataFrame:
    """Aggregate fidelity distances to per-book means."""
    if dist_cols is None:
        dist_cols = [c for c in fidelity_df.columns if c.startswith("dist_") or c == "fidelity_distance"]
    return (
        fidelity_df.groupby("book")[dist_cols]
        .mean()
        .reset_index()
        .sort_values(dist_cols[0], ascending=False)
    )
