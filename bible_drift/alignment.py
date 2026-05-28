"""
Align two word embedding spaces via orthogonal Procrustes.

Given models A and B trained on different translations, we find an
orthogonal matrix W such that W @ B ≈ A for all shared vocabulary.
After alignment, cosine distance between A[word] and (W @ B[word])
measures how much the word's semantic context shifted between translations.

Reference: Mikolov et al. (2013); Hamilton et al. (2016) "Diachronic Word
Embeddings Reveal Statistical Laws of Semantic Change."
"""

import numpy as np
from scipy.linalg import orthogonal_procrustes
from gensim.models import Word2Vec


def procrustes_align(
    model_a: Word2Vec,
    model_b: Word2Vec,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Align model_b's embedding space to model_a's space.

    Returns:
        vecs_a: (N, d) matrix of embeddings from model_a for shared vocab
        vecs_b_aligned: (N, d) aligned embeddings from model_b
        shared_vocab: list of N words, in matching row order
    """
    shared = sorted(set(model_a.wv.key_to_index) & set(model_b.wv.key_to_index))
    if len(shared) < 10:
        raise ValueError(
            f"Only {len(shared)} shared words — translations may be from "
            "incompatible corpora or tokenization differs."
        )

    vecs_a = np.array([model_a.wv[w] for w in shared])
    vecs_b = np.array([model_b.wv[w] for w in shared])

    # Normalize rows before Procrustes (standard practice)
    vecs_a = _row_normalize(vecs_a)
    vecs_b = _row_normalize(vecs_b)

    rotation, _ = orthogonal_procrustes(vecs_b, vecs_a)
    vecs_b_aligned = vecs_b @ rotation

    return vecs_a, vecs_b_aligned, shared


def _row_normalize(mat: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return mat / norms


def aligned_word_vectors(
    model_a: Word2Vec,
    model_b: Word2Vec,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Convenience wrapper: returns two dicts word → aligned_vector.
    Both dicts share the same vocabulary (intersection only).
    """
    vecs_a, vecs_b_aligned, shared = procrustes_align(model_a, model_b)
    dict_a = {w: vecs_a[i] for i, w in enumerate(shared)}
    dict_b = {w: vecs_b_aligned[i] for i, w in enumerate(shared)}
    return dict_a, dict_b
