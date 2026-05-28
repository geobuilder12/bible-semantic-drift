"""
Train or load word embedding models for Bible translations.
"""

from pathlib import Path
from gensim.models import Word2Vec
from tqdm import tqdm


def train_word2vec(
    sentences: list[list[str]],
    vector_size: int = 200,
    window: int = 10,
    min_count: int = 1,
    workers: int = 4,
    epochs: int = 20,
    seed: int = 42,
) -> Word2Vec:
    """
    Train a Word2Vec model on tokenized Bible verses.

    window=10 is intentionally large: biblical prose has long-range
    thematic context that short windows miss.
    min_count=1 because the biblical vocabulary is finite and rare words
    (proper nouns, hapax legomena) carry translation signal.
    """
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
        epochs=epochs,
        seed=seed,
    )
    return model


def save_model(model: Word2Vec, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model.save(str(path))


def load_model(path: str | Path) -> Word2Vec:
    return Word2Vec.load(str(path))
