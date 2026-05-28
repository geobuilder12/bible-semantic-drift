"""
Koine Greek source-fidelity pipeline — scaffold (v1).

Planned approach:
  1. Load the SBL Greek New Testament (SBLGNT) as the source.
  2. Embed each Greek word using the Ancient-Greek-BERT model
     (pranaydeeps/Ancient-Greek-BERT on HuggingFace).
  3. For each English translation, embed the corresponding verse.
  4. Measure cosine distance between source Greek embedding and
     each English translation's verse embedding (cross-lingual via
     language-neutral sentence transformer or aligned space).
  5. Aggregate per book → "source fidelity score."

Lower score = closer to the Greek source.
Higher score = greater interpretive distance from the source.

This is novel: existing work measures English↔English drift.
Measuring Greek→English fidelity per book could reveal which books
translators diverged from the source most (e.g., poetry vs. epistles).

Install extras to use:
    pip install "bible-semantic-drift[greek]"
"""

# TODO: implement when `transformers` and `torch` are available
# Model: https://huggingface.co/pranaydeeps/Ancient-Greek-BERT


def load_greek_bert():
    """Load Ancient-Greek-BERT tokenizer and model."""
    raise NotImplementedError(
        "Koine Greek pipeline coming in v2. "
        "Install extras: pip install 'bible-semantic-drift[greek]'"
    )


def embed_greek_verse(text: str, model, tokenizer) -> "np.ndarray":
    """Return a mean-pooled BERT embedding for a Koine Greek verse."""
    raise NotImplementedError


def source_fidelity_score(greek_verse: str, english_verse: str, model, tokenizer) -> float:
    """
    Cosine distance between the Greek source embedding and an English
    translation embedding in a shared cross-lingual space.
    """
    raise NotImplementedError


def load_sblgnt(path: str) -> dict:
    """
    Load the SBL Greek New Testament text.
    Expects a verse-per-line text file from https://sblgnt.com
    Returns {(book, chapter, verse): greek_text}
    """
    raise NotImplementedError
