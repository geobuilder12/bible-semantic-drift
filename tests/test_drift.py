"""Unit tests for core drift computation."""

import numpy as np
import pytest
from gensim.models import Word2Vec

from bible_drift.drift import cosine_distance, verse_drift_score, word_drift_scores
from bible_drift.phonetic import metaphone_distance, verse_phonetic_drift
from bible_drift.corpus import _tokenize


def _tiny_model(sentences, size=10):
    return Word2Vec(sentences=sentences, vector_size=size, window=3, min_count=1, epochs=5, seed=0)


SENTENCES_A = [
    ["in", "the", "beginning", "god", "created", "heaven", "earth"],
    ["the", "earth", "was", "without", "form", "and", "void"],
    ["and", "darkness", "was", "upon", "the", "face", "of", "deep"],
    ["god", "said", "let", "there", "be", "light"],
    ["god", "saw", "that", "light", "was", "good"],
]
SENTENCES_B = [
    ["in", "the", "beginning", "god", "created", "the", "heavens", "and", "earth"],
    ["now", "the", "earth", "was", "formless", "and", "empty"],
    ["darkness", "was", "over", "the", "surface", "of", "deep"],
    ["god", "said", "let", "there", "be", "light"],
    ["god", "saw", "that", "light", "was", "good"],
]


class TestCosinDistance:
    def test_identical_vectors(self):
        v = np.array([1.0, 0.0, 0.0])
        assert cosine_distance(v, v) == pytest.approx(0.0, abs=1e-6)

    def test_opposite_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([-1.0, 0.0])
        assert cosine_distance(a, b) == pytest.approx(2.0, abs=1e-6)

    def test_orthogonal_vectors(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        assert cosine_distance(a, b) == pytest.approx(1.0, abs=1e-6)

    def test_zero_vector(self):
        a = np.array([0.0, 0.0])
        b = np.array([1.0, 0.0])
        assert cosine_distance(a, b) == 0.0


class TestWordDriftScores:
    def test_returns_dict_of_floats(self):
        model_a = _tiny_model(SENTENCES_A)
        model_b = _tiny_model(SENTENCES_B)
        scores = word_drift_scores(model_a, model_b)
        assert isinstance(scores, dict)
        assert all(isinstance(v, float) for v in scores.values())

    def test_scores_in_range(self):
        model_a = _tiny_model(SENTENCES_A)
        model_b = _tiny_model(SENTENCES_B)
        scores = word_drift_scores(model_a, model_b)
        for word, score in scores.items():
            assert 0.0 <= score <= 2.0, f"{word}: {score} out of range"

    def test_shared_vocab_only(self):
        model_a = _tiny_model(SENTENCES_A)
        model_b = _tiny_model(SENTENCES_B)
        scores = word_drift_scores(model_a, model_b)
        vocab_a = set(model_a.wv.key_to_index)
        vocab_b = set(model_b.wv.key_to_index)
        shared = vocab_a & vocab_b
        assert set(scores.keys()).issubset(shared)


class TestVersePhoneticDrift:
    def test_identical_text(self):
        score = verse_phonetic_drift("and god said", "and god said")
        assert score == pytest.approx(0.0, abs=1e-6)

    def test_different_text(self):
        score = verse_phonetic_drift("thou art", "you are")
        assert score is not None
        assert score > 0.0

    def test_empty_verse(self):
        assert verse_phonetic_drift("", "some text") is None
        assert verse_phonetic_drift("some text", "") is None


class TestMetaphoneDistance:
    def test_same_word(self):
        assert metaphone_distance("god", "god") == pytest.approx(0.0)

    def test_different_word(self):
        assert metaphone_distance("thou", "you") > 0.0

    def test_similar_sound(self):
        # "heaven" vs "heavens" should be close
        assert metaphone_distance("heaven", "heavens") < metaphone_distance("god", "lord")


class TestTokenize:
    def test_strips_punctuation(self):
        assert _tokenize("In the beginning,") == ["in", "the", "beginning"]

    def test_lowercase(self):
        assert _tokenize("GOD") == ["god"]

    def test_apostrophe_kept(self):
        tokens = _tokenize("God's grace")
        assert "god's" in tokens
