# bible-semantic-drift

Per-book semantic and phonetic shift analysis across Bible translations using word embeddings.

Answers questions like:
- Which books of the Bible drift most between the KJV and the World English Bible?
- Which individual words changed meaning most between translations?
- Does the phonetic character of a book change as much as its semantic content?
- *(v2)* How far does each English translation diverge from the Koine Greek source?

## Method

1. **Corpus**: Download parallel verse-aligned translations from [BibleNLP/ebible](https://github.com/BibleNLP/ebible)
2. **Embeddings**: Train separate Word2Vec models on each translation (window=10, 200d)
3. **Alignment**: Orthogonal Procrustes rotation maps both spaces into alignment ([Hamilton et al. 2016](https://arxiv.org/abs/1605.09096))
4. **Semantic drift**: Cosine distance per word in the aligned space; averaged per verse → per book
5. **Phonetic drift**: Metaphone code distance between position-aligned word pairs per verse

The Procrustes step is key: it removes the arbitrary rotation of each independently-trained embedding space so that distances reflect genuine semantic divergence, not training randomness.

## Quickstart

```bash
pip install -e ".[dev]"

# Download two public-domain translations
bible-drift download kjv web

# Train embeddings (~2 min each on a laptop)
bible-drift train kjv web

# Compute drift (semantic + phonetic)
bible-drift analyze --source kjv --target web

# Print table and save charts
bible-drift report --source kjv --target web --book Psalms
```

## Available public-domain translations

| Code | Translation | Year |
|------|-------------|------|
| `kjv` | King James Version | 1769 |
| `web` | World English Bible | 2000 |
| `asv` | American Standard Version | 1901 |
| `ylt` | Young's Literal Translation | 1898 |
| `darby` | Darby Bible | 1890 |

> **NIV / ESV**: These are under copyright and not included in the ebible corpus. You can provide your own text files in the `data/` directory in the same verse-per-line format.

## Output

`bible-drift analyze` produces three files in `results/`:

| File | Contents |
|------|----------|
| `drift_kjv_to_web.csv` | Per-verse semantic drift score |
| `phonetic_kjv_to_web.csv` | Per-verse phonetic drift score |
| `words_kjv_to_web.csv` | Top-100 words by semantic drift |

`bible-drift report` adds:
- `drift_by_book_kjv_to_web.png` — horizontal bar chart, OT/NT colored
- `timeline_Psalms_kjv_to_web.png` — verse-level drift across a book

## Notebooks

| Notebook | Contents |
|----------|----------|
| `notebooks/01_corpus_exploration.ipynb` | Vocabulary sizes, verse counts, token distributions |
| `notebooks/02_drift_analysis.ipynb` | Full KJV→WEB analysis with commentary |

## Koine Greek (v2)

The `bible_drift/greek/scaffold.py` module stubs out a source-fidelity pipeline using [Ancient-Greek-BERT](https://github.com/pranaydeeps/Ancient-Greek-BERT). The planned approach embeds each NT Greek verse and measures cosine distance to each English translation, giving a per-book "fidelity to source" score. Install extras:

```bash
pip install -e ".[greek]"
```

## Project structure

```
bible-semantic-drift/
├── bible_drift/
│   ├── corpus.py       # download + parse ebible translations
│   ├── embeddings.py   # Word2Vec training
│   ├── alignment.py    # Procrustes alignment
│   ├── drift.py        # semantic drift scoring
│   ├── phonetic.py     # Metaphone phonetic drift
│   ├── report.py       # charts + summary tables
│   └── greek/
│       └── scaffold.py # Koine Greek pipeline (v2)
├── cli.py              # `bible-drift` CLI entry point
├── notebooks/          # Jupyter analysis notebooks
├── tests/              # pytest suite
├── data/               # downloaded corpus files (gitignored)
├── models/             # trained Word2Vec models (gitignored)
└── results/            # output CSVs and charts (gitignored)
```

## Citation / related work

- Hamilton, W. L., Leskovec, J., & Jurafsky, D. (2016). [Diachronic Word Embeddings Reveal Statistical Laws of Semantic Change](https://arxiv.org/abs/1605.09096). ACL 2016.
- Mitra, B. et al. (2021). [BibleNLP/ebible](https://github.com/BibleNLP/ebible) — parallel Bible corpus.
- Panagiotopoulos, P. et al. (2021). [Ancient-Greek-BERT](https://github.com/pranaydeeps/Ancient-Greek-BERT). LaTeCH 2021.

## License

MIT
