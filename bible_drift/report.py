"""
Visualization and reporting utilities.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from .corpus import BOOKS
from .drift import book_summary


sns.set_theme(style="whitegrid", palette="muted")

OT_BOOKS = set(BOOKS[:39])
NT_BOOKS = set(BOOKS[39:])


def plot_drift_by_book(
    drift_df: pd.DataFrame,
    title: str = "Semantic Drift by Book",
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Horizontal bar chart of mean drift per book, colored by OT/NT.
    Books sorted by canonical order.
    """
    summary = book_summary(drift_df).sort_values("book_order")

    colors = ["#4e79a7" if b in OT_BOOKS else "#f28e2b" for b in summary["book"]]

    fig, ax = plt.subplots(figsize=(10, max(8, len(summary) * 0.22)))
    ax.barh(summary["book"], summary["mean_drift"], xerr=summary["std_drift"],
            color=colors, alpha=0.85, error_kw={"elinewidth": 0.8, "alpha": 0.5})
    ax.set_xlabel("Mean semantic drift (cosine distance)")
    ax.set_title(title)
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#4e79a7", label="Old Testament"),
        Patch(facecolor="#f28e2b", label="New Testament"),
    ]
    ax.legend(handles=legend_elements, loc="lower right")

    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig


def plot_drift_timeline(
    drift_df: pd.DataFrame,
    book: str,
    title: str | None = None,
    output_path: str | Path | None = None,
) -> plt.Figure:
    """
    Line plot of verse-level semantic drift across a single book.
    """
    book_df = drift_df[drift_df["book"] == book].copy()
    book_df["verse_num"] = range(len(book_df))

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(book_df["verse_num"], book_df["drift_score"], linewidth=0.8, color="#4e79a7")
    ax.fill_between(book_df["verse_num"], book_df["drift_score"], alpha=0.2, color="#4e79a7")

    # Chapter markers
    chapter_starts = book_df.groupby("chapter")["verse_num"].first()
    for ch, pos in chapter_starts.items():
        ax.axvline(pos, color="gray", linewidth=0.4, linestyle="--")
        ax.text(pos + 0.3, ax.get_ylim()[1] * 0.95, str(ch),
                fontsize=6, color="gray", va="top")

    ax.set_xlabel("Verse (sequential)")
    ax.set_ylabel("Semantic drift")
    ax.set_title(title or f"Verse-level semantic drift — {book}")
    plt.tight_layout()
    if output_path:
        fig.savefig(output_path, dpi=150)
    return fig


def top_drifted_words(
    word_scores: dict[str, float],
    n: int = 30,
    exclude_stopwords: bool = True,
) -> pd.DataFrame:
    """
    Return the N words with highest semantic drift.
    """
    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to",
        "of", "for", "with", "he", "she", "it", "they", "we", "i",
        "his", "her", "their", "our", "my", "thy", "thee", "thou",
        "shall", "will", "not", "be", "is", "are", "was", "were",
        "that", "this", "which", "who", "whom", "said", "unto", "upon",
    }
    scores = {
        w: s for w, s in word_scores.items()
        if not (exclude_stopwords and w in stopwords)
    }
    df = pd.DataFrame(list(scores.items()), columns=["word", "drift_score"])
    return df.nlargest(n, "drift_score").reset_index(drop=True)


def print_summary_table(drift_df: pd.DataFrame, phonetic_df: pd.DataFrame | None = None) -> None:
    summary = book_summary(drift_df)
    if phonetic_df is not None:
        ph = (
            phonetic_df.groupby("book")["phonetic_drift"]
            .mean()
            .reset_index()
            .rename(columns={"phonetic_drift": "mean_phonetic_drift"})
        )
        summary = summary.merge(ph, on="book", how="left")

    cols = ["book", "verse_count", "mean_drift", "std_drift"]
    if "mean_phonetic_drift" in summary.columns:
        cols.append("mean_phonetic_drift")

    print(summary[cols].sort_values("book_order" if "book_order" in summary.columns else "mean_drift")
          .to_string(index=False, float_format="%.4f"))
