"""
bible-drift  —  CLI for Bible translation semantic drift analysis.

Commands:
    download    Fetch translation text files from BibleNLP/ebible
    train       Train Word2Vec models on one or more translations
    analyze     Compute semantic + phonetic drift between two translations
    report      Print summary table and save charts from a results file
"""

from pathlib import Path
import click
import pandas as pd

from bible_drift.corpus import download_ebible, load_translation, PUBLIC_TRANSLATIONS
from bible_drift.embeddings import train_word2vec, save_model, load_model
from bible_drift.corpus import all_sentences
from bible_drift.drift import compute_drift, word_drift_scores
from bible_drift.phonetic import compute_phonetic_drift
from bible_drift.report import (
    plot_drift_by_book,
    plot_drift_timeline,
    top_drifted_words,
    print_summary_table,
)


@click.group()
def main():
    """Bible semantic drift analysis tool."""
    pass


@main.command()
@click.argument("translations", nargs=-1, required=True)
@click.option("--data-dir", default="data", show_default=True, help="Directory to save corpus files.")
def download(translations, data_dir):
    """Download translation files from BibleNLP/ebible.

    TRANSLATIONS: one or more translation codes (e.g. kjv web asv ylt).

    Available public-domain translations: kjv, web, asv, ylt, darby
    """
    known = list(PUBLIC_TRANSLATIONS.keys())
    for t in translations:
        if t not in known:
            click.echo(f"Warning: '{t}' not in known list {known}. Trying anyway.")
        try:
            path = download_ebible(t, data_dir)
            click.echo(f"  Saved {t} → {path}")
        except Exception as e:
            click.echo(f"  Error downloading {t}: {e}", err=True)


@main.command()
@click.argument("translations", nargs=-1, required=True)
@click.option("--data-dir", default="data", show_default=True)
@click.option("--model-dir", default="models", show_default=True)
@click.option("--vector-size", default=200, show_default=True)
@click.option("--window", default=10, show_default=True)
@click.option("--epochs", default=20, show_default=True)
def train(translations, data_dir, model_dir, vector_size, window, epochs):
    """Train Word2Vec models for one or more translations.

    TRANSLATIONS: e.g. kjv web
    """
    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    for t in translations:
        # Resolve file path
        from bible_drift.corpus import PUBLIC_TRANSLATIONS
        code = PUBLIC_TRANSLATIONS.get(t, t)
        path = data_dir / f"{code}.txt"
        if not path.exists():
            click.echo(f"  {t}: file not found at {path}. Run `bible-drift download {t}` first.")
            continue

        click.echo(f"  Loading {t}...")
        translation = load_translation(path)
        sentences = all_sentences(translation)
        click.echo(f"  Training Word2Vec on {len(sentences):,} verses...")

        model = train_word2vec(
            sentences,
            vector_size=vector_size,
            window=window,
            epochs=epochs,
        )
        out = model_dir / f"{t}.model"
        save_model(model, out)
        click.echo(f"  Saved → {out}  (vocab: {len(model.wv):,} words)")


@main.command()
@click.option("--source", "-s", required=True, help="Source translation code (e.g. kjv)")
@click.option("--target", "-t", required=True, help="Target translation code (e.g. web)")
@click.option("--data-dir", default="data", show_default=True)
@click.option("--model-dir", default="models", show_default=True)
@click.option("--output-dir", default="results", show_default=True)
@click.option("--books", default=None, help="Comma-separated list of books to analyze (default: all)")
def analyze(source, target, data_dir, model_dir, output_dir, books):
    """Compute semantic and phonetic drift between two translations.

    Results are saved as CSV files in --output-dir.

    Example:
        bible-drift analyze --source kjv --target web
    """
    from bible_drift.corpus import PUBLIC_TRANSLATIONS

    data_dir = Path(data_dir)
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    def resolve_path(code_short):
        code = PUBLIC_TRANSLATIONS.get(code_short, code_short)
        return data_dir / f"{code}.txt"

    click.echo(f"Loading translations: {source}, {target}...")
    trans_a = load_translation(resolve_path(source))
    trans_b = load_translation(resolve_path(target))
    click.echo(f"  {source}: {len(trans_a):,} verses  |  {target}: {len(trans_b):,} verses")

    click.echo("Loading models...")
    model_a = load_model(model_dir / f"{source}.model")
    model_b = load_model(model_dir / f"{target}.model")

    book_filter = [b.strip() for b in books.split(",")] if books else None

    click.echo("Computing semantic drift...")
    drift_df = compute_drift(trans_a, trans_b, model_a, model_b, books=book_filter)
    drift_path = output_dir / f"drift_{source}_to_{target}.csv"
    drift_df.to_csv(drift_path, index=False)
    click.echo(f"  Saved → {drift_path}  ({len(drift_df):,} verses)")

    click.echo("Computing phonetic drift...")
    phonetic_df = compute_phonetic_drift(trans_a, trans_b, books=book_filter)
    phonetic_path = output_dir / f"phonetic_{source}_to_{target}.csv"
    phonetic_df.to_csv(phonetic_path, index=False)
    click.echo(f"  Saved → {phonetic_path}")

    click.echo("Computing word-level drift scores...")
    word_scores = word_drift_scores(model_a, model_b)
    top_words = top_drifted_words(word_scores, n=100)
    words_path = output_dir / f"words_{source}_to_{target}.csv"
    top_words.to_csv(words_path, index=False)
    click.echo(f"  Saved → {words_path}")

    click.echo("\nSummary:")
    print_summary_table(drift_df, phonetic_df)


@main.command("report")
@click.option("--source", "-s", required=True)
@click.option("--target", "-t", required=True)
@click.option("--results-dir", default="results", show_default=True)
@click.option("--book", default=None, help="Also plot verse-level timeline for this book.")
def report(source, target, results_dir, book):
    """Generate charts from saved analysis results.

    Example:
        bible-drift report --source kjv --target web --book Psalms
    """
    results_dir = Path(results_dir)
    drift_path = results_dir / f"drift_{source}_to_{target}.csv"
    phonetic_path = results_dir / f"phonetic_{source}_to_{target}.csv"

    if not drift_path.exists():
        click.echo(f"No results found at {drift_path}. Run `analyze` first.", err=True)
        return

    drift_df = pd.read_csv(drift_path)
    phonetic_df = pd.read_csv(phonetic_path) if phonetic_path.exists() else None

    click.echo("Summary table:")
    print_summary_table(drift_df, phonetic_df)

    chart_path = results_dir / f"drift_by_book_{source}_to_{target}.png"
    plot_drift_by_book(
        drift_df,
        title=f"Semantic Drift: {source.upper()} → {target.upper()}",
        output_path=chart_path,
    )
    click.echo(f"\nSaved chart → {chart_path}")

    if book:
        timeline_path = results_dir / f"timeline_{book.replace(' ', '_')}_{source}_to_{target}.png"
        plot_drift_timeline(drift_df, book, output_path=timeline_path)
        click.echo(f"Saved timeline → {timeline_path}")


if __name__ == "__main__":
    main()
