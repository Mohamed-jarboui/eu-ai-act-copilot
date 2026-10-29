"""`python -m copilot.ingest`: fetch missing raw files, parse, write corpus.jsonl, print counts."""

from __future__ import annotations

from copilot.ingest.corpus import CORPUS_PATH, build_corpus, check_unique, write_corpus
from copilot.ingest.datacard import DATA_CARD, render_counts, update_data_card
from copilot.ingest.download import download_all


def main() -> None:
    download_all()  # skips files that are already in data/raw
    units = build_corpus()
    check_unique(units)
    write_corpus(units)
    update_data_card(units)
    print(render_counts(units))
    print(f"\nwrote {CORPUS_PATH} and updated {DATA_CARD}")


if __name__ == "__main__":
    main()
