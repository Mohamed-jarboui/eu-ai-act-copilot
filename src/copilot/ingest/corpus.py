"""Build the corpus from data/raw, write it as JSONL, and count units."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from copilot.ingest.download import LANGUAGES, RAW_DIR, REGULATIONS, raw_path
from copilot.ingest.models import UNIT_TYPES, Unit
from copilot.ingest.parse import parse_file

CORPUS_PATH = Path("data/processed/corpus.jsonl")


def build_corpus(raw_dir: Path = RAW_DIR) -> list[Unit]:
    units: list[Unit] = []
    for regulation in REGULATIONS:
        for lang in LANGUAGES:
            units.extend(parse_file(raw_path(regulation, lang, raw_dir), regulation, lang))
    return units


def check_unique(units: list[Unit]) -> None:
    """Raise if an id repeats within one (regulation, language)."""
    seen = Counter((u.regulation, u.language, u.id) for u in units)
    dupes = [key for key, n in seen.items() if n > 1]
    if dupes:
        raise ValueError(f"{len(dupes)} duplicate ids, e.g. {dupes[:5]}")


def write_corpus(units: list[Unit], path: Path = CORPUS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for unit in units:
            fh.write(json.dumps(unit.to_dict(), ensure_ascii=False) + "\n")


def read_corpus(path: Path = CORPUS_PATH) -> list[Unit]:
    with path.open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh]
    return [Unit(**{**row, "refs": tuple(row["refs"])}) for row in rows]


def count_units(units: list[Unit]) -> dict[tuple[str, str, str], int]:
    """Counts keyed by (regulation, language, type)."""
    counts = Counter((u.regulation, u.language, u.type) for u in units)
    return {
        (reg, lang, utype): counts.get((reg, lang, utype), 0)
        for reg in REGULATIONS
        for lang in LANGUAGES
        for utype in UNIT_TYPES
    }
