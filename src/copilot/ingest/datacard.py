"""Render unit counts as a markdown table and splice it into docs/data-card.md."""

from __future__ import annotations

import json
from pathlib import Path

from copilot.ingest.corpus import count_units
from copilot.ingest.download import LANGUAGES, RAW_DIR, REGULATIONS
from copilot.ingest.models import UNIT_TYPES, Unit

DATA_CARD = Path("docs/data-card.md")
START, END = "<!-- counts:start -->", "<!-- counts:end -->"


def render_counts(units: list[Unit], raw_dir: Path = RAW_DIR) -> str:
    manifest = json.loads((raw_dir / "manifest.json").read_text("utf-8"))
    downloaded = sorted({entry["downloaded"] for entry in manifest.values()})
    counts = count_units(units)
    header = ["Regulation", "Lang", *UNIT_TYPES, "Total"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for reg in REGULATIONS:
        for lang in LANGUAGES:
            row = [counts[(reg, lang, t)] for t in UNIT_TYPES]
            lines.append(f"| {reg} | {lang} | " + " | ".join(map(str, row)) + f" | {sum(row)} |")
    lines.append(
        "| **all** | | "
        + " | ".join(
            str(sum(counts[(r, lg, t)] for r in REGULATIONS for lg in LANGUAGES))
            for t in UNIT_TYPES
        )
        + f" | {len(units)} |"
    )
    lines += [
        "",
        f"Source: {len(manifest)} raw files (manifest.json), downloaded {', '.join(downloaded)}; "
        f"{len(units)} units total, produced by `make ingest`.",
    ]
    return "\n".join(lines)


def update_data_card(units: list[Unit], path: Path = DATA_CARD) -> None:
    text = path.read_text("utf-8")
    start, end = text.index(START) + len(START), text.index(END)
    path.write_text(text[:start] + "\n" + render_counts(units) + "\n" + text[end:], "utf-8")
