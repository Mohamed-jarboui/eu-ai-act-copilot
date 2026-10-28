"""Download the official XHTML of both regulations in EN/FR/DE from the Cellar REST endpoint.

eur-lex.europa.eu answers scripted requests with a bot challenge (HTTP 202, empty body), so we use
the Publications Office's Cellar endpoint, which serves the same Official Journal text.
"""

from __future__ import annotations

import hashlib
import json
import urllib.request
from datetime import date
from pathlib import Path

REGULATIONS: dict[str, str] = {"aiact": "32024R1689", "gdpr": "32016R0679"}
LANGUAGES: dict[str, str] = {"en": "eng", "fr": "fra", "de": "deu"}  # our code -> Cellar code
RAW_DIR = Path("data/raw")
USER_AGENT = "eu-ai-act-copilot/0.1 (research project; github.com/Mohamed-jarboui)"


def source_url(regulation: str) -> str:
    return f"https://publications.europa.eu/resource/celex/{REGULATIONS[regulation]}"


def raw_path(regulation: str, lang: str, raw_dir: Path = RAW_DIR) -> Path:
    return raw_dir / f"{regulation}_{lang}.html"


def fetch(regulation: str, lang: str, *, timeout: float = 120.0) -> tuple[bytes, str]:
    """Return (body, final URL after redirects) for one regulation in one language."""
    request = urllib.request.Request(
        source_url(regulation),
        headers={
            "Accept": "application/xhtml+xml",
            "Accept-Language": LANGUAGES[lang],
            "User-Agent": USER_AGENT,
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return response.read(), response.geturl()


def download_all(raw_dir: Path = RAW_DIR, *, force: bool = False) -> dict:
    """Download missing files and (re)write manifest.json with provenance for each file."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = raw_dir / "manifest.json"
    manifest: dict = json.loads(manifest_path.read_text("utf-8")) if manifest_path.exists() else {}
    for regulation in REGULATIONS:
        for lang in LANGUAGES:
            path = raw_path(regulation, lang, raw_dir)
            key = path.name
            if path.exists() and key in manifest and not force:
                print(f"skip {key} (already downloaded)")
                continue
            print(f"download {key} ...", flush=True)
            body, final_url = fetch(regulation, lang)
            path.write_bytes(body)
            manifest[key] = {
                "regulation": regulation,
                "celex": REGULATIONS[regulation],
                "language": lang,
                "source_url": source_url(regulation),
                "resolved_url": final_url,
                "downloaded": date.today().isoformat(),
                "bytes": len(body),
                "sha256": hashlib.sha256(body).hexdigest(),
            }
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", "utf-8")
    return manifest


if __name__ == "__main__":
    for name, info in download_all().items():
        print(f"{name}: {info['bytes']:,} bytes")
