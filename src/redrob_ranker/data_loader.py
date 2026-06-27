"""Loading candidates from the JSONL dataset (and JSON sample format)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterator


def iter_candidates_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    """Stream candidates from a .jsonl (one JSON object per line) file."""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_candidates_json(path: str | Path) -> list[dict[str, Any]]:
    """Load candidates from a JSON array file (e.g. sample_candidates.json)."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_candidates(path: str | Path) -> list[dict[str, Any]]:
    """
    Load candidates from either a .jsonl or .json file, auto-detected by
    extension and falling back to content sniffing if the extension is
    ambiguous (e.g. ".jsonl.gz" handling is left to callers who need it).
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        return list(iter_candidates_jsonl(path))
    if suffix == ".json":
        return load_candidates_json(path)

    # Fallback: sniff first non-whitespace character.
    with open(path, "r", encoding="utf-8") as f:
        first_char = f.read(1)
    if first_char == "[":
        return load_candidates_json(path)
    return list(iter_candidates_jsonl(path))
