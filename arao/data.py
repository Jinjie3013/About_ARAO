"""Strict CSV/JSONL loading and transparent experiment records."""
from __future__ import annotations
import csv
from dataclasses import asdict
import json
from pathlib import Path
from typing import Sequence

from .types import Candidate, Example, SearchResult


def load_examples(path: str, text_column: str = "text", label_column: str = "label") -> list[Example]:
    file = Path(path)
    with file.open(encoding="utf-8-sig", newline="") as stream:
        if file.suffix.lower() == ".csv":
            rows = list(csv.DictReader(stream))
        elif file.suffix.lower() == ".jsonl":
            rows = [json.loads(line) for line in stream if line.strip()]
        else:
            raise ValueError("Data must be CSV or JSONL.")
    examples, seen_ids = [], set()
    for i, row in enumerate(rows):
        # Matches the reference script's title/text support. An explicit column
        # other than 'text' is honored exactly.
        key = "title" if text_column == "text" and "text" not in row and "title" in row else text_column
        text = row.get(key)
        raw_label = row.get(label_column)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"{file.name}, row {i + 1}: missing/empty text.")
        if isinstance(raw_label, bool) or str(raw_label).strip() not in {str(x) for x in range(100)}:
            raise ValueError(f"{file.name}, row {i + 1}: label must be an integer class ID in [0,99].")
        uid = str(row.get("id", f"{file.stem}:{i}"))
        if not uid or uid in seen_ids:
            raise ValueError(f"{file.name}: duplicate or empty example ID: {uid}")
        seen_ids.add(uid)
        examples.append(Example(uid, text.strip(), int(raw_label)))
    if not examples:
        raise ValueError(f"Empty dataset: {path}")
    return examples


def require_disjoint(train: Sequence[Example], dev: Sequence[Example]) -> None:
    normalize = lambda text: " ".join(text.casefold().split())
    if {normalize(e.text) for e in train} & {normalize(e.text) for e in dev}:
        raise ValueError("Train/dev text overlap detected; use separate splits.")


def save_json(path: str, value) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    file.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def save_results(path: str, results: Sequence[SearchResult]) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w", encoding="utf-8") as stream:
        for result in results:
            stream.write(json.dumps(asdict(result), ensure_ascii=False, allow_nan=False) + "\n")


def save_examples(path: str, examples: Sequence[Example]) -> None:
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w", encoding="utf-8") as stream:
        for e in examples:
            stream.write(json.dumps({"id": e.uid, "text": e.text, "label": e.label}, ensure_ascii=False) + "\n")


def load_results(path: str) -> list[SearchResult]:
    results = []

    def candidate(row):
        return Candidate(**{**row, "strategy": tuple(row["strategy"]), "probabilities": tuple(row["probabilities"])})

    with Path(path).open(encoding="utf-8") as stream:
        for line in stream:
            if not line.strip():
                continue
            row = json.loads(line)
            results.append(SearchResult(
                Example(**row["example"]), tuple(row["original_probabilities"]),
                [candidate(c) for c in row["candidates"]],
                candidate(row["best"]) if row["best"] is not None else None,
                tuple(row.get("control_trace", ())),
            ))
    return results
