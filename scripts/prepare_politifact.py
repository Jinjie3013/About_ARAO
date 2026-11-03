"""Build reproducible PolitiFact train/dev/test CSVs (Appendix D).

FakeNewsNet ships only ``id, news_url, title, tweet_ids``, so ``title`` is the
text field. Duplicate headlines are removed before splitting, otherwise the
train/dev overlap guard in ``arao.data.require_disjoint`` would (correctly)
refuse to run.
"""
from __future__ import annotations
import argparse
import csv
import json
import re
from pathlib import Path

csv.field_size_limit(10**9)


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def read(path: str, label: int, field: str) -> list[dict]:
    with open(path, encoding="utf-8-sig", errors="replace", newline="") as stream:
        rows = list(csv.DictReader(stream))
    out = []
    for i, row in enumerate(rows):
        text = (row.get(field) or "").strip()
        if not text:
            continue
        out.append({"id": row.get("id") or f"{Path(path).stem}:{i}", "text": text, "label": label})
    return out


def stratified_split(rows: list[dict], ratios: tuple[float, float, float], seed: int):
    import random

    rng = random.Random(seed)
    by_label: dict[int, list[dict]] = {}
    for row in rows:
        by_label.setdefault(row["label"], []).append(row)
    parts = ([], [], [])
    for label in sorted(by_label):
        group = by_label[label][:]
        rng.shuffle(group)
        n = len(group)
        n_train = int(round(n * ratios[0]))
        n_dev = int(round(n * ratios[1]))
        n_dev = min(n_dev, n - n_train - 1)
        slices = (group[:n_train], group[n_train:n_train + n_dev], group[n_train + n_dev:])
        for bucket, chunk in zip(parts, slices):
            bucket.extend(chunk)
    # Shuffle each split after slicing so a head slice (``--limit``) stays
    # class-mixed; membership is unchanged by this step.
    for bucket in parts:
        rng.shuffle(bucket)
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare PolitiFact splits.")
    parser.add_argument("--real", required=True)
    parser.add_argument("--fake", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--field", default="title")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ratios", default="0.7,0.15,0.15")
    args = parser.parse_args()

    ratios = tuple(float(x) for x in args.ratios.split(","))
    if len(ratios) != 3 or abs(sum(ratios) - 1) > 1e-9:
        raise SystemExit("ratios must be three values summing to 1")

    raw = read(args.real, 0, args.field) + read(args.fake, 1, args.field)
    # Drop repeated headlines, and report what that costs.
    seen: dict[str, dict] = {}
    conflicts = 0
    for row in raw:
        key = normalize(row["text"])
        if key in seen:
            if seen[key]["label"] != row["label"]:
                conflicts += 1
            continue
        seen[key] = row
    rows = list(seen.values())

    parts = stratified_split(rows, ratios, args.seed)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    names = ("train", "dev", "test")
    for name, chunk in zip(names, parts):
        target = out / f"{name}.csv"
        with target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=["id", "text", "label"])
            writer.writeheader()
            writer.writerows(chunk)

    summary = {
        "source_rows": len(raw),
        "kept_after_dedup": len(rows),
        "dropped_duplicates": len(raw) - len(rows),
        "conflicting_duplicate_labels": conflicts,
        "field": args.field,
        "seed": args.seed,
        "ratios": ratios,
        "splits": {name: {"n": len(chunk),
                          "by_label": {str(k): sum(1 for r in chunk if r["label"] == k)
                                       for k in sorted({r["label"] for r in chunk})}}
                   for name, chunk in zip(names, parts)},
    }
    (out / "split_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
