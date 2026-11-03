"""CLI composition and I/O. Core modules never depend on argparse."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
from importlib import metadata
from pathlib import Path

from .config import Config
from .data import load_examples, load_results, require_disjoint, save_examples, save_json
from .diagnosis import axis_alignment, single_axis_sensitivity
from .factory import build_detector, build_engine, build_space
from .metrics import attack_metrics
from .training import fit, generate_repair_data


def arguments():
    parser = argparse.ArgumentParser(description="ARAO core: discover, diagnose, and repair.")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "configs" / "politifact.json"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--label-column", default="label")
    parser.add_argument("--allow-overlap", action="store_true",
                        help="Accept train/dev text overlap (the official PolitiFact split needs this).")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "attack", "diagnose", "probe", "repair"):
        p = sub.add_parser(command)
        p.add_argument("--output", required=True, help="A new output directory.")
        if command in {"train", "repair"}:
            p.add_argument("--train", required=True)
            p.add_argument("--dev", required=True)
        if command in {"attack", "probe"}:
            p.add_argument("--data", required=True)
            p.add_argument("--limit", type=int, help="Explicit exploratory subset; omit for full evaluation.")
        if command != "diagnose":
            p.add_argument("--checkpoint", required=command != "train", help="Hugging Face checkpoint directory or identifier.")
        if command == "train":
            p.add_argument("--mode", choices=["vanilla", "online"], default="online")
        if command in {"probe", "repair"}:
            p.add_argument("--axes", nargs="+", default=["authority", "scarcity", "commitment"] if command == "probe" else ["authority", "scarcity"])
        if command == "probe":
            p.add_argument("--strengths", nargs="+", type=float, default=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
        if command == "diagnose":
            p.add_argument("--results", required=True, help="Frozen-detector attack results.jsonl.")
    return parser.parse_args()


def provenance(args, config):
    versions = {}
    for name in ("torch", "transformers", "numpy", "scikit-learn", "openai"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    hashes = {}
    for name in ("train", "dev", "data", "results"):
        value = getattr(args, name, None)
        if value:
            hashes[name] = hashlib.sha256(Path(value).read_bytes()).hexdigest()
    return {"created_utc": datetime.now(timezone.utc).isoformat(), "arguments": vars(args),
            "effective_config": config.as_dict(), "package_versions": versions, "input_sha256": hashes}


def main():
    args = arguments()
    config = Config.load(args.config)
    output = Path(args.output)
    # Avoid replacing earlier experiment results or checkpoints by accident.
    output.mkdir(parents=True, exist_ok=False)
    save_json(str(output / "run.json"), provenance(args, config))
    if args.command == "diagnose":
        summary = axis_alignment(load_results(args.results), build_space(config), seed=args.seed)
        save_json(str(output / "summary.json"), summary)
        print(summary)
        return

    def load(path):
        rows = load_examples(path, args.text_column, args.label_column)
        if any(e.label >= config.models.num_labels for e in rows):
            raise ValueError("A data label exceeds configured num_labels.")
        return rows

    if args.command in {"train", "repair"}:
        train, dev = load(args.train), load(args.dev)
        require_disjoint(train, dev, args.allow_overlap)
        if set(e.label for e in dev) != set(range(config.models.num_labels)):
            raise ValueError("The dev split must contain every configured class.")
    else:
        examples = load(args.data)
        if args.limit is not None:
            if args.limit < 1:
                raise ValueError("limit must be positive.")
            examples = examples[:args.limit]

    detector = build_detector(config, args.checkpoint, args.seed)
    engine = chat = None
    if args.command != "train" or args.mode == "online":
        engine, chat = build_engine(config, detector, args.seed)

    if args.command == "train":
        summary = fit(detector, train, dev, str(output), config.run.epochs,
                      config.models.batch_size, args.seed, engine, config.run.attack_fraction,
                      args.allow_overlap)
    elif args.command == "attack":
        results = []
        trace = output / "results.jsonl"
        # Stream completed trajectories: completed examples survive interruption.
        from dataclasses import asdict
        import json
        with trace.open("w", encoding="utf-8") as stream:
            for i, example in enumerate(examples):
                result = engine.search(example)
                results.append(result)
                stream.write(json.dumps(asdict(result), ensure_ascii=False, allow_nan=False) + "\n")
                stream.flush()
                print(f"attack {i + 1}/{len(examples)} strict_success={result.strict_success}", flush=True)
        summary = attack_metrics(results)
    elif args.command == "probe":
        summary = {"single_axis_sensitivity": single_axis_sensitivity(engine, examples, args.axes, args.strengths)}
    else:
        rewrites, generation = generate_repair_data(engine, train, args.axes, config.run.repair_strength,
                                                    config.run.augmentation_ratio, args.seed)
        save_examples(str(output / "augmentation.jsonl"), rewrites)
        save_json(str(output / "augmentation_summary.json"), generation)
        summary = fit(detector, train + rewrites, dev, str(output), config.run.repair_epochs,
                      config.models.batch_size, args.seed, allow_overlap=args.allow_overlap)
        summary["augmentation"] = generation
    if chat is not None:
        summary["llm_usage"] = dict(chat.usage)
        summary["nli_contradiction_index"] = engine.gate.nli.contradiction_index
    save_json(str(output / "summary.json"), summary)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
