"""Training orchestration; optimizer and tensors belong to the detector adapter."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence
import numpy as np

from .data import require_disjoint, save_json
from .interfaces import TrainableDetector
from .metrics import classification_metrics
from .search import ARAO
from .types import Example


def fit(detector: TrainableDetector, train: Sequence[Example], dev: Sequence[Example],
        output: str, epochs: int = 3, batch_size: int = 8, seed: int = 42,
        engine: Optional[ARAO] = None, attack_fraction: float = 1.0,
        allow_overlap: bool = False) -> dict:
    """Clean updates plus optional per-example online Algorithm 1 updates.

    attack_fraction is the fraction of original examples selected per clean
    batch. Positive fractions use ceil; zero performs only clean training.
    Only the DEV split selects checkpoints. Test data never enters this API.
    """
    if not train or not dev or epochs < 1 or batch_size < 1 or not 0 <= attack_fraction <= 1:
        raise ValueError("Invalid training sizes, epochs, batch size or attack fraction.")
    if engine is not None and engine.detector is not detector:
        raise ValueError("Online search and training must share the same detector instance.")
    require_disjoint(train, dev, allow_overlap)
    rng, history, best_auc = np.random.default_rng(seed), [], -float("inf")
    for epoch in range(epochs):
        order = rng.permutation(len(train))
        clean_losses, adversarial_losses, accepted = [], [], 0

        def update_valid(rewrites: list[Example]) -> None:
            nonlocal accepted
            accepted += len(rewrites)
            adversarial_losses.append(detector.train_step(rewrites))

        for start in range(0, len(order), batch_size):
            batch = [train[int(i)] for i in order[start:start + batch_size]]
            clean_losses.append(detector.train_step(batch))
            if engine is not None and attack_fraction > 0:
                n = min(len(batch), int(np.ceil(len(batch) * attack_fraction)))
                for i in rng.choice(len(batch), size=n, replace=False):
                    engine.search(batch[int(i)], on_valid=update_valid)
        dev_metrics = classification_metrics([e.label for e in dev], detector.predict_proba([e.text for e in dev]))
        auc = dev_metrics["roc_auc"]
        if auc is None:
            raise ValueError("Dev ROC-AUC is undefined: the dev set must contain all detector classes.")
        if auc > best_auc:
            best_auc = auc
            detector.save(str(Path(output) / "best"))
        row = {"epoch": epoch + 1, "clean_loss": float(np.mean(clean_losses)),
               "adversarial_loss": float(np.mean(adversarial_losses)) if adversarial_losses else None,
               "accepted_training_rewrites": accepted, "dev": dev_metrics}
        history.append(row)
        save_json(str(Path(output) / "history.json"), history)
        print(f"epoch={epoch + 1} dev_auc={auc:.6f} accepted_rewrites={accepted}", flush=True)
    return {"best_checkpoint": str(Path(output) / "best"), "best_dev_auc": best_auc, "history": history}


def generate_repair_data(engine: ARAO, train: Sequence[Example], axes: Sequence[str],
                         strength: float = 0.6, augmentation_ratio: float = 0.25,
                         seed: int = 42) -> tuple[list[Example], dict]:
    """Generate a fixed training-only augmentation set WITHOUT NES search.

    Ratio means requested additional rewrites / original training examples.
    Try distinct training sources in random order until the accepted quota is
    reached. Each source is attempted at most once. Raise if the quota cannot
    be filled, rather than silently reducing the augmentation ratio.
    """
    if not train or not 0 < augmentation_ratio <= 1:
        raise ValueError("Repair requires training examples and ratio in (0,1].")
    s = engine.space.intervention(axes, strength)
    target = int(np.ceil(len(train) * augmentation_ratio))
    indices = np.random.default_rng(seed).permutation(len(train))
    rewrites, attempts = [], 0
    for i in indices:
        attempts += 1
        example = train[int(i)]
        result = engine.rewrite_at(example, s)
        if result.best is not None:
            rewrites.append(Example(f"{example.uid}/repair", result.text, example.label))
        if len(rewrites) == target:
            break
    if len(rewrites) < target:
        raise RuntimeError(f"Repair quota not reached: retained {len(rewrites)}/{target} after {attempts} distinct sources.")
    return rewrites, {"axes": list(axes), "strength": strength, "source_examples": len(train),
                      "attempts": attempts, "retained": len(rewrites), "requested_augmentation_ratio": augmentation_ratio,
                      "realized_augmentation_ratio": len(rewrites) / len(train)}
