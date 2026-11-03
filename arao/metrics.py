"""Task metrics with explicit undefined-AUC handling."""
from __future__ import annotations
from typing import Sequence
import numpy as np

from .objective import validate_probabilities
from .types import SearchResult


def classification_metrics(labels: Sequence[int], probabilities) -> dict:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    if not len(labels):
        return {"n": 0, "accuracy": None, "macro_f1": None, "roc_auc": None}
    p = validate_probabilities(probabilities, len(labels))
    y = np.asarray(labels, dtype=int)
    classes = p.shape[1]
    if ((y < 0) | (y >= classes)).any():
        raise ValueError("Evaluation label outside the detector class range.")
    auc = None
    if len(np.unique(y)) == classes:
        auc = roc_auc_score(y, p[:, 1]) if classes == 2 else roc_auc_score(
            y, p, labels=np.arange(classes), multi_class="ovr", average="macro",
        )
    return {"n": len(y), "accuracy": float(accuracy_score(y, p.argmax(axis=1))),
            "macro_f1": float(f1_score(y, p.argmax(axis=1), labels=np.arange(classes), average="macro", zero_division=0)),
            "roc_auc": float(auc) if auc is not None else None}


def attack_metrics(results: Sequence[SearchResult]) -> dict:
    y = [r.example.label for r in results]
    eligible = sum(r.clean_correct for r in results)
    success = sum(r.strict_success for r in results)
    attempted = sum(len(r.candidates) for r in results)
    accepted = sum(c.objective is not None for r in results for c in r.candidates)
    return {
        "original": classification_metrics(y, [r.original_probabilities for r in results]),
        "attack": classification_metrics(y, [r.probabilities for r in results]),
        "clean_correct": eligible, "strict_successes": success,
        "asr": success / eligible if eligible else None,
        "candidate_attempts": attempted, "gate_passes": accepted,
        "gate_pass_rate": accepted / attempted if attempted else None,
        "original_fallbacks": sum(r.best is None for r in results),
    }
