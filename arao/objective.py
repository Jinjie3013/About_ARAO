"""Dense adversarial objective; ROC-AUC is used only for evaluation."""
from __future__ import annotations
import numpy as np


def validate_probabilities(values, rows: int) -> np.ndarray:
    p = np.asarray(values, dtype=float)
    if p.ndim != 2 or p.shape[0] != rows or p.shape[1] < 2:
        raise ValueError("Detector must return an (examples, classes) probability matrix.")
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any() or not np.allclose(p.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Detector probabilities must be finite, nonnegative and normalized.")
    return p


def negative_log_true_probability(probabilities: np.ndarray, label: int) -> np.ndarray:
    p = validate_probabilities(probabilities, len(probabilities))
    if not 0 <= label < p.shape[1]:
        raise ValueError("Label is outside the detector class range.")
    return -np.log(np.clip(p[:, label], 1e-12, 1))
