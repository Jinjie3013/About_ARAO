"""Bidirectional non-contradiction gate, Equation (4)."""
from __future__ import annotations
from typing import Sequence
import numpy as np

from .interfaces import ContradictionScorer
from .types import GateDecision


class NonContradictionGate:
    def __init__(self, nli: ContradictionScorer, threshold: float = 0.85):
        if not 0 <= threshold <= 1:
            raise ValueError("Gate threshold must lie in [0,1].")
        self.nli, self.threshold = nli, threshold

    def check(self, original: str, rewrites: Sequence[str]) -> list[GateDecision]:
        n = len(rewrites)
        if n == 0:
            return []
        rewrites = list(rewrites)
        risks = np.asarray(self.nli.contradiction_probabilities(
            [original] * n + rewrites, rewrites + [original] * n,
        ), dtype=float)
        if risks.shape != (2 * n,) or not np.isfinite(risks).all() or ((risks < 0) | (risks > 1)).any():
            raise ValueError("NLI must return one finite contradiction probability per ordered pair.")
        scores = 1 - risks.reshape(2, n).max(axis=0)
        return [GateDecision(float(s), bool(s >= self.threshold)) for s in scores]
