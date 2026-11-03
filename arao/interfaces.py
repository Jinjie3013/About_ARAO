"""Dependency boundaries implemented by local models, APIs, or test doubles."""
from __future__ import annotations

from typing import Protocol, Sequence
import numpy as np

from .types import Example, GateDecision


class ChatModel(Protocol):
    def complete(self, system: str, user: str) -> str: ...


class Planner(Protocol):
    def plan(self, strategy: np.ndarray) -> str: ...


class Rewriter(Protocol):
    def rewrite(self, original: str, guideline: str) -> str: ...


class ContradictionScorer(Protocol):
    def contradiction_probabilities(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray: ...


class SemanticGate(Protocol):
    def check(self, original: str, rewrites: Sequence[str]) -> list[GateDecision]: ...


class Detector(Protocol):
    def predict_proba(self, texts: Sequence[str]) -> np.ndarray: ...


class TrainableDetector(Detector, Protocol):
    def train_step(self, examples: Sequence[Example]) -> float: ...

    def save(self, directory: str) -> None: ...
