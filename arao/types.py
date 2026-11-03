"""Shared values; no model, API, training, or filesystem dependencies."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class Example:
    uid: str
    text: str
    label: int


@dataclass(frozen=True)
class GateDecision:
    score: float
    passed: bool


@dataclass(frozen=True)
class Candidate:
    iteration: int
    strategy: tuple[float, ...]
    guideline: str
    text: str
    gate_score: float
    probabilities: tuple[float, ...] = ()
    objective: Optional[float] = None

    @property
    def prediction(self) -> Optional[int]:
        return max(range(len(self.probabilities)), key=self.probabilities.__getitem__) if self.probabilities else None


@dataclass
class SearchResult:
    example: Example
    original_probabilities: tuple[float, ...]
    candidates: list[Candidate] = field(default_factory=list)
    best: Optional[Candidate] = None
    # Per-iteration drift of the search distribution (Algorithm 1's mu_t, Sigma_t).
    control_trace: tuple[dict, ...] = ()

    @property
    def text(self) -> str:
        return self.best.text if self.best is not None else self.example.text

    @property
    def probabilities(self) -> tuple[float, ...]:
        return self.best.probabilities if self.best is not None else self.original_probabilities

    @property
    def clean_correct(self) -> bool:
        p = self.original_probabilities
        return max(range(len(p)), key=p.__getitem__) == self.example.label

    @property
    def strict_success(self) -> bool:
        p = self.probabilities
        return self.clean_correct and max(range(len(p)), key=p.__getitem__) != self.example.label
