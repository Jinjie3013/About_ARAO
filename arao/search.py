"""Coordinate generation, filtering, scoring and NES; no concrete model imports."""
from __future__ import annotations

from typing import Callable, Optional, Sequence
import numpy as np

from .controller import NESConfig, NESController
from .interfaces import Detector, Planner, Rewriter, SemanticGate
from .objective import negative_log_true_probability, validate_probabilities
from .space import StrategySpace
from .types import Candidate, Example, SearchResult


class ARAO:
    def __init__(self, space: StrategySpace, planner: Planner, rewriter: Rewriter,
                 gate: SemanticGate, detector: Detector, config: NESConfig, seed: int = 42):
        self.space, self.planner, self.rewriter = space, planner, rewriter
        self.gate, self.detector, self.config = gate, detector, config
        self.rng = np.random.default_rng(seed)

    def _start(self, example: Example) -> SearchResult:
        p = validate_probabilities(self.detector.predict_proba([example.text]), 1)
        negative_log_true_probability(p, example.label)  # validate label before generation
        return SearchResult(example, tuple(float(x) for x in p[0]))

    def _evaluate(self, example: Example, strategies: Sequence[np.ndarray], iteration: int) -> list[Candidate]:
        vectors = [self.space.validate(s) for s in strategies]
        guidelines = [self.planner.plan(s) for s in vectors]
        texts = [self.rewriter.rewrite(example.text, r) for r in guidelines]
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Rewriter returned an empty or non-text response.")
        decisions = self.gate.check(example.text, texts)
        if len(decisions) != len(texts):
            raise ValueError("Gate must return one decision per candidate.")
        accepted = [i for i, decision in enumerate(decisions) if decision.passed]
        scores, probabilities = {}, {}
        # Rejected candidates never reach the detector objective or training.
        if accepted:
            p = validate_probabilities(self.detector.predict_proba([texts[i] for i in accepted]), len(accepted))
            losses = negative_log_true_probability(p, example.label)
            for row, i in enumerate(accepted):
                probabilities[i] = tuple(float(v) for v in p[row])
                scores[i] = float(losses[row])
        return [Candidate(iteration, tuple(float(x) for x in s), guideline, text,
                          decision.score, probabilities.get(i, ()), scores.get(i))
                for i, (s, guideline, text, decision) in enumerate(zip(vectors, guidelines, texts, decisions))]

    @staticmethod
    def _record(result: SearchResult, candidates: list[Candidate]) -> list[Candidate]:
        result.candidates.extend(candidates)
        valid = [c for c in candidates if c.objective is not None]
        for c in valid:
            if result.best is None or c.objective > result.best.objective:
                result.best = c
        return valid

    def search(self, example: Example,
               on_valid: Optional[Callable[[list[Example]], None]] = None) -> SearchResult:
        """Fresh distribution per example; exactly N*T candidate attempts.

        For frozen evaluation omit on_valid. In online training it receives ALL
        gate-passing rewrites ONCE per iteration, after the NES update.
        Objective values across online iterations use changing detectors; the
        resulting online trace is not a fixed-detector evaluation artifact.
        """
        result = self._start(example)
        controller = NESController(self.space.dimension, self.config, self.rng)
        for t in range(self.config.iterations):
            valid = self._record(result, self._evaluate(example, controller.sample(), t))
            controller.update(np.array([c.strategy for c in valid]), np.array([c.objective for c in valid]))
            if on_valid is not None and valid:
                on_valid([Example(f"{example.uid}/t{t}/v{i}", c.text, example.label) for i, c in enumerate(valid)])
        # Algorithm 1 returns (mu_{T+1}, Sigma_{T+1}); keep its drift observable.
        result.control_trace = tuple(controller.history)
        # Select an ALREADY evaluated candidate; never generate an extra rewrite
        # from the final mean outside the declared budget.
        return result

    def rewrite_at(self, example: Example, strategy: np.ndarray) -> SearchResult:
        """One fixed intervention. A zero vector still invokes both LLM stages."""
        result = self._start(example)
        self._record(result, self._evaluate(example, [strategy], 0))
        return result
