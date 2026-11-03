"""Descriptive alignment, trajectory-matched null, and fixed-axis probes."""
from __future__ import annotations
from typing import Sequence
import numpy as np

from .metrics import classification_metrics
from .search import ARAO
from .space import StrategySpace
from .types import Example, SearchResult


def axis_alignment(results: Sequence[SearchResult], space: StrategySpace,
                   null_draws: int = 50, seed: int = 42) -> dict:
    """One uniformly sampled valid candidate per successful trajectory per draw.

    Sampling includes successful candidates: the null controls for trajectory
    exposure, rather than treating every non-selected rewrite as a failure.
    """
    successful = [r for r in results if r.strict_success and r.best is not None]
    if null_draws < 1:
        raise ValueError("null_draws must be positive.")

    def alignment(strategy):
        s = space.validate(strategy)
        norm = np.linalg.norm(s)
        return np.abs(s / norm) if norm else np.zeros(space.dimension)

    if not successful:
        return {"strict_successes": 0, "axes": {}, "null_draws": null_draws}
    observed = np.mean([alignment(r.best.strategy) for r in successful], axis=0)
    trajectories = [[c for c in r.candidates if c.objective is not None] for r in successful]
    rng = np.random.default_rng(seed)
    null = np.mean([
        np.mean([alignment(cs[int(rng.integers(len(cs)))].strategy) for cs in trajectories], axis=0)
        for _ in range(null_draws)
    ], axis=0)
    return {"strict_successes": len(successful), "null_draws": null_draws,
            "trajectory_gate_passes": sum(len(cs) for cs in trajectories),
            "axes": {axis.name: {"maa": float(observed[i]), "null_maa": float(null[i]),
                                 "enrichment": float(observed[i] - null[i])}
                     for i, axis in enumerate(space.axes)}}


def single_axis_sensitivity(engine: ARAO, examples: Sequence[Example], axis_names: Sequence[str],
                            strengths: Sequence[float]) -> list[dict]:
    rows = []
    for name in axis_names:
        for strength in strengths:
            strategy = engine.space.intervention([name], strength)
            results = [engine.rewrite_at(e, strategy) for e in examples]
            valid = [r for r in results if r.best is not None]
            clean_all = classification_metrics([e.label for e in examples], [r.original_probabilities for r in results])
            clean_paired = classification_metrics([r.example.label for r in valid], [r.original_probabilities for r in valid])
            rewritten = classification_metrics([r.example.label for r in valid], [r.probabilities for r in valid])
            auc, full_auc, paired_auc = rewritten["roc_auc"], clean_all["roc_auc"], clean_paired["roc_auc"]
            rows.append({"axis": name, "strength": float(strength), "n": len(results), "retained": len(valid),
                         "gate_pass_rate": len(valid) / len(results) if results else None,
                         "original_auc_full": full_auc, "original_auc_retained": paired_auc,
                         "rewrite_auc_retained": auc,
                         "delta_auc_paper": full_auc - auc if auc is not None and full_auc is not None else None,
                         "delta_auc_paired": paired_auc - auc if auc is not None and paired_auc is not None else None})
    return rows
