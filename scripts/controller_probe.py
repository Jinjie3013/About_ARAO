"""Diagnose whether the NES controller actually adapts: no LLM, no detector.

Algorithm 1 leaves RANKNORM's normalization constant undefined, and that
constant is the unit in which ``mean_lr`` is expressed. This probe makes the
consequence measurable on a synthetic vulnerability direction: how far the
search centre travels, whether later iterations sample better than earlier ones,
and whether the mean saturates the strategy box.

    python scripts/controller_probe.py
    python scripts/controller_probe.py --norm unit l1 --mean-lr 0.03 0.1
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from arao.controller import NESConfig, NESController  # noqa: E402

DIMENSION = 6


def trial(norm: str, mean_lr: float, population: int, iterations: int, seed: int) -> dict:
    """Noisy linear objective along a random unit direction inside the box."""
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=DIMENSION)
    direction /= np.linalg.norm(direction)
    config = NESConfig(population_size=population, iterations=iterations,
                       mean_lr=mean_lr, rank_norm=norm)
    controller = NESController(DIMENSION, config, rng)
    per_iteration = []
    for _ in range(iterations):
        strategies = controller.sample()
        scores = strategies @ direction + rng.normal(0, 0.10, size=population)
        per_iteration.append(float(scores.mean()))
        controller.update(strategies, scores)
    return {
        "mean_norm": float(np.linalg.norm(controller.mean)),
        "gain": per_iteration[-1] - per_iteration[0],
        "saturated": float(np.mean(np.abs(controller.mean) > 0.99)),
        "sigma": float(np.sqrt(np.trace(controller.covariance) / DIMENSION)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--norm", nargs="+", default=["unit", "l1", "raw"])
    parser.add_argument("--mean-lr", nargs="+", type=float, default=[0.03])
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=8)
    parser.add_argument("--seeds", type=int, default=40)
    args = parser.parse_args()

    print(f"population={args.population} iterations={args.iterations} "
          f"budget={args.population * args.iterations} seeds={args.seeds} "
          f"(sampling sigma=0.5, box half-width=1.0)")
    header = f"{'norm':<6}{'mean_lr':>9}{'||mu_T||':>10}{'gain':>9}{'saturated':>11}{'sigma_T':>9}"
    print(header)
    print("-" * len(header))
    for norm in args.norm:
        for mean_lr in args.mean_lr:
            rows = [trial(norm, mean_lr, args.population, args.iterations, seed)
                    for seed in range(args.seeds)]
            agg = {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
            print(f"{norm:<6}{mean_lr:>9.2f}{agg['mean_norm']:>10.3f}{agg['gain']:>9.3f}"
                  f"{agg['saturated']:>11.2%}{agg['sigma']:>9.3f}")
    print("\n||mu_T|| is how far the search centre travelled (0 = controller inert,"
          " 1.0 = one box half-width). 'gain' is the mean objective of the last"
          " iteration minus the first; positive means the distribution moved uphill.")


if __name__ == "__main__":
    main()
