"""Full-covariance NES updates matching Algorithm 1, lines 13-15.

RANKNORM's normalization constant is unspecified in the paper, so it is an
explicit configuration knob: the scale of the rank utilities multiplies the
effect of ``mean_lr``/``covariance_lr`` (Table 10), and therefore changes the
optimizer's dynamics rather than only its bookkeeping. Each update also logs
the distribution's own drift so the controller's contribution is auditable.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

RANK_NORMS = ("l1", "unit", "raw")


@dataclass(frozen=True)
class NESConfig:
    population_size: int = 16
    iterations: int = 8
    mean_lr: float = 0.03
    covariance_lr: float = 0.02
    initial_std: float = 0.5
    eigenvalue_floor: float = 1e-6
    rank_norm: str = "unit"

    def __post_init__(self):
        if self.population_size < 1 or self.iterations < 1:
            raise ValueError("Population size and iterations must be positive.")
        if not all(np.isfinite(x) and x > 0 for x in (self.mean_lr, self.covariance_lr, self.initial_std, self.eigenvalue_floor)):
            raise ValueError("NES rates, initial standard deviation, and eigenvalue floor must be positive and finite.")
        if self.rank_norm not in RANK_NORMS:
            raise ValueError(f"rank_norm must be one of {RANK_NORMS}.")


def centered_rank_utilities(scores: np.ndarray, norm: str = "unit") -> np.ndarray:
    """Average ranks for ties, center them, then normalize by ``norm``.

    A singleton or an all-tied population supplies no relative information.

    The paper specifies centered ranks but no normalization constant, and that
    constant is exactly the unit in which ``mean_lr`` is expressed: for a
    population of size n, sum(|u|) is n^2/(4(n-1)) under ``unit`` but 1 under
    ``l1``, so ``l1`` silently scales the paper's eta down by that factor
    (4.27x at n=16, and it grows with n). ``unit`` keeps the conventional
    evolution-strategy scale, which is what makes eta=0.03 move the mean:

    ``l1``   sum(|u|) == 1          -> max|u| = 2(n-1)/n^2     (n=16: 0.1172)
    ``unit`` u = rank/(n-1) - 0.5   -> max|u| = 0.5            (n=16: 0.5000)
    ``raw``  u = rank - (n-1)/2     -> max|u| = (n-1)/2        (n=16: 7.5000)

    ``raw`` is offered for ablation only: it saturates the [-1,1] box within a
    few iterations and inflates the covariance, so it is not a default.

    Switching the mode is therefore equivalent to rescaling ``mean_lr`` by
    n^2/(4(n-1)) (``unit``) or n^2/4 (``raw``); at n=16 those factors are 4.27
    and 64. The utilities' direction is identical in all three modes.
    """
    if norm not in RANK_NORMS:
        raise ValueError(f"norm must be one of {RANK_NORMS}.")
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or not np.isfinite(scores).all():
        raise ValueError("Rank utilities require a finite vector.")
    if len(scores) < 2:
        return np.zeros_like(scores)
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    ends = np.cumsum(counts)
    ranks = ((ends - counts + ends - 1) / 2)[inverse]
    u = ranks - (len(scores) - 1) / 2
    if norm == "raw":
        return u
    if norm == "unit":
        return u / (len(scores) - 1)
    scale = np.abs(u).sum()
    return u / scale if scale else u


class NESController:
    def __init__(self, dimension: int, config: NESConfig, rng: np.random.Generator):
        self.config, self.rng = config, rng
        self.mean = np.zeros(dimension)
        self.covariance = np.eye(dimension) * config.initial_std ** 2
        # One row per iteration: the distribution's own drift, not its samples.
        self.history: list[dict] = []

    def sample(self) -> np.ndarray:
        raw = self.rng.multivariate_normal(self.mean, self.covariance, self.config.population_size)
        return np.clip(raw, -1, 1)

    def _record(self, accepted: int, step: float, max_abs_utility: float) -> None:
        self.history.append({
            "iteration": len(self.history),
            "accepted": accepted,
            "max_abs_utility": max_abs_utility,
            "mean_step": step,
            "mean_norm": float(np.linalg.norm(self.mean)),
            "cov_trace": float(np.trace(self.covariance)),
            "cov_min_eig": float(np.linalg.eigvalsh(self.covariance).min()),
        })

    def update(self, strategies: np.ndarray, scores: np.ndarray) -> None:
        """Receive only gate-passing candidates. Use the OLD mean in both updates."""
        if len(scores) == 0:
            # Algorithm 1 line 17: keep (mu, Sigma) unchanged, but stay observable.
            self._record(0, 0.0, 0.0)
            return
        s, m = np.asarray(strategies, dtype=float), np.asarray(scores, dtype=float)
        if s.shape != (len(m), len(self.mean)) or not np.isfinite(s).all():
            raise ValueError("Invalid accepted strategy matrix.")
        u = centered_rank_utilities(m, self.config.rank_norm)
        if not np.any(u):
            # A fully tied population carries no ordering; keep the distribution.
            self._record(len(m), 0.0, float(np.abs(u).max()))
            return
        delta = s - self.mean
        mean_next = self.mean + self.config.mean_lr * (u @ delta)
        outer = delta[:, :, None] * delta[:, None, :]
        covariance_next = self.covariance + self.config.covariance_lr * np.einsum(
            "i,ijk->jk", u, outer - self.covariance,
        )
        # Signed rank updates may leave the positive-definite cone; stabilize.
        eigenvalues, eigenvectors = np.linalg.eigh((covariance_next + covariance_next.T) / 2)
        eigenvalues = np.maximum(eigenvalues, self.config.eigenvalue_floor)
        step = float(np.linalg.norm(mean_next - self.mean))
        self.mean = mean_next
        self.covariance = (eigenvectors * eigenvalues) @ eigenvectors.T
        self._record(len(m), step, float(np.abs(u).max()))
