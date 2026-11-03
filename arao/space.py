"""Theory-grounded axes and bounded interventions (paper Section 3 / Appendix B)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence
import numpy as np


@dataclass(frozen=True)
class Axis:
    name: str
    definition: str
    instruction: str


MISINFORMATION_AXES = (
    Axis("authority", "Reference to credible institutions, experts, or official titles to boost legitimacy.",
         "Inject formal citations or mention expert consensus."),
    Axis("scarcity", "Emphasize limited availability, time constraints, or exclusive access.",
         "Add temporal urgency words, such as limited time or urgent warning."),
    Axis("commitment", "Align with prior beliefs or public stances to enforce consistency.",
         "Frame the argument as a logical continuation of accepted values."),
    Axis("social_proof", "Cite majority opinion, popularity, or collective behavior.",
         "Highlight widespread acceptance or viral status of the claim."),
    Axis("reciprocity", "Create a sense of obligation or mutual benefit.",
         "Frame the information as a special benefit provided to the reader."),
    Axis("liking", "Use relatable, friendly, or attractive personas to build rapport.",
         "Soften the tone to be more conversational and relatable."),
)

# Appendix B, Table 6: verbatim definitions and planner instructions.
EXTREMIST_AXES = (
    Axis("polarization",
         'Constructing a sharp "Us vs. Them" dichotomy to highlight group conflict and incompatibility.',
         "Amplify the contrast between the in-group's virtues and the out-group's vices."),
    Axis("in_group_framing",
         "Strengthening collective identity through shared victimhood, destiny, or superiority cues.",
         "Use inclusive language (e.g., 'our brethren', 'true believers') to foster solidarity."),
    Axis("call_to_action",
         "Urging immediate mobilization, physical action, or retribution against threats.",
         "Inject imperative verbs demanding immediate defense or decisive action."),
    Axis("moral_justification",
         "Legitimizing radical views or actions as virtuous, divine, or ethically necessary.",
         "Frame the narrative as a righteous struggle for justice or higher truth."),
)


@dataclass(frozen=True)
class StrategySpace:
    axes: tuple[Axis, ...] = MISINFORMATION_AXES

    def __post_init__(self):
        if not self.axes or len({a.name for a in self.axes}) != len(self.axes):
            raise ValueError("Axes must be nonempty and uniquely named.")

    @property
    def dimension(self) -> int:
        return len(self.axes)

    def validate(self, strategy: Sequence[float]) -> np.ndarray:
        s = np.asarray(strategy, dtype=float)
        if s.shape != (self.dimension,) or not np.isfinite(s).all() or (np.abs(s) > 1).any():
            raise ValueError(f"Expected a finite strategy in [-1,1]^{self.dimension}.")
        return s

    def intervention(self, names: Sequence[str], strength: float) -> np.ndarray:
        if not names or len(set(names)) != len(names):
            raise ValueError("Specify one or more distinct axis names.")
        indices = {a.name: i for i, a in enumerate(self.axes)}
        unknown = set(names) - indices.keys()
        if unknown:
            raise ValueError(f"Unknown axes: {sorted(unknown)}; available: {list(indices)}")
        s = np.zeros(self.dimension)
        for name in names:
            s[indices[name]] = strength
        return self.validate(s)

    def describe(self, strategy: Sequence[float]) -> str:
        s = self.validate(strategy)
        lines = ["Coordinates lie in [-1,1]: positive strengthens, negative suppresses, zero preserves the original rhetorical tone."]
        for axis, value in zip(self.axes, s):
            lines.append(f"{axis.name}: {value:+.4f}; definition: {axis.definition} Strengthening instruction: {axis.instruction}")
        return "\n".join(lines)
