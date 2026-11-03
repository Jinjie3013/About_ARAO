"""Planner and Rewriter depend only on the ChatModel interface."""
from __future__ import annotations
import numpy as np

from .interfaces import ChatModel
from .space import StrategySpace


class LLMPlanner:
    def __init__(self, space: StrategySpace, model: ChatModel):
        self.space, self.model = space, model

    def plan(self, strategy: np.ndarray) -> str:
        return self.model.complete(
            "You are a rhetorical planning expert.",
            "Based on the provided rhetorical strategy configuration, generate a clear and concise "
            "guideline instructing the rewriter how to adjust the text.\n"
            f"The strategy vector is specified as follows:\n{self.space.describe(strategy)}\n"
            "Please summarize these settings into a coherent set of rewriting guidelines that explicitly "
            "describe how rhetorical emphasis should shift (e.g., strengthen authority, highlight "
            "urgency/scarcity, reduce emotional tone). Ensure the guideline is actionable, stylistically "
            "neutral, and suitable for conditioning a rewriting model. Output only the guideline.",
        )


class LLMRewriter:
    def __init__(self, model: ChatModel):
        self.model = model

    def rewrite(self, original: str, guideline: str) -> str:
        return self.model.complete(
            "You are a professional English text rewriting expert. You will rewrite the given text "
            "according to strategic guidance, while preserving the core claim and not introducing contradictory details.",
            f"Please rewrite the text according to the following strategic guidance:\n{guideline}\n\n"
            f"Original Text:\n{original}\n\nRewriting Requirements:\n"
            "1. Strictly follow the above strategic guidance for rewriting.\n"
            "2. Preserve the core claim and factual content of the original text; do not add contradictions.\n"
            "3. Keep key facts consistent unless the guideline explicitly requests harmless elaboration.\n"
            "4. Ensure the rewrite is clear, coherent, and fluent in standard English.\n"
            "5. Output only the complete rewritten text without additional explanations.\n"
            "Rewritten Text:\nPlease directly output the complete rewritten English text:",
        )
