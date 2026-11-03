"""Composition root: the only place tying concrete models to the core."""
from __future__ import annotations
from dataclasses import asdict
import random
import numpy as np

from .adapters.huggingface import HFDetector, HFNLI
from .adapters.openai_chat import OpenAIChat
from .config import Config
from .gate import NonContradictionGate
from .generation import LLMPlanner, LLMRewriter
from .search import ARAO
from .space import EXTREMIST_AXES, MISINFORMATION_AXES, StrategySpace


def build_space(config: Config) -> StrategySpace:
    if config.run.domain not in {"misinformation", "extremist"}:
        raise ValueError("domain must be misinformation or extremist.")
    return StrategySpace(MISINFORMATION_AXES if config.run.domain == "misinformation" else EXTREMIST_AXES)


def build_detector(config: Config, checkpoint: str | None, seed: int) -> HFDetector:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    m = config.models
    return HFDetector(checkpoint or m.detector_name, m.num_labels, m.detector_device,
                      m.batch_size, m.max_length, m.learning_rate, m.weight_decay, m.max_grad_norm)


def build_engine(config: Config, detector: HFDetector, seed: int) -> tuple[ARAO, OpenAIChat]:
    space, m = build_space(config), config.models
    chat = OpenAIChat(**asdict(config.llm))
    nli = HFNLI(m.nli_name, m.nli_device, m.batch_size, m.nli_max_length, m.contradiction_index)
    gate = NonContradictionGate(nli, config.run.gate_threshold)
    return ARAO(space, LLMPlanner(space, chat), LLMRewriter(chat), gate, detector, config.search, seed), chat
