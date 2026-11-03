"""Typed experiment settings; JSON holds model names, never API secrets."""
from __future__ import annotations
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Optional

from .controller import NESConfig


@dataclass(frozen=True)
class ModelConfig:
    detector_name: str = "bert-base-uncased"
    num_labels: int = 2
    detector_device: str = "auto"
    nli_name: str = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    nli_device: str = "cpu"
    contradiction_index: Optional[int] = None
    batch_size: int = 8
    max_length: int = 512
    nli_max_length: int = 512
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class LLMConfig:
    model: str = "gpt-4o-mini"
    temperature: float = 0
    top_p: float = 1
    max_tokens: int = 1024
    disable_thinking: bool = False
    api_key_env: str = "OPENAI_API_KEY"
    base_url_env: str = "OPENAI_BASE_URL"


@dataclass(frozen=True)
class RunConfig:
    domain: str = "misinformation"
    gate_threshold: float = 0.85
    epochs: int = 3
    attack_fraction: float = 1.0
    repair_epochs: int = 2
    repair_strength: float = 0.6
    augmentation_ratio: float = 0.25


@dataclass(frozen=True)
class Config:
    models: ModelConfig = field(default_factory=ModelConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    search: NESConfig = field(default_factory=NESConfig)
    run: RunConfig = field(default_factory=RunConfig)

    @classmethod
    def load(cls, path: str) -> Config:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        unknown = set(value) - {"models", "llm", "search", "run"}
        if unknown:
            raise ValueError(f"Unknown configuration sections: {unknown}")
        return cls(ModelConfig(**value.get("models", {})), LLMConfig(**value.get("llm", {})),
                   NESConfig(**value.get("search", {})), RunConfig(**value.get("run", {})))

    def as_dict(self) -> dict:
        return asdict(self)
