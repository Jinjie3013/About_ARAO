"""PyTorch/Hugging Face implementations of the detector and NLI interfaces."""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Sequence
import numpy as np

from ..types import Example


class _SequenceClassifier:
    def __init__(self, model_name: str, device: str = "auto", batch_size: int = 8,
                 max_length: int = 512, num_labels: Optional[int] = None):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if batch_size < 1 or max_length < 2:
            raise ValueError("Invalid model batch size or maximum length.")
        self.torch = torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.batch_size, self.max_length = batch_size, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        kwargs = {"num_labels": num_labels} if num_labels is not None else {}
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, **kwargs).to(self.device)
        self.model.eval()

    def _encode(self, texts, pairs=None):
        return self.tokenizer(list(texts), text_pair=list(pairs) if pairs is not None else None,
                              padding=True, truncation=True, max_length=self.max_length,
                              return_tensors="pt").to(self.device)

    def _predict(self, texts: Sequence[str], pairs: Optional[Sequence[str]] = None) -> np.ndarray:
        if pairs is not None and len(texts) != len(pairs):
            raise ValueError("Premise/hypothesis lengths differ.")
        if not len(texts):
            return np.empty((0, self.model.config.num_labels))
        was_training, outputs = self.model.training, []
        self.model.eval()
        try:
            with self.torch.inference_mode():
                for start in range(0, len(texts), self.batch_size):
                    end = start + self.batch_size
                    encoded = self._encode(texts[start:end], pairs[start:end] if pairs is not None else None)
                    outputs.append(self.model(**encoded).logits.softmax(-1).cpu().numpy())
        finally:
            self.model.train(was_training)
        return np.concatenate(outputs)


class HFDetector(_SequenceClassifier):
    def __init__(self, model_name: str, num_labels: int = 2, device: str = "auto",
                 batch_size: int = 8, max_length: int = 512, learning_rate: float = 2e-5,
                 weight_decay: float = 0.01, max_grad_norm: float = 1.0):
        super().__init__(model_name, device, batch_size, max_length, num_labels)
        self.learning_rate, self.weight_decay = learning_rate, weight_decay
        self.max_grad_norm, self.optimizer = max_grad_norm, None

    def predict_proba(self, texts: Sequence[str]) -> np.ndarray:
        return self._predict(texts)

    def train_step(self, examples: Sequence[Example]) -> float:
        """One optimizer step on the exact MEAN loss of ALL given rewrites.

        Microbatching changes memory use, not the averaging weights or the
        number of optimizer steps for an accepted population.
        """
        if not len(examples):
            raise ValueError("Cannot train on an empty population.")
        if self.optimizer is None:
            self.optimizer = self.torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        was_training = self.model.training
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        try:
            for start in range(0, len(examples), self.batch_size):
                batch = examples[start:start + self.batch_size]
                encoded = self._encode([e.text for e in batch])
                labels = self.torch.tensor([e.label for e in batch], dtype=self.torch.long, device=self.device)
                loss = self.model(**encoded, labels=labels).loss * (len(batch) / len(examples))
                loss.backward()
                total_loss += float(loss.detach())
            self.torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()
        finally:
            self.model.train(was_training)
        return total_loss

    def save(self, directory: str) -> None:
        Path(directory).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(directory, safe_serialization=True)
        self.tokenizer.save_pretrained(directory)


class HFNLI(_SequenceClassifier):
    def __init__(self, model_name: str, device: str = "cpu", batch_size: int = 8,
                 max_length: int = 512, contradiction_index: Optional[int] = None):
        super().__init__(model_name, device, batch_size, max_length)
        if contradiction_index is None:
            matches = [int(i) for i, name in self.model.config.id2label.items() if "contradiction" in name.lower()]
            if len(matches) != 1:
                raise ValueError("NLI id2label does not identify contradiction. Set contradiction_index explicitly; do not guess LABEL_0/1/2.")
            contradiction_index = matches[0]
        if not 0 <= contradiction_index < self.model.config.num_labels:
            raise ValueError("contradiction_index is outside the NLI label range.")
        self.contradiction_index = contradiction_index

    def contradiction_probabilities(self, premises: Sequence[str], hypotheses: Sequence[str]) -> np.ndarray:
        return self._predict(premises, hypotheses)[:, self.contradiction_index]
