"""Offline adapter integration using randomly initialized, tiny local BERTs."""
from __future__ import annotations
from pathlib import Path
import tempfile
import unittest
import numpy as np

from arao.adapters.huggingface import HFDetector, HFNLI
from arao.types import Example


class HuggingFaceIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from transformers import BertConfig, BertForSequenceClassification, BertTokenizerFast

        torch.set_num_threads(1)
        cls.torch = torch
        cls.folder = tempfile.TemporaryDirectory(prefix="arao_test_", dir=Path.cwd())
        root = Path(cls.folder.name)
        vocab = root / "vocab.txt"
        vocab.write_text("[PAD]\n[UNK]\n[CLS]\n[SEP]\n[MASK]\nhello\nworld\na\nb\nc\n", encoding="utf-8")
        tokenizer = BertTokenizerFast(vocab_file=str(vocab), do_lower_case=True)
        cls.detector_path, cls.nli_path, cls.unknown_path = [str(root / name) for name in ("detector", "nli", "unknown")]
        for directory, labels in [(cls.detector_path, {0: "negative", 1: "positive"}),
                                  (cls.nli_path, {0: "contradiction", 1: "entailment", 2: "neutral"}),
                                  (cls.unknown_path, {0: "LABEL_0", 1: "LABEL_1", 2: "LABEL_2"})]:
            config = BertConfig(vocab_size=10, hidden_size=16, num_hidden_layers=1,
                                num_attention_heads=2, intermediate_size=32, max_position_embeddings=32,
                                hidden_dropout_prob=0, attention_probs_dropout_prob=0,
                                num_labels=len(labels), id2label=labels, label2id={v: k for k, v in labels.items()})
            BertForSequenceClassification(config).save_pretrained(directory)
            tokenizer.save_pretrained(directory)

    @classmethod
    def tearDownClass(cls):
        cls.folder.cleanup()

    def test_detector_probability_and_mode_restore(self):
        detector = HFDetector(self.detector_path, device="cpu", max_length=16)
        detector.model.train()
        p = detector.predict_proba(["hello world", "hello " * 100])
        np.testing.assert_allclose(p.sum(1), 1, atol=1e-6)
        self.assertTrue(detector.model.training)
        self.assertEqual(p.shape, (2, 2))

    def test_microbatch_is_one_mean_loss_optimizer_step(self):
        full = HFDetector(self.detector_path, device="cpu", batch_size=7, max_length=16, max_grad_norm=100)
        micro = HFDetector(self.detector_path, device="cpu", batch_size=2, max_length=16, max_grad_norm=100)
        examples = [Example(str(i), "hello world" if i % 2 else "a b c", i % 2) for i in range(7)]
        full_loss, micro_loss = full.train_step(examples), micro.train_step(examples)
        self.assertAlmostEqual(full_loss, micro_loss, places=6)
        for p, q in zip(full.model.parameters(), micro.model.parameters()):
            self.torch.testing.assert_close(p, q, atol=1e-6, rtol=1e-5)
        self.assertFalse(micro.model.training)

    def test_nli_uses_label_names_not_fixed_index(self):
        nli = HFNLI(self.nli_path, device="cpu", max_length=16)
        self.assertEqual(nli.contradiction_index, 0)
        probabilities = nli.contradiction_probabilities(["hello"], ["world"])
        self.assertEqual(probabilities.shape, (1,))
        self.assertTrue((probabilities >= 0).all() and (probabilities <= 1).all())
        with self.assertRaises(ValueError):
            HFNLI(self.unknown_path, device="cpu", max_length=16)
        self.assertEqual(HFNLI(self.unknown_path, contradiction_index=2).contradiction_index, 2)


if __name__ == "__main__":
    unittest.main()
