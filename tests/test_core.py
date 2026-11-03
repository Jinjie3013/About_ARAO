"""Behavioral checks using test doubles: no network, weights, or LLM fees."""
from __future__ import annotations
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np

from arao.controller import centered_rank_utilities, NESConfig, NESController
from arao.data import load_examples, load_results, require_disjoint, save_results
from arao.diagnosis import axis_alignment
from arao.gate import NonContradictionGate
from arao.metrics import attack_metrics, classification_metrics
from arao.objective import negative_log_true_probability
from arao.search import ARAO
from arao.space import StrategySpace
from arao.training import generate_repair_data
from arao.types import Example, GateDecision


class Planner:
    def __init__(self):
        self.strategies = []

    def plan(self, strategy):
        self.strategies.append(strategy.copy())
        return json.dumps(strategy.tolist())


class Rewriter:
    def __init__(self):
        self.calls = 0

    def rewrite(self, original, guideline):
        self.calls += 1
        return f"rewrite-{self.calls}"


class Gate:
    def __init__(self, reject_all=False):
        self.reject_all = reject_all

    def check(self, original, rewrites):
        return [GateDecision(0.1, False) if self.reject_all or int(t.split("-")[-1]) % 3 == 0
                else GateDecision(0.95, True) for t in rewrites]


class Detector:
    def __init__(self):
        self.seen = []

    def predict_proba(self, texts):
        self.seen.extend(texts)
        return np.array([[1 - 1 / (int(t.split("-")[-1]) + 2), 1 / (int(t.split("-")[-1]) + 2)]
                         if t.startswith("rewrite-") else [0.1, 0.9] for t in texts])


def make_engine(reject_all=False):
    return ARAO(StrategySpace(), Planner(), Rewriter(), Gate(reject_all), Detector(),
                NESConfig(population_size=3, iterations=2), seed=7)


class GateTests(unittest.TestCase):
    def test_worst_direction_and_pair_order(self):
        class NLI:
            def contradiction_probabilities(self, premises, hypotheses):
                self.pairs = list(zip(premises, hypotheses))
                return np.array([0.10, 0.05, 0.20, 0.02])
        nli = NLI()
        decisions = NonContradictionGate(nli).check("original", ["a", "b"])
        self.assertEqual(nli.pairs, [("original", "a"), ("original", "b"), ("a", "original"), ("b", "original")])
        np.testing.assert_allclose([d.score for d in decisions], [0.8, 0.95])
        self.assertEqual([d.passed for d in decisions], [False, True])

    def test_invalid_nli_cannot_pass(self):
        class NLI:
            def contradiction_probabilities(self, premises, hypotheses):
                return np.array([np.nan, 0.0])
        with self.assertRaises(ValueError):
            NonContradictionGate(NLI()).check("x", ["y"])


class NESTests(unittest.TestCase):
    def test_centered_ranks_and_ties(self):
        np.testing.assert_allclose(centered_rank_utilities([1, 1, 4]), [-0.25, -0.25, 0.5])
        np.testing.assert_array_equal(centered_rank_utilities([2, 2, 2]), [0, 0, 0])
        np.testing.assert_array_equal(centered_rank_utilities([8]), [0])

    def test_updates_match_paper_using_old_mean(self):
        config = NESConfig()
        controller = NESController(2, config, np.random.default_rng(1))
        controller.mean = np.array([0.2, -0.1])
        before_mean, before_cov = controller.mean.copy(), controller.covariance.copy()
        strategies, scores = np.array([[1, 0], [-1, 0], [0, 1]]), np.array([3, 2, 1])
        u = np.array([0.5, 0, -0.5])
        delta = strategies - before_mean
        expected_cov = before_cov + config.covariance_lr * sum(
            ui * (np.outer(d, d) - before_cov) for ui, d in zip(u, delta))
        controller.update(strategies, scores)
        np.testing.assert_allclose(controller.mean, before_mean + config.mean_lr * (u @ delta))
        np.testing.assert_allclose(controller.covariance, expected_cov)

    def test_empty_and_singleton_leave_distribution_unchanged(self):
        c = NESController(2, NESConfig(), np.random.default_rng(1))
        cov = c.covariance.copy()
        c.update(np.array([]), np.array([]))
        c.update(np.array([[1, 1]]), np.array([10]))
        np.testing.assert_array_equal(c.mean, [0, 0])
        np.testing.assert_array_equal(c.covariance, cov)

    def test_covariance_stays_positive_and_samples_bounded(self):
        c = NESController(2, NESConfig(covariance_lr=10), np.random.default_rng(1))
        c.update(np.array([[1, 0], [0, 1]]), np.array([2, 1]))
        self.assertGreaterEqual(np.linalg.eigvalsh(c.covariance).min(), 0.999e-6)
        self.assertTrue((np.abs(c.sample()) <= 1).all())

    def test_rank_norm_modes_differ_only_by_scale(self):
        scores = np.arange(16, dtype=float)
        l1, unit, raw = (centered_rank_utilities(scores, m) for m in ("l1", "unit", "raw"))
        self.assertAlmostEqual(float(np.abs(l1).max()), 2 * 15 / 256, places=6)
        self.assertAlmostEqual(float(np.abs(unit).max()), 0.5, places=9)
        self.assertAlmostEqual(float(np.abs(raw).max()), 7.5, places=9)
        for variant, factor in ((unit, 256 / 60), (raw, 64.0)):
            np.testing.assert_allclose(variant, l1 * factor, rtol=1e-9, atol=1e-12)
        for m in ("unit", "raw"):
            self.assertAlmostEqual(float(centered_rank_utilities(scores, m).sum()), 0.0, places=9)
        with self.assertRaises(ValueError):
            NESConfig(rank_norm="rms")
        with self.assertRaises(ValueError):
            centered_rank_utilities(scores, "rms")

    def test_rank_norm_unit_moves_mean_further(self):
        strategies = np.repeat(np.eye(2), 8, axis=0) - 0.5
        scores = np.arange(16, dtype=float)
        moved = {}
        for mode in ("l1", "unit"):
            c = NESController(2, NESConfig(rank_norm=mode), np.random.default_rng(0))
            c.update(strategies, scores)
            moved[mode] = float(np.linalg.norm(c.mean))
        self.assertAlmostEqual(moved["unit"] / moved["l1"], 256 / 60, places=3)

    def test_control_trace_logs_drift_every_iteration(self):
        c = NESController(2, NESConfig(population_size=4, iterations=2), np.random.default_rng(1))
        c.update(np.array([]), np.array([]))
        c.update(np.array([[1, 0], [0, 1]]), np.array([2, 1]))
        self.assertEqual(len(c.history), 2)
        self.assertEqual(c.history[0]["accepted"], 0)
        self.assertEqual(c.history[0]["mean_norm"], 0.0)
        self.assertEqual(c.history[1]["accepted"], 2)
        self.assertIn("cov_min_eig", c.history[1])
        self.assertGreater(c.history[1]["mean_step"], 0.0)

    def test_shipped_default_actually_adapts_at_paper_budget(self):
        """The default normalization must move the search centre, not just rank noise."""
        config = NESConfig()
        self.assertEqual(config.rank_norm, "unit")
        direction = np.array([1.0, 0.0, 0.0, 0.0, 0.0, -1.0]) / np.sqrt(2)
        controller = NESController(6, config, np.random.default_rng(0))
        for _ in range(config.iterations):
            strategies = controller.sample()
            controller.update(strategies, strategies @ direction)
        drift = float(np.linalg.norm(controller.mean))
        self.assertGreater(drift, 0.3, "controller should travel a meaningful share of the box")
        self.assertLess(float(np.mean(np.abs(controller.mean) > 0.99)), 0.2,
                        "controller should not be pinned to the boundary")


class SearchTests(unittest.TestCase):
    def test_budget_filtering_best_candidate_and_all_valid_training(self):
        engine = make_engine()
        batches = []
        result = engine.search(Example("ex", "original", 1), on_valid=batches.append)
        self.assertEqual(engine.rewriter.calls, 6)
        self.assertEqual(len(result.candidates), 6)
        self.assertEqual(result.best.text, "rewrite-5")
        self.assertTrue(result.strict_success)
        self.assertEqual([[e.text for e in b] for b in batches], [["rewrite-1", "rewrite-2"], ["rewrite-4", "rewrite-5"]])
        self.assertNotIn("rewrite-3", engine.detector.seen)
        self.assertNotIn("rewrite-6", engine.detector.seen)
        self.assertIsNone(result.candidates[2].objective)
        self.assertEqual(len(result.control_trace), 2)
        self.assertEqual([row["accepted"] for row in result.control_trace], [2, 2])

    def test_all_rejected_retains_original_and_never_trains(self):
        engine, batches = make_engine(True), []
        result = engine.search(Example("ex", "original", 1), on_valid=batches.append)
        self.assertIsNone(result.best)
        self.assertEqual(result.text, "original")
        self.assertEqual(engine.detector.seen, ["original"])
        self.assertEqual(batches, [])

    def test_zero_intervention_still_rewrites(self):
        engine = make_engine()
        result = engine.rewrite_at(Example("ex", "original", 1), np.zeros(6))
        self.assertEqual(len(result.candidates), 1)
        np.testing.assert_array_equal(engine.planner.strategies[0], np.zeros(6))

    def test_seeded_search_is_reproducible(self):
        example = Example("ex", "original", 1)
        self.assertEqual(make_engine().search(example), make_engine().search(example))

    def test_repair_replenishes_gate_failures_to_fixed_quota(self):
        engine = make_engine()
        train = [Example(str(i), "original", 1) for i in range(16)]
        rewrites, stats = generate_repair_data(engine, train, ["authority"], augmentation_ratio=0.25)
        self.assertEqual(len(rewrites), 4)
        self.assertEqual(stats["attempts"], 5)
        self.assertEqual(stats["realized_augmentation_ratio"], 0.25)
        with self.assertRaises(RuntimeError):
            generate_repair_data(make_engine(True), train, ["authority"])


class DataAndEvaluationTests(unittest.TestCase):
    def test_nll_uses_true_class_and_is_finite(self):
        losses = negative_log_true_probability(np.array([[0.8, 0.2], [1.0, 0.0]]), 1)
        self.assertAlmostEqual(losses[0], -np.log(0.2))
        self.assertTrue(np.isfinite(losses).all())

    def test_trace_round_trip_and_trajectory_null(self):
        result = make_engine().search(Example("ex", "original", 1))
        with tempfile.TemporaryDirectory(prefix="arao_test_", dir=Path.cwd()) as folder:
            path = str(Path(folder) / "trace.jsonl")
            save_results(path, [result])
            self.assertEqual(load_results(path), [result])
        summary = axis_alignment([result], StrategySpace())
        self.assertEqual(summary["strict_successes"], 1)
        self.assertEqual(summary["trajectory_gate_passes"], 4)
        self.assertEqual(set(summary["axes"]), {a.name for a in StrategySpace().axes})

    def test_missing_data_never_substitutes_synthetic_samples(self):
        with self.assertRaises(FileNotFoundError):
            load_examples("does-not-exist.csv")

    def test_title_column_and_strict_labels(self):
        with tempfile.TemporaryDirectory(prefix="arao_test_", dir=Path.cwd()) as folder:
            path = Path(folder) / "data.csv"
            path.write_text("title,label\nhello,1\nworld,0\n", encoding="utf-8")
            self.assertEqual([e.text for e in load_examples(str(path))], ["hello", "world"])
            path.write_text("text,label\nhello,invalid\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_examples(str(path))

    def test_split_overlap_rejected(self):
        with self.assertRaises(ValueError):
            require_disjoint([Example("a", "Hello WORLD", 1)], [Example("b", "hello   world", 1)])

    def test_auc_and_asr_denominators(self):
        result = make_engine().search(Example("ex", "original", 1))
        originally_wrong = make_engine().search(Example("ex2", "original", 0))
        report = attack_metrics([result, originally_wrong])
        self.assertEqual(report["clean_correct"], 1)
        self.assertEqual(report["strict_successes"], 1)
        self.assertEqual(report["asr"], 1)
        self.assertIsNone(classification_metrics([1], [[0.1, 0.9]])["roc_auc"])
        multi = classification_metrics([0, 1, 2], [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]])
        self.assertEqual(multi["roc_auc"], 1)


if __name__ == "__main__":
    unittest.main()
