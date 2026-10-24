"""Tests for condition-held-out ML evaluation helpers."""

import unittest

import pandas as pd

from ml_pipeline import calculate_strategy_throughputs, cross_validate_conditions


def make_complete_conditions():
    rows = []
    winners = ("reno", "cubic", "bbr")
    for condition in range(30):
        winner = winners[condition % len(winners)]
        throughputs = {"reno": 10.0, "cubic": 10.0, "bbr": 10.0}
        throughputs[winner] = 20.0
        rows.append({
            "bandwidth_mbps": condition + 1,
            "delay_ms": 5 + (condition % 5) * 10,
            "loss_pct": float(condition % 3),
            "baseline_rtt": 10 + condition,
            "reno_tp": throughputs["reno"],
            "cubic_tp": throughputs["cubic"],
            "bbr_tp": throughputs["bbr"],
            "best_throughput": 20.0,
            "best_variant": winner,
        })
    return pd.DataFrame(rows)


class ConditionCrossValidationTests(unittest.TestCase):
    def test_strategy_comparison_requires_one_valid_prediction_per_condition(self):
        data = make_complete_conditions().iloc[:2]
        with self.assertRaises(ValueError):
            calculate_strategy_throughputs(data, ["bbr"])
        with self.assertRaises(ValueError):
            calculate_strategy_throughputs(data, ["bbr", "vegas"])

    def test_cross_validation_returns_out_of_fold_baseline_comparison(self):
        result = cross_validate_conditions(make_complete_conditions(), n_splits=5)
        self.assertEqual(result["n_splits"], 5)
        self.assertEqual(len(result["folds"]), 5)
        self.assertEqual(sum(fold["test_size"] for fold in result["folds"]), 30)
        self.assertIn("ml_adaptive_tp", result["baseline_comparison"])
        self.assertGreaterEqual(result["out_of_fold_accuracy"], 0.0)
        self.assertLessEqual(result["out_of_fold_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
