"""Тесты таблицы интерпретации гипотез без реальных метрик."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from interpret_results import evaluate_hypotheses, write_results


class InterpretationTests(unittest.TestCase):
    def test_missing_metrics_are_not_presented_as_negative_results(self):
        results = evaluate_hypotheses(None)
        self.assertEqual({item["status"] for item in results}, {"not_evaluable"})
        self.assertEqual({item["hypothesis"] for item in results}, {"H1", "H2", "H3", "H4"})

    def test_synthetic_metrics_use_explicit_hypothesis_statuses(self):
        metrics = {
            "pd": {"specifications": {
                "financials": {"n": 10, "pr_auc": 0.2, "brier": 0.2},
                "sentiment": {"n": 10, "pr_auc": 0.25, "brier": 0.19,
                              "coefficients": [
                                  {"feature": "report_negative_share", "coefficient": 0.2},
                                  {"feature": "report_risk_share", "coefficient": 0.1},
                              ]},
                "combined": {"n": 10, "pr_auc": 0.3, "brier": 0.15, "model_type": "logistic_regression",
                             "coefficient_intervals": {
                                 "report_negative_share": {"lower": 0.05, "upper": 0.35},
                                 "report_risk_share": {"lower": 0.02, "upper": 0.20},
                             },
                             "paired_comparison": {
                                 "pr_auc": {"lower": 0.03, "upper": 0.20},
                                 "brier": {"lower": -0.20, "upper": -0.02},
                             },
                             "coefficients": [
                                 {"feature": "report_negative_share", "coefficient": 0.2},
                                 {"feature": "report_risk_share", "coefficient": 0.1},
                                 {"feature": "ROA", "coefficient": -0.3},
                             ]},
            }},
            "rating": {"status": "ok", "macro_f1": 0.4, "balanced_accuracy": 0.45,
                       "class_counts": {"test": {"downgrade": 2, "stable": 4, "upgrade": 1}}},
        }
        results = evaluate_hypotheses(metrics)
        statuses = {item["hypothesis"]: item["status"] for item in results}
        self.assertEqual(statuses["H1"], "supported")
        self.assertEqual(statuses["H2"], "supported")
        self.assertEqual(statuses["H3"], "inconclusive")
        self.assertEqual(statuses["H4"], "inconclusive")

    def test_results_can_be_written_as_json_and_csv(self):
        with tempfile.TemporaryDirectory() as directory:
            json_path = Path(directory) / "hypotheses.json"
            csv_path = Path(directory) / "hypotheses.csv"
            results = evaluate_hypotheses(None)
            write_results(results, json_path, csv_path)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["protocol_version"], "interpretation-1.0")
            self.assertTrue(csv_path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)