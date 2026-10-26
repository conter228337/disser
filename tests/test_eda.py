"""Тесты описательной статистики без внешних данных и графических зависимостей."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eda import event_comparison, grouped_target_summary, run_eda, target_summary, yearly_table
from model_evaluation import bootstrap_emitter_intervals
from run_experiments import _rating_class, run_pd, run_rating


class EdaTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame([
            {"inn": "001", "publication_date": "2020-04-01", "default_12m": 0,
             "default_24m": 0, "rating_target": 0, "ROA": 0.1, "report_negative_share": 0.01,
             "report_word_count": 100, "report_ocr_pages": 0},
            {"inn": "002", "publication_date": "2021-05-01", "default_12m": 1,
             "default_24m": 1, "rating_target": 1, "ROA": -0.1, "report_negative_share": 0.2,
             "report_word_count": 200, "report_ocr_pages": 2},
            {"inn": "001", "publication_date": "2021-06-01", "default_12m": None,
             "default_24m": None, "rating_target": None, "ROA": None, "report_negative_share": None,
             "report_word_count": None, "report_ocr_pages": None},
        ])

    def test_targets_keep_censoring_out_of_event_rate(self):
        summary = target_summary(self.frame).set_index("target")
        self.assertEqual(summary.loc["default_12m", "n_observed"], 2)
        self.assertEqual(summary.loc["default_12m", "n_events"], 1)
        self.assertAlmostEqual(summary.loc["default_12m", "event_rate_observed"], 0.5)

    def test_observation_flag_excludes_malformed_censored_label(self):
        frame = self.frame.copy()
        frame["default_12m_observed"] = [1, 1, 0]
        frame.loc[2, "default_12m"] = 0
        summary = target_summary(frame).set_index("target")
        self.assertEqual(summary.loc["default_12m", "n_observed"], 2)
        self.assertEqual(summary.loc["default_12m", "n_events"], 1)

    def test_invalid_target_count_uses_raw_value_before_observation_mask(self):
        frame = self.frame.copy()
        frame["default_12m_observed"] = [1, 1, 0]
        frame.loc[2, "default_12m"] = 7
        summary = target_summary(frame).set_index("target")
        self.assertEqual(summary.loc["default_12m", "n_invalid_target"], 1)

    def test_yearly_table_counts_documents_and_companies(self):
        result = yearly_table(self.frame).set_index("year")
        self.assertEqual(result.loc[2021, "n_documents"], 2)
        self.assertEqual(result.loc[2021, "n_companies"], 2)
        self.assertEqual(result.loc[2021, "default_12m_n_observed"], 1)

    def test_event_comparison_has_both_groups(self):
        result = event_comparison(self.frame)
        roa = result[result.feature.eq("ROA")]
        self.assertEqual(set(roa.group), {"event", "no_event"})

    def test_industry_summary_is_optional_and_explicit(self):
        frame = self.frame.assign(industry=["industry_a", "industry_b", "industry_a"])
        result = grouped_target_summary(frame)
        self.assertEqual(set(result["group"]), {"industry_a", "industry_b"})
        self.assertEqual(set(result["target"]), {"default_12m", "default_24m", "rating_target"})

    def test_run_writes_tables_and_svg(self):
        with tempfile.TemporaryDirectory() as directory:
            metadata = run_eda(self.frame, directory)
            self.assertEqual(metadata["n_rows"], 3)
            self.assertTrue((Path(directory) / "missingness.csv").exists())
            self.assertTrue((Path(directory) / "documents_by_year.svg").exists())
            self.assertTrue((Path(directory) / "eda_summary.json").exists())

    def test_missing_rating_transition_is_not_stable(self):
        self.assertEqual(_rating_class(-2), -1)
        self.assertEqual(_rating_class(0), 0)
        self.assertEqual(_rating_class(2), 1)
        self.assertTrue(pd.isna(_rating_class(float("nan"))))

    def test_rating_model_emits_named_directional_classes(self):
        rows = []
        directions = [-1, 0, 1]
        feature_names = ["ROA", "ROE", "Debt_to_EBITDA", "Total_Liabilities_to_Assets",
                         "Current_Ratio", "Cash_to_Assets", "Ln_Total_Assets", "total_assets",
                         "equity", "revenue", "net_income", "report_positive_share",
                         "report_negative_share", "report_neutral_share", "report_risk_share",
                         "report_uncertainty_share", "report_sentiment_score", "report_word_count",
                         "report_ocr_pages"]
        for index in range(18):
            row = {"publication_date": pd.Timestamp("2021-01-01" if index < 12 else "2023-01-01"),
                   "rating_observed": 1, "rating_direction": directions[index % 3]}
            row.update({feature: float(index % 5) for feature in feature_names})
            rows.append(row)
        result = run_rating(pd.DataFrame(rows), "2022-12-31")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["train_n"], 12)
        self.assertEqual(result["test_n"], 6)
        self.assertEqual(set(result["class_counts"]["test"]), {"downgrade", "stable", "upgrade"})

    def test_rating_missing_columns_are_reported_without_key_error(self):
        result = run_rating(pd.DataFrame({"publication_date": pd.to_datetime(["2020-01-01"])}), "2022-12-31")
        self.assertEqual(result["status"], "missing_rating_target_columns")
        self.assertIn("rating_observed", result["missing_columns"])

    def test_pd_excludes_invalid_observed_targets(self):
        frame = pd.DataFrame([
            {"inn": "a", "publication_date": "2021-01-01", "default_12m": 0, "default_12m_observed": 1},
            {"inn": "b", "publication_date": "2021-02-01", "default_12m": 1, "default_12m_observed": 1},
            {"inn": "c", "publication_date": "2023-01-01", "default_12m": 0, "default_12m_observed": 1},
            {"inn": "d", "publication_date": "2023-02-01", "default_12m": 7, "default_12m_observed": 1},
        ])
        result = run_pd(frame, "2022-12-31", bootstrap_repetitions=2)
        self.assertEqual(result["invalid_target_n"], 1)
        self.assertEqual(result["train_n"], 2)
        self.assertEqual(result["test_n"], 1)

    def test_emitter_bootstrap_reports_cluster_intervals(self):
        result = bootstrap_emitter_intervals(
            [0, 1, 0, 1], [0.1, 0.8, 0.2, 0.7], ["a", "a", "b", "b"],
            repetitions=5, seed=7,
        )
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["n_emitters"], 2)
        self.assertEqual(result["seed"], 7)
        self.assertEqual(result["intervals"]["roc_auc"]["valid_replicates"], 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)