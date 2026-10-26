"""Контрактные тесты robustness-матрицы без внешних данных."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustness import run_robustness


class RobustnessTests(unittest.TestCase):
    def test_matrix_reports_available_and_unavailable_scenarios(self):
        rows = []
        feature_names = [
            "ROA", "ROE", "Debt_to_EBITDA", "Total_Liabilities_to_Assets",
            "Current_Ratio", "Cash_to_Assets", "Ln_Total_Assets", "total_assets",
            "equity", "revenue", "net_income", "report_positive_share",
            "report_negative_share", "report_neutral_share", "report_risk_share",
            "report_uncertainty_share", "report_sentiment_score", "report_word_count",
            "report_ocr_pages",
        ]
        for index in range(12):
            is_test = index >= 6
            rows.append({
                "inn": f"company_{index % 3}",
                "publication_date": "2023-01-01" if is_test else "2021-01-01",
                "default_12m": int(index % 2 == 0), "default_12m_observed": 1,
                "default_24m": int(index % 2 == 0), "default_24m_observed": 1,
                "default_technical_12m": int(index % 3 == 0), "default_technical_12m_observed": 1,
                "industry": "industry_a" if index % 2 else "industry_b",
                **{feature: float(index % 5) for feature in feature_names},
            })
        result = run_robustness(
            pd.DataFrame(rows), "2022-12-31", min_text_lengths=[0],
            min_emitter_observations=[1], bootstrap_repetitions=1,
        )
        self.assertEqual(result["protocol_version"], "robustness-1.0")
        self.assertEqual(result["n_scenarios"], 17)
        statuses = {item["status"] for item in result["scenarios"]}
        self.assertIn("ok", statuses)
        self.assertTrue(any(item["scenario"] == "hard_12m__text_layer_only" for item in result["scenarios"]))

        incomplete = pd.DataFrame(rows).drop(columns=["report_ocr_pages", "industry"])
        incomplete_result = run_robustness(
            incomplete, "2022-12-31", min_text_lengths=[0],
            min_emitter_observations=[1], bootstrap_repetitions=1,
        )
        self.assertIn("unavailable", {item["status"] for item in incomplete_result["scenarios"]})


if __name__ == "__main__":
    unittest.main(verbosity=2)