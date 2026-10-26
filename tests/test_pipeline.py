"""Быстрые тесты границ метода без сети и внешних данных."""
from __future__ import annotations

import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from model_evaluation import average_precision, brier_score, evaluate_default_predictions, roc_auc
from temporal_targets import (
    add_point_in_time_targets,
    assert_no_lookahead,
    build_default_target,
    build_rating_target,
    parse_date,
)
from text_features import analyze_text, extract_document, features_from_document


class TextTests(unittest.TestCase):
    def test_sentiment_and_negation(self):
        result = analyze_text("Не прибыль, а убыток; сохраняется риск дефолта и неопределенность.")
        self.assertGreater(result["negative_count"], 0)
        self.assertGreater(result["risk_count"], 0)
        self.assertGreater(result["uncertainty_count"], 0)
        self.assertGreaterEqual(result["negated_positive_count"], 1)
        self.assertAlmostEqual(
            result["positive_share"] + result["negative_share"] + result["neutral_share"], 1.0
        )

    def test_empty_text_is_valid(self):
        result = analyze_text("")
        self.assertEqual(result["word_count"], 0)
        self.assertEqual(result["sentiment_score"], 0.0)

    def test_text_and_zip_extraction_keep_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            text_path = root / "report.txt"
            text_path.write_text("Рост прибыли и снижение риска", encoding="utf-8")
            result = extract_document(text_path, use_ocr=False)
            self.assertEqual(result.source, "text")
            self.assertEqual(len(result.sha256), 64)
            self.assertIn("Рост", result.text)
            with zipfile.ZipFile(root / "report.zip", "w") as archive:
                archive.writestr("inside.txt", "Риск дефолта")
            nested = extract_document(root / "report.zip", use_ocr=False)
            self.assertEqual(nested.source, "zip")
            self.assertIn("Риск", nested.text)
            features = features_from_document(text_path, use_ocr=False)
            self.assertEqual(features["extraction_source"], "text")
            self.assertIn("sentiment_algorithm", features)


class TargetTests(unittest.TestCase):
    def test_event_after_publication_is_positive(self):
        result = build_default_target(
            "2023-04-01",
            [{"pub_date": "2023-06-01", "stage": "default", "kind": "bond"}],
            12,
            censoring_date="2026-01-01",
        )
        self.assertEqual(result["target"], 1)
        self.assertEqual(result["observed"], 1)

    def test_same_day_and_technical_event_are_not_hard_default(self):
        same_day = build_default_target(
            "2023-04-01", [{"pub_date": "2023-04-01", "stage": "default"}], 12,
            censoring_date="2026-01-01"
        )
        technical = build_default_target(
            "2023-04-01", [{"pub_date": "2023-06-01", "stage": "technical"}], 12,
            censoring_date="2026-01-01"
        )
        self.assertEqual(same_day["target"], 0)
        self.assertEqual(technical["target"], 0)

    def test_right_censoring_does_not_become_zero(self):
        result = build_default_target("2024-06-01", [], 24, censoring_date="2025-01-01")
        self.assertIsNone(result["target"])
        self.assertEqual(result["observed"], 0)

    def test_event_before_censor_is_kept_positive(self):
        result = build_default_target(
            "2024-06-01", [{"pub_date": "2024-12-01", "stage": "default"}], 24,
            censoring_date="2025-01-01"
        )
        self.assertEqual(result["target"], 1)
        self.assertEqual(result["observed"], 1)

    def test_rating_downgrade_and_withdrawal(self):
        downgrade = build_rating_target(
            "2023-04-01", "BBB", [{"pub_date": "2023-09-01", "rating": "BB+"}],
            censoring_date="2026-01-01"
        )
        withdrawn = build_rating_target(
            "2023-04-01", "BBB", [{"pub_date": "2023-09-01", "rating": "WD"}],
            censoring_date="2026-01-01"
        )
        self.assertEqual(downgrade["rating_target"], 1)
        self.assertLess(downgrade["rating_direction"], 0)
        self.assertIsNone(withdrawn["rating_target"])

    def test_point_in_time_guard(self):
        rows = [{"publication_date": "2023-04-01", "feature_cutoff_date": "2023-04-01",
                 "annual_report_published_at": "2023-05-01"}]
        with self.assertRaises(ValueError):
            assert_no_lookahead(rows)
        rows[0]["annual_report_published_at"] = "2023-04-01"
        panel = add_point_in_time_targets(rows, censoring_date="2026-01-01")
        assert_no_lookahead(panel)


class MetricTests(unittest.TestCase):
    def test_metrics_and_missing_class(self):
        y = [0, 1, 0, 1]
        p = [0.1, 0.8, 0.3, 0.6]
        self.assertEqual(roc_auc(y, p), 1.0)
        self.assertEqual(average_precision(y, p), 1.0)
        self.assertAlmostEqual(brier_score([0, 1], [0.0, 1.0]), 0.0)
        self.assertIsNone(roc_auc([0, 0], [0.1, 0.2]))
        self.assertEqual(evaluate_default_predictions(y, p)["n"], 4)


class DateTests(unittest.TestCase):
    def test_supported_date_forms(self):
        self.assertEqual(str(parse_date("2023-04-01T12:00:00")), "2023-04-01")
        self.assertEqual(str(parse_date("01.04.2023")), "2023-04-01")


if __name__ == "__main__":
    unittest.main(verbosity=2)
