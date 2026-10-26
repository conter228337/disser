"""Тесты финансового аудита CSV."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from audit_financials import audit_csv, audit_rows


class FinancialAuditTests(unittest.TestCase):
    def test_schema_duplicates_missing_and_unit_review(self):
        rows = [
            {"inn": "0012345678", "year": "2020", "total_assets": "1000,5", "equity": "400", "revenue": "2000", "net_income": "100"},
            {"inn": "0012345678", "year": "2020", "total_assets": "bad", "equity": "450", "revenue": "2100", "net_income": "110"},
        ]
        report = audit_rows(rows)
        self.assertEqual(report["company_count"], 1)
        self.assertEqual(report["duplicate_inn_year_count"], 1)
        self.assertEqual(report["numeric_parse_errors"]["total_assets"][0]["value"], "bad")
        self.assertTrue(report["manual_unit_review_required"])
        self.assertIn("current_assets", report["missing_required_columns"])

    def test_semicolon_csv_and_explicit_unit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "financial.csv"
            path.write_text("inn;year;total_assets;equity;revenue;net_income\n0012345678;2020;1000;400;2000;100\n", encoding="utf-8-sig")
            report = audit_csv(path, unit="thousand_rubles")
            self.assertEqual(report["row_count"], 1)
            self.assertEqual(report["year_min"], 2020)
            self.assertFalse(report["manual_unit_review_required"])
            self.assertEqual(report["column_coverage"]["total_assets"]["max"], 1000.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
