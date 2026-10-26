"""Тесты построения реестра компаний без сети и внешних зависимостей."""
from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from company_registry import build_from_csv, build_registry, normalize_inn


class RegistryTests(unittest.TestCase):
    def test_inn_is_string_and_leading_zero_is_preserved(self):
        self.assertEqual(normalize_inn("0012345678"), "0012345678")
        self.assertEqual(normalize_inn("7707083893.0"), "7707083893")
        self.assertIsNone(normalize_inn("not-an-inn"))

    def test_duplicates_are_merged_and_invalid_rows_are_excluded(self):
        rows = [
            {"ИНН": "0012345678", "name": "А", "first_year": "2019", "last_year": "2021"},
            {"inn": "0012345678", "ticker": "AAA", "first_year": "2018", "last_year": "2022"},
            {"inn": "123", "name": "bad"},
            {"inn": "7707083893", "first_year": "2010", "last_year": "2011"},
        ]
        registry, excluded = build_registry(rows, min_year=2018, max_year=2025)
        self.assertEqual([item["inn"] for item in registry], ["0012345678"])
        self.assertEqual(registry[0]["ticker"], "AAA")
        self.assertEqual(registry[0]["first_year"], 2018)
        self.assertEqual(registry[0]["source_duplicate_count"], 2)
        self.assertEqual(len(excluded), 2)
        self.assertEqual({item["exclusion_reason"] for item in excluded}, {"invalid_inn", "outside_protocol_period"})

    def test_csv_round_trip_uses_utf8_sig_and_semicolon(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.csv"
            source.write_text("inn;name;first_year;last_year\n0012345678;Тест;2018;2020\n", encoding="utf-8-sig")
            output = root / "registry.csv"
            excluded = root / "excluded.csv"
            registry, rejected = build_from_csv(source, output, excluded, min_year=2018, max_year=2025)
            self.assertEqual(len(registry), 1)
            self.assertEqual(rejected, [])
            with output.open("r", encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter=";"))
            self.assertEqual(rows[0]["inn"], "0012345678")
            self.assertEqual(rows[0]["company_name"], "Тест")


if __name__ == "__main__":
    unittest.main(verbosity=2)
