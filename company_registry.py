"""Нормализация реестра компаний для исследовательской панели.

Модуль работает с уже полученным CSV (например, rfsd_all_inns.csv), не скачивает
данные, сохраняет ИНН строкой, объединяет дубликаты и пишет исключения отдельно.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Iterable

ALGORITHM_VERSION = "company-registry-1.0"
DEFAULT_MIN_YEAR = 2018
DEFAULT_MAX_YEAR = 2025
_MISSING = {"", "nan", "none", "null", "не найден", "не найдено", "-"}


def _key(value: Any) -> str:
    text = str(value or "").strip().lstrip("\ufeff").lower()
    return re.sub(r"[^a-zа-я0-9]+", "_", text).strip("_")


def _value(row: dict[str, Any], *names: str) -> str:
    normalized = {_key(k): v for k, v in row.items()}
    for name in names:
        value = normalized.get(_key(name), "")
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def normalize_inn(value: Any) -> str | None:
    """Вернуть 10- или 12-значный ИНН как строку, иначе None."""
    text = str(value or "").strip().replace("\u00a0", "")
    if text.lower() in _MISSING:
        return None
    if re.fullmatch(r"\d+\.0+", text):
        text = text.split(".", 1)[0]
    return text if re.fullmatch(r"\d{10}|\d{12}", text) else None


def _year(value: Any) -> int | None:
    match = re.search(r"\d{4}", str(value or "").strip())
    return int(match.group(0)) if match else None


def _read_rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(8192)
        stream.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
        except csv.Error:
            dialect = csv.excel
        return [dict(row) for row in csv.DictReader(stream, dialect=dialect)]


def _write_rows(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    if not fields:
        fields = ["message"]
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def _first(current: Any, candidate: Any) -> Any:
    return current if str(current or "").strip() else candidate



def build_registry(rows, source="rfsd", source_file="", min_year=DEFAULT_MIN_YEAR,
                   max_year=DEFAULT_MAX_YEAR):
    """Return deterministic (registry, excluded) lists."""
    registry = OrderedDict()
    excluded = []
    for row_number, raw in enumerate(rows, start=2):
        raw_inn = _value(raw, "inn", "ИНН", "emitent_inn", "emitter_inn", "company_inn")
        inn = normalize_inn(raw_inn)
        first = _year(_value(raw, "first_year", "year_from", "min_year", "начальный_год"))
        last = _year(_value(raw, "last_year", "year_to", "max_year", "конечный_год"))
        reason = ""
        if inn is None:
            reason = "invalid_inn"
        elif first is not None and last is not None and first > last:
            reason = "first_year_after_last_year"
        elif first is not None and last is not None and (last < min_year or first > max_year):
            reason = "outside_protocol_period"
        if reason:
            excluded.append({"source_row": row_number, "raw_inn": raw_inn,
                             "source": source, "exclusion_reason": reason})
            continue
        ticker = _value(raw, "ticker", "тикер")
        name = _value(raw, "name", "company_name", "company", "наименование", "название")
        years_count = _value(raw, "n_years", "years_count", "число_лет")
        if inn not in registry:
            registry[inn] = {
                "registry_algorithm": ALGORITHM_VERSION, "inn": inn, "ticker": ticker,
                "company_name": name, "first_year": first or "", "last_year": last or "",
                "n_years": years_count, "min_year": min_year, "max_year": max_year,
                "source": source, "source_file": source_file,
                "registry_status": "coverage_unknown" if first is None or last is None else "ok",
                "source_duplicate_count": 1,
            }
            continue
        item = registry[inn]
        item["ticker"] = _first(item["ticker"], ticker)
        item["company_name"] = _first(item["company_name"], name)
        firsts = [year for year in (item["first_year"], first) if isinstance(year, int)]
        lasts = [year for year in (item["last_year"], last) if isinstance(year, int)]
        item["first_year"] = min(firsts) if firsts else ""
        item["last_year"] = max(lasts) if lasts else ""
        item["n_years"] = _first(item["n_years"], years_count)
        item["source_duplicate_count"] += 1
        item["registry_status"] = "duplicate_merged"
    return list(registry.values()), excluded


def build_from_csv(input_csv, output_csv="company_registry.csv",
                   excluded_csv="company_registry_excluded.csv", source="rfsd",
                   min_year=DEFAULT_MIN_YEAR, max_year=DEFAULT_MAX_YEAR):
    registry, excluded = build_registry(_read_rows(input_csv), source, str(input_csv), min_year, max_year)
    _write_rows(output_csv, registry)
    _write_rows(excluded_csv, excluded)
    return registry, excluded



def main():
    parser = argparse.ArgumentParser(description="Построить реестр компаний и журнал исключений")
    parser.add_argument("input_csv", help="CSV со списком ИНН")
    parser.add_argument("output_csv", nargs="?", default="company_registry.csv")
    parser.add_argument("--excluded-output", default="company_registry_excluded.csv")
    parser.add_argument("--source", default="rfsd")
    parser.add_argument("--min-year", type=int, default=DEFAULT_MIN_YEAR)
    parser.add_argument("--max-year", type=int, default=DEFAULT_MAX_YEAR)
    args = parser.parse_args()
    registry, excluded = build_from_csv(args.input_csv, args.output_csv, args.excluded_output,
                                        args.source, args.min_year, args.max_year)
    print(f"Реестр: {len(registry)} компаний; исключено строк: {len(excluded)}")


if __name__ == "__main__":
    main()
