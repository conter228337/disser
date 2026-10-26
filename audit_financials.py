"""Сетенезависимый аудит финансового CSV перед моделированием."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REQUIRED = ("inn", "year", "total_assets", "equity", "revenue", "net_income", "current_assets", "current_liabilities", "cash")
NUMERIC = ("year", "total_assets", "total_liabilities", "equity", "revenue", "net_income", "ebitda", "current_assets", "current_liabilities", "cash", "ROA", "ROE", "Debt_to_EBITDA", "Total_Liabilities_to_Assets", "Current_Ratio", "Cash_to_Assets", "Ln_Total_Assets")
EMPTY = {"", "na", "nan", "none", "null", "-"}


def number(value: Any) -> float | None:
    text = str(value or "").strip().replace("\u00a0", "").replace(" ", "")
    if text.lower() in EMPTY:
        return None
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".") if text.rfind(",") > text.rfind(".") else text.replace(",", "")
    else:
        text = text.replace(",", ".")
    try:
        value = float(text)
    except ValueError:
        return None
    return value if math.isfinite(value) else None


def quantile(values: list[float], fraction: float) -> float:
    values = sorted(values)
    position = (len(values) - 1) * fraction
    low, high = math.floor(position), math.ceil(position)
    return values[low] if low == high else values[low] + (values[high] - values[low]) * (position - low)


def audit_rows(rows: list[dict[str, str]], input_file: str = "", unit: str = "unspecified") -> dict[str, Any]:
    columns = list(rows[0]) if rows else []
    coverage, errors, values = {}, defaultdict(list), defaultdict(list)
    for column in NUMERIC:
        if column not in columns:
            continue
        missing = invalid = 0
        for row_number, row in enumerate(rows, start=2):
            raw = row.get(column, "")
            if str(raw or "").strip().lower() in EMPTY:
                missing += 1
            else:
                parsed = number(raw)
                if parsed is None:
                    invalid += 1
                    if len(errors[column]) < 20:
                        errors[column].append({"row": row_number, "value": raw})
                else:
                    values[column].append(parsed)
        coverage[column] = {
            "present": len(rows) - missing,
            "missing": missing,
            "missing_share": round(missing / len(rows), 6) if rows else None,
            "invalid_numeric": invalid,
            "min": min(values[column]) if values[column] else None,
            "max": max(values[column]) if values[column] else None,
        }
    pairs = Counter((str(row.get("inn", "")).strip(), str(row.get("year", "")).strip()) for row in rows)
    duplicates = [{"inn": inn, "year": year, "rows": count} for (inn, year), count in pairs.items() if inn and year and count > 1]
    years = [int(number(row.get("year"))) for row in rows if number(row.get("year")) is not None and number(row.get("year")).is_integer()]
    outliers = {}
    for column, column_values in values.items():
        if len(column_values) < 4:
            continue
        q1, q3 = quantile(column_values, .25), quantile(column_values, .75)
        spread = q3 - q1
        low, high = q1 - 1.5 * spread, q3 + 1.5 * spread
        outliers[column] = {"iqr_low": low, "iqr_high": high, "flagged_count": sum(x < low or x > high for x in column_values), "interpretation": "review_only_not_removed"}
    warnings = []
    if any(column not in columns for column in REQUIRED): warnings.append("missing_required_columns")
    if duplicates: warnings.append("duplicate_inn_year")
    if errors: warnings.append("numeric_parse_errors")
    return {
        "audit_version": "financial-audit-1.0", "input_file": input_file,
        "row_count": len(rows), "company_count": len({str(r.get('inn', '')).strip() for r in rows if str(r.get('inn', '')).strip()}),
        "year_min": min(years) if years else None, "year_max": max(years) if years else None,
        "columns": columns, "missing_required_columns": [c for c in REQUIRED if c not in columns],
        "column_coverage": coverage, "numeric_parse_errors": dict(errors),
        "duplicate_inn_year_count": len(duplicates), "duplicate_inn_year_examples": duplicates[:100],
        "iqr_outliers": outliers, "financial_unit": unit, "manual_unit_review_required": unit == "unspecified",
        "unit_review_note": "Сверить рубли/тысячи рублей с первичным источником; автоматический аудит единицу не определяет.",
        "warnings": warnings,
    }


def audit_csv(path: str | Path, unit: str = "unspecified") -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(8192)
        stream.seek(0)
        try:
            dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
        except csv.Error:
            dialect = csv.excel
        return audit_rows([dict(row) for row in csv.DictReader(stream, dialect=dialect)], str(path), unit)


def main() -> None:
    parser = argparse.ArgumentParser(description="Аудит финансового CSV без изменения исходных данных")
    parser.add_argument("input_csv")
    parser.add_argument("--report-json", default="financial_data_audit.json")
    parser.add_argument("--unit", choices=("rubles", "thousand_rubles", "unspecified"), default="unspecified")
    args = parser.parse_args()
    report = audit_csv(args.input_csv, args.unit)
    Path(args.report_json).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"row_count": report["row_count"], "company_count": report["company_count"], "warnings": report["warnings"], "report_json": args.report_json}, ensure_ascii=False))


if __name__ == "__main__":
    main()
