"""Построение исследовательской панели из собранных строк и признаков документов."""
from __future__ import annotations

import csv
import json
from datetime import date
from pathlib import Path
from typing import Any, Mapping

from text_features import features_from_document
from temporal_targets import add_point_in_time_targets, assert_no_lookahead


def _read_rows(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        sample = stream.read(4096)
        stream.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=";,\t")
        reader = csv.DictReader(stream, dialect=dialect)
        return [dict(row) for row in reader]


def _write_rows(path: str | Path, rows: list[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=keys, delimiter=";", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def enrich_documents(rows: list[dict[str, Any]], use_ocr: bool = True) -> list[dict[str, Any]]:
    """Добавить признаки годового отчёта, оставляя исходные пути и поля без изменений."""
    out = []
    for row in rows:
        rec = dict(row)
        document_path = rec.get("annual_report_path") or rec.get("document_path")
        if document_path:
            features = features_from_document(document_path, use_ocr=use_ocr)
            for key, value in features.items():
                if key == "text":
                    continue
                rec[f"report_{key}"] = value
        else:
            rec["report_extraction_errors"] = "missing_document_path"
        publication = rec.get("annual_report_published_at") or rec.get("publication_date")
        rec["publication_date"] = publication or ""
        rec["feature_cutoff_date"] = publication or ""
        out.append(rec)
    return out


def build_panel(
    rows: list[dict[str, Any]],
    default_events_by_inn: Mapping[str, list[Mapping[str, Any]]] | None = None,
    rating_events_by_inn: Mapping[str, list[Mapping[str, Any]]] | None = None,
    censoring_date: Any | None = None,
    use_ocr: bool = True,
) -> list[dict[str, Any]]:
    enriched = enrich_documents(rows, use_ocr=use_ocr)
    panel = add_point_in_time_targets(
        enriched,
        default_events_by_inn=default_events_by_inn,
        rating_events_by_inn=rating_events_by_inn,
        censoring_date=censoring_date,
    )
    assert_no_lookahead(panel)
    return panel


def build_from_csv(
    input_csv: str | Path,
    output_csv: str | Path,
    default_events_json: str | Path | None = None,
    rating_events_json: str | Path | None = None,
    censoring_date: Any | None = None,
    use_ocr: bool = True,
) -> list[dict[str, Any]]:
    rows = _read_rows(input_csv)
    defaults = json.loads(Path(default_events_json).read_text(encoding="utf-8")) if default_events_json else {}
    ratings = json.loads(Path(rating_events_json).read_text(encoding="utf-8")) if rating_events_json else {}
    panel = build_panel(rows, defaults, ratings, censoring_date=censoring_date, use_ocr=use_ocr)
    _write_rows(output_csv, panel)
    return panel


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Построить point-in-time панель для диссертации")
    parser.add_argument("input_csv")
    parser.add_argument("output_csv")
    parser.add_argument("--default-events-json")
    parser.add_argument("--rating-events-json")
    parser.add_argument("--censoring-date", default=date.today().isoformat())
    parser.add_argument("--no-ocr", action="store_true")
    args = parser.parse_args()
    build_from_csv(args.input_csv, args.output_csv, args.default_events_json, args.rating_events_json,
                   args.censoring_date, use_ocr=not args.no_ocr)
