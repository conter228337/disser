"""Воспроизводимая описательная статистика исследовательской панели.

Модуль не обучает модели и не делает причинных выводов. Он принимает готовую
point-in-time панель, формирует таблицы для главы 3 и простые SVG-графики,
используя только pandas из исследовательского окружения.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Iterable

import pandas as pd


FINANCIAL_FEATURES = [
    "ROA", "ROE", "Debt_to_EBITDA", "Total_Liabilities_to_Assets",
    "Current_Ratio", "Cash_to_Assets", "Ln_Total_Assets", "total_assets",
    "equity", "revenue", "net_income",
]
TEXT_FEATURES = [
    "report_positive_share", "report_negative_share", "report_risk_share",
    "report_uncertainty_share", "report_sentiment_score", "report_word_count",
    "report_extraction_pages", "report_ocr_pages",
]
TARGETS = ["default_12m", "default_24m", "rating_target"]


def load_panel(path: str | Path) -> pd.DataFrame:
    """Прочитать CSV панели с теми же разделителями, что и build_panel.py."""
    path = Path(path)
    return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig", decimal=",")


def _date_series(frame: pd.DataFrame) -> pd.Series:
    source = frame.get("publication_date", pd.Series(index=frame.index, dtype="object"))
    if "feature_cutoff_date" in frame:
        source = source.where(source.notna() & source.astype(str).ne(""), frame["feature_cutoff_date"])
    return pd.to_datetime(source, errors="coerce")


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(index=frame.index, dtype="float64")
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.where(values.map(lambda value: pd.isna(value) or abs(float(value)) != float("inf")))
    values = pd.to_numeric(frame[column], errors="coerce")
    return values.where(values.map(lambda value: pd.isna(value) or abs(float(value)) != float("inf")))


def _observed_mask(frame: pd.DataFrame, target: str, values: pd.Series) -> pd.Series:
    """Учитывать явный индикатор наблюдаемости, если он есть в панели."""
    observed_column = {"default_12m": "default_12m_observed",
                       "default_24m": "default_24m_observed",
                       "rating_target": "rating_observed"}.get(target)
    mask = values.isin([0, 1])
    if observed_column in frame:
        mask &= _numeric(frame, observed_column).eq(1)
    return mask


def _write_table(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, sep=";", decimal=",", index=False, encoding="utf-8-sig")


def missingness_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for column in frame.columns:
        missing = frame[column].isna() | frame[column].astype(str).str.strip().isin({"", "nan", "None"})
        rows.append({"field": column, "n": len(frame), "missing_n": int(missing.sum()),
                     "missing_share": float(missing.mean()) if len(frame) else None})
    return pd.DataFrame(rows)


def yearly_table(frame: pd.DataFrame) -> pd.DataFrame:
    dates = _date_series(frame)
    work = pd.DataFrame({"year": dates.dt.year})
    work["n_documents"] = 1
    if "inn" in frame:
        work["inn"] = frame["inn"].astype("string")
    result = work.groupby("year", dropna=True).agg(
        n_documents=("n_documents", "sum"),
        n_companies=("inn", "nunique") if "inn" in work else ("n_documents", "sum"),
    ).reset_index()
    for target in TARGETS:
        values = _numeric(frame, target)
        observed = _observed_mask(frame, target, values)
        values = values.where(observed)
        tmp = pd.DataFrame({"year": dates.dt.year, "value": values, "observed": observed})
        grouped = tmp.groupby("year", dropna=True).agg(
            **{f"{target}_n_observed": ("observed", "sum"),
               f"{target}_events": ("value", "sum"),}
        ).reset_index()
        result = result.merge(grouped, on="year", how="left")
        result[f"{target}_rate"] = result[f"{target}_events"] / result[f"{target}_n_observed"].replace(0, pd.NA)
    return result.sort_values("year").reset_index(drop=True)


def target_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target in TARGETS:
        raw_values = _numeric(frame, target)
        observed = _observed_mask(frame, target, raw_values)
        values = raw_values.where(observed)
        rows.append({"target": target, "n_total": len(frame), "n_observed": int(observed.sum()),
                     "n_censored_or_missing": int((~observed).sum()),
                     "n_invalid_target": int((raw_values.notna() & ~raw_values.isin([0, 1])).sum()),
                     "n_events": int((values == 1).sum()),
                     "event_rate_observed": float(values[observed].mean()) if observed.any() else None})
    return pd.DataFrame(rows)


def numeric_summary(frame: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        values = _numeric(frame, column).dropna()
        rows.append({"field": column, "n": int(values.size),
                     "missing_n": int(len(frame) - values.size),
                     "mean": float(values.mean()) if len(values) else None,
                     "std": float(values.std(ddof=1)) if len(values) > 1 else None,
                     "min": float(values.min()) if len(values) else None,
                     "median": float(values.median()) if len(values) else None,
                     "max": float(values.max()) if len(values) else None})
    return pd.DataFrame(rows)


def event_comparison(frame: pd.DataFrame, target: str = "default_12m") -> pd.DataFrame:
    label = _numeric(frame, target)
    label = label.where(_observed_mask(frame, target, label))
    rows = []
    for feature in FINANCIAL_FEATURES + TEXT_FEATURES:
        values = _numeric(frame, feature)
        for group, mask in (("no_event", label.eq(0)), ("event", label.eq(1))):
            selected = values[mask].dropna()
            rows.append({"target": target, "feature": feature, "group": group,
                         "n": int(len(selected)),
                         "mean": float(selected.mean()) if len(selected) else None,
                         "median": float(selected.median()) if len(selected) else None})
    return pd.DataFrame(rows)


def correlation_table(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [c for c in FINANCIAL_FEATURES + TEXT_FEATURES if c in frame]
    if len(columns) < 2:
        return pd.DataFrame(columns=["feature_a", "feature_b", "pearson_r", "n"])
    values = frame[columns].apply(pd.to_numeric, errors="coerce")
    rows = []
    for index, first in enumerate(columns):
        for second in columns[index + 1:]:
            pair = values[[first, second]].dropna()
            rows.append({"feature_a": first, "feature_b": second,
                         "pearson_r": float(pair[first].corr(pair[second])) if len(pair) > 1 else None,
                         "n": int(len(pair))})
    return pd.DataFrame(rows)


def ocr_summary(frame: pd.DataFrame) -> pd.DataFrame:
    source = frame.get("report_extraction_source", frame.get("extraction_source", pd.Series(index=frame.index)))
    ocr = _numeric(frame, "report_ocr_pages")
    errors = frame.get("report_extraction_errors", frame.get("extraction_errors", pd.Series(index=frame.index)))
    return pd.DataFrame([{
        "n_documents": len(frame), "n_with_text_source": int(source.astype(str).isin({"text", "pdf", "zip"}).sum()),
        "n_with_ocr_pages": int(ocr.fillna(0).gt(0).sum()), "ocr_pages_total": int(ocr.fillna(0).sum()),
        "n_with_extraction_errors": int(errors.astype(str).str.strip().ne("").sum()),
        "mean_word_count": float(_numeric(frame, "report_word_count").mean()),
    }])


def grouped_target_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Доли наблюдаемых исходов по отрасли, если поле присутствует в источнике."""
    industry_column = next((name for name in ("industry", "sector", "industry_name") if name in frame), None)
    columns = ["group_field", "group", "target", "n_total", "n_observed", "n_events", "event_rate_observed"]
    if industry_column is None:
        return pd.DataFrame(columns=columns)
    rows = []
    groups = frame[industry_column].astype("string").fillna("").str.strip()
    for group in sorted(value for value in groups.unique() if value):
        mask = groups.eq(group)
        for target in TARGETS:
            group_frame = frame.loc[mask]
            raw_values = _numeric(group_frame, target)
            observed = _observed_mask(group_frame, target, raw_values)
            values = raw_values.where(observed)
            rows.append({"group_field": industry_column, "group": group, "target": target,
                         "n_total": int(mask.sum()), "n_observed": int(observed.sum()),
                         "n_events": int(values.eq(1).sum()),
                         "event_rate_observed": float(values[observed].mean()) if observed.any() else None})
    return pd.DataFrame(rows, columns=columns)


def distribution_table(frame: pd.DataFrame, columns: Iterable[str], bins: int = 10) -> pd.DataFrame:
    """Равномерные гистограммные интервалы; n отражает только непустые значения."""
    rows = []
    for column in columns:
        values = _numeric(frame, column).dropna()
        if values.empty:
            continue
        minimum, maximum = float(values.min()), float(values.max())
        if minimum == maximum:
            rows.append({"field": column, "bin": 0, "bin_left": minimum, "bin_right": maximum,
                         "n": int(len(values)), "n_nonmissing": int(len(values))})
            continue
        width = (maximum - minimum) / bins
        counts = [0] * bins
        for value in values:
            index = min(bins - 1, max(0, int((float(value) - minimum) / width)))
            counts[index] += 1
        for index, count in enumerate(counts):
            rows.append({"field": column, "bin": index,
                         "bin_left": minimum + index * width,
                         "bin_right": maximum if index == bins - 1 else minimum + (index + 1) * width,
                         "n": count, "n_nonmissing": int(len(values))})
    return pd.DataFrame(rows, columns=["field", "bin", "bin_left", "bin_right", "n", "n_nonmissing"])


def _svg_bar(values: list[tuple[str, float]], title: str, path: Path) -> None:
    width, height, left, bottom = 900, 500, 70, 410
    maximum = max((value for _, value in values), default=1) or 1
    bar_width = max(8, (width - left - 30) / max(len(values), 1) - 5)
    bars = []
    for index, (label, value) in enumerate(values):
        x = left + index * (bar_width + 5)
        bar_height = max(0, 330 * value / maximum)
        y = bottom - bar_height
        bars.append(f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" fill="#4472c4"/>')
        bars.append(f'<text x="{x + bar_width / 2:.1f}" y="435" text-anchor="middle" font-size="11">{html.escape(str(label))}</text>')
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">'
           f'<text x="{width / 2}" y="28" text-anchor="middle" font-size="18">{html.escape(title)}</text>'
           f'<line x1="{left}" y1="{bottom}" x2="870" y2="{bottom}" stroke="black"/>' + "".join(bars) + "</svg>")
    path.write_text(svg, encoding="utf-8")


def run_eda(frame: pd.DataFrame, output_dir: str | Path) -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dates = _date_series(frame)
    tables = {
        "yearly_counts_and_targets": yearly_table(frame),
        "missingness": missingness_table(frame),
        "target_summary": target_summary(frame),
        "grouped_targets": grouped_target_summary(frame),
        "numeric_summary": numeric_summary(frame, FINANCIAL_FEATURES + TEXT_FEATURES),
        "numeric_distributions": distribution_table(frame, FINANCIAL_FEATURES + TEXT_FEATURES),
        "event_comparison_default_12m": event_comparison(frame),
        "correlations": correlation_table(frame),
        "ocr_quality": ocr_summary(frame),
    }
    for name, table in tables.items():
        _write_table(table, output / f"{name}.csv")
    yearly = tables["yearly_counts_and_targets"]
    if not yearly.empty:
        _svg_bar([(str(int(row.year)), float(row.n_documents)) for row in yearly.itertuples()],
                 "Documents by publication year", output / "documents_by_year.svg")
        _svg_bar([(str(int(row.year)), float(row.default_12m_events)) for row in yearly.itertuples()],
                 "Observed 12-month hard defaults by year", output / "default_12m_by_year.svg")
        rates = [(str(int(row.year)), float(row.default_12m_rate))
                 for row in yearly.itertuples() if pd.notna(row.default_12m_rate)]
        if rates:
            _svg_bar(rates, "Observed 12-month hard-default rate by year", output / "default_12m_rate_by_year.svg")
    distributions = tables["numeric_distributions"]
    for feature, values in distributions.groupby("field", sort=False):
        _svg_bar([(str(int(row.bin)), float(row.n)) for row in values.itertuples()],
                 f"Distribution: {feature}", output / f"distribution_{feature}.svg")
    metadata = {
        "protocol": "eda-1.0", "n_rows": int(len(frame)),
        "n_companies": int(frame["inn"].nunique()) if "inn" in frame else None,
        "publication_min": dates.min().date().isoformat() if dates.notna().any() else None,
        "publication_max": dates.max().date().isoformat() if dates.notna().any() else None,
        "tables": list(tables),
        "interpretation_note": "Описательные различия не являются доказательством причинности или прогностической силы.",
    }
    (output / "eda_summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser(description="Сформировать описательную статистику и EDA панели")
    parser.add_argument("panel_csv")
    parser.add_argument("--output-dir", default="eda_output")
    args = parser.parse_args()
    frame = load_panel(args.panel_csv)
    if frame.empty:
        raise SystemExit("Панель пуста: EDA не сформирован")
    result = run_eda(frame, args.output_dir)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()