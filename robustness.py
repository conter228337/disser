"""Воспроизводимые проверки устойчивости для этапа 8.

Скрипт не меняет исходную панель и не подбирает спецификацию по тестовым метрикам.
Каждый запуск фиксирует фильтры, целевую переменную, покрытие train/test и статус.
Если нужного поля нет, результат получает ``unavailable`` с причиной вместо
молчаливого пропуска.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_experiments import _load_panel, run_pd


def _as_int_list(values: list[str]) -> list[int]:
    result = []
    for value in values:
        number = int(value)
        if number < 0:
            raise ValueError("Пороговые значения не могут быть отрицательными")
        result.append(number)
    return sorted(set(result))


def _industry_column(panel: pd.DataFrame) -> str | None:
    return next((column for column in ("industry", "sector", "industry_name") if column in panel), None)


def _filter_panel(
    panel: pd.DataFrame,
    *,
    min_text_length: int = 0,
    text_layer_only: bool = False,
    min_emitter_observations: int = 1,
    emitter_count_cutoff: str | None = None,
    industry_column: str | None = None,
    industry_value: str | None = None,
) -> tuple[pd.DataFrame | None, str | None]:
    result = panel.copy()
    if min_text_length > 0:
        if "report_word_count" not in result:
            return None, "missing_report_word_count"
        words = pd.to_numeric(result["report_word_count"], errors="coerce")
        result = result[words >= min_text_length].copy()
    if text_layer_only:
        if "report_ocr_pages" not in result:
            return None, "missing_report_ocr_pages"
        ocr_pages = pd.to_numeric(result["report_ocr_pages"], errors="coerce")
        result = result[ocr_pages.eq(0)].copy()
    if min_emitter_observations > 1:
        if "inn" not in result:
            return None, "missing_inn"
        if emitter_count_cutoff is not None:
            dates = pd.to_datetime(result["publication_date"], errors="coerce")
            historical = dates.le(pd.Timestamp(emitter_count_cutoff))
            counts = result.loc[historical, "inn"].astype("string").value_counts(dropna=False)
        else:
            counts = result["inn"].astype("string").value_counts(dropna=False)
        result = result[result["inn"].astype("string").map(counts) >= min_emitter_observations].copy()
    if industry_value is not None:
        if industry_column is None or industry_column not in result:
            return None, "missing_industry_column"
        groups = result[industry_column].astype("string").str.strip()
        result = result[groups.eq(industry_value)].copy()
    return result, None


def _unavailable(name: str, reason: str, filters: dict[str, Any]) -> dict[str, Any]:
    return {"scenario": name, "status": "unavailable", "reason": reason, "filters": filters}


def _run_scenario(
    panel: pd.DataFrame,
    *,
    name: str,
    train_end: str,
    test_end: str | None,
    target_column: str,
    observed_column: str,
    seed: int,
    bootstrap_repetitions: int,
    filters: dict[str, Any],
) -> dict[str, Any]:
    filter_options = {key: value for key, value in filters.items()
                      if key not in {"feature_mode", "winsorize"}}
    filter_options["emitter_count_cutoff"] = train_end
    feature_mode = filters.get("feature_mode", "all")
    winsorize = filters.get("winsorize", False)
    filtered, reason = _filter_panel(panel, **filter_options)
    if reason:
        return _unavailable(name, reason, {"target_column": target_column, **filters})
    assert filtered is not None
    required = {target_column, observed_column, "publication_date"}
    missing = sorted(required - set(filtered.columns))
    if missing:
        return _unavailable(name, "missing_columns:" + ",".join(missing),
                            {"target_column": target_column, **filters})
    result = run_pd(
        filtered,
        train_end,
        test_end,
        seed=seed,
        bootstrap_repetitions=bootstrap_repetitions,
        target_column=target_column,
        observed_column=observed_column,
        feature_mode=feature_mode,
        winsorize=winsorize,
    )
    return {"scenario": name, "status": result.get("status", "unknown"),
            "filters": {"target_column": target_column, **filters}, "result": result}


def run_robustness(
    panel: pd.DataFrame,
    train_end: str,
    test_end: str | None = None,
    seed: int = 42,
    bootstrap_repetitions: int = 200,
    min_text_lengths: list[int] | None = None,
    min_emitter_observations: list[int] | None = None,
) -> dict[str, Any]:
    """Запустить заранее заданную матрицу robustness-спецификаций."""
    min_text_lengths = min_text_lengths or [0, 1000, 5000]
    min_emitter_observations = min_emitter_observations or [1, 3, 5]
    scenarios: list[dict[str, Any]] = []

    targets = [("hard_12m", "default_12m", "default_12m_observed"),
               ("hard_24m", "default_24m", "default_24m_observed"),
               ("hard_plus_technical_12m", "default_technical_12m", "default_technical_12m_observed")]
    for name, target, observed in targets:
        for winsorize in (False, True):
            for feature_mode in ("all", "risk_only"):
                for text_length in min_text_lengths:
                    filters = {"min_text_length": text_length, "text_layer_only": False,
                               "min_emitter_observations": 1, "feature_mode": feature_mode,
                               "winsorize": winsorize}
                    scenarios.append(_run_scenario(
                        panel, name=f"{name}__text_{text_length}__{feature_mode}__winsor_{winsorize}",
                        train_end=train_end, test_end=test_end, target_column=target,
                        observed_column=observed, seed=seed, bootstrap_repetitions=bootstrap_repetitions,
                        filters=filters,
                    ))
    for text_layer_only in (True,):
        filters = {"min_text_length": 0, "text_layer_only": text_layer_only,
                   "min_emitter_observations": 1, "feature_mode": "all", "winsorize": False}
        scenarios.append(_run_scenario(
            panel, name="hard_12m__text_layer_only", train_end=train_end, test_end=test_end,
            target_column="default_12m", observed_column="default_12m_observed", seed=seed,
            bootstrap_repetitions=bootstrap_repetitions, filters=filters,
        ))
    for minimum in min_emitter_observations:
        filters = {"min_text_length": 0, "text_layer_only": False,
                   "min_emitter_observations": minimum, "feature_mode": "all", "winsorize": False}
        scenarios.append(_run_scenario(
            panel, name=f"hard_12m__min_emitter_observations_{minimum}", train_end=train_end,
            test_end=test_end, target_column="default_12m", observed_column="default_12m_observed",
            seed=seed, bootstrap_repetitions=bootstrap_repetitions, filters=filters,
        ))

    column = _industry_column(panel)
    if column is None:
        scenarios.append(_unavailable("hard_12m__industry_subsamples", "missing_industry_column", {"target_column": "default_12m"}))
    else:
        values = sorted(value for value in panel[column].astype("string").str.strip().dropna().unique() if value)
        for value in values:
            filters = {"min_text_length": 0, "text_layer_only": False,
                       "min_emitter_observations": 1, "industry_column": column,
                       "industry_value": value, "feature_mode": "all", "winsorize": False}
            scenarios.append(_run_scenario(
                panel, name=f"hard_12m__industry_{value}", train_end=train_end, test_end=test_end,
                target_column="default_12m", observed_column="default_12m_observed", seed=seed,
                bootstrap_repetitions=bootstrap_repetitions, filters=filters,
            ))
    scenarios.append(_unavailable(
        "hard_12m__alternate_lexicon", "alternate_lexicon_not_configured_or_validated",
        {"target_column": "default_12m"},
    ))
    return {"protocol_version": "robustness-1.0", "seed": seed,
            "bootstrap_repetitions": bootstrap_repetitions,
            "train_end": train_end, "test_end": test_end,
            "n_scenarios": len(scenarios), "scenarios": scenarios}


def main() -> None:
    parser = argparse.ArgumentParser(description="Запустить robustness-проверки исследовательской панели")
    parser.add_argument("panel_csv")
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--test-end")
    parser.add_argument("--output-json", default="robustness_results.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repetitions", type=int, default=200)
    parser.add_argument("--min-text-lengths", nargs="+", default=["0", "1000", "5000"])
    parser.add_argument("--min-emitter-observations", nargs="+", default=["1", "3", "5"])
    args = parser.parse_args()
    panel = _load_panel(args.panel_csv)
    result = run_robustness(
        panel, args.train_end, args.test_end, seed=args.seed,
        bootstrap_repetitions=args.bootstrap_repetitions,
        min_text_lengths=_as_int_list(args.min_text_lengths),
        min_emitter_observations=_as_int_list(args.min_emitter_observations),
    )
    Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("protocol_version", "n_scenarios", "seed")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()