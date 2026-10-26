"""Формирование проверяемой таблицы интерпретации гипотез.

Модуль читает только сохраненные результаты. Он не пересчитывает модели и не
подменяет отсутствующие реальные данные smoke-значениями.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


HYPOTHESES = ["H1", "H2", "H3", "H4"]
RESULT_COLUMNS = ["hypothesis", "status", "test", "evidence", "limitation"]


def _not_evaluable(hypothesis: str, test: str, limitation: str) -> dict[str, Any]:
    return {"hypothesis": hypothesis, "status": "not_evaluable", "test": test,
            "evidence": {}, "limitation": limitation}


def _specification(metrics: dict[str, Any], name: str) -> dict[str, Any] | None:
    value = metrics.get("pd", {}).get("specifications", {}).get(name)
    return value if isinstance(value, dict) and value.get("status", "ok") not in {"not_run"} else None


def _coefficient_map(specification: dict[str, Any] | None) -> dict[str, float]:
    if not specification:
        return {}
    result = {}
    for item in specification.get("coefficients", []):
        if not isinstance(item, dict) or "feature" not in item or "coefficient" not in item:
            continue
        try:
            result[str(item["feature"])] = float(item["coefficient"])
        except (TypeError, ValueError):
            continue
    return result


def _paired_bootstrap_difference(
    first: dict[str, Any], second: dict[str, Any], metric: str,
) -> dict[str, Any]:
    """Read a precomputed paired interval; never infer uncertainty from point estimates."""
    comparisons = second.get("paired_comparison", {})
    interval = comparisons.get(metric) if isinstance(comparisons, dict) else None
    if isinstance(interval, dict) and interval.get("lower") is not None and interval.get("upper") is not None:
        return {"status": "available", "lower": interval["lower"], "upper": interval["upper"],
                "method": interval.get("method", "paired_bootstrap")}
    return {"status": "not_available", "reason": f"paired_{metric}_interval_missing"}


def _sign_stability(robustness: dict[str, Any] | None, feature_names: set[str]) -> dict[str, Any]:
    if not robustness:
        return {"status": "not_available", "reason": "robustness_results_missing"}
    relevant = []
    for scenario in robustness.get("scenarios", []):
        result = scenario.get("result", {})
        for spec in result.get("specifications", {}).values():
            for item in spec.get("coefficients", []):
                if item.get("feature") in feature_names:
                    relevant.append(item)
    if not relevant:
        return {"status": "not_available", "reason": "robustness_coefficients_missing"}
    positive = sum(float(item.get("coefficient", 0)) > 0 for item in relevant)
    negative = sum(float(item.get("coefficient", 0)) < 0 for item in relevant)
    total = positive + negative
    return {"status": "available", "positive": positive, "negative": negative,
            "nonzero_total": total, "positive_share": positive / total if total else None}


def evaluate_hypotheses(metrics: dict[str, Any] | None, robustness: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if not metrics:
        return [_not_evaluable(hypothesis, "saved model metrics", "model_metrics.json отсутствует") for hypothesis in HYPOTHESES]

    results: list[dict[str, Any]] = []
    sentiment = _specification(metrics, "sentiment")
    combined = _specification(metrics, "combined")
    financials = _specification(metrics, "financials")
    coefficients = _coefficient_map(combined or sentiment)
    h1_features = {"report_negative_share", "report_risk_share"}
    h1_values = {name: coefficients[name] for name in h1_features if name in coefficients}
    if len(h1_values) < 2:
        results.append(_not_evaluable("H1", "sentiment/combined coefficients and PD metrics",
                                      "в сохраненных результатах нет коэффициентов negative/risk для проверки H1"))
    else:
        h1_ok = all(value > 0 for value in h1_values.values())
        uncertainty = (combined or sentiment).get("coefficient_intervals", {})
        intervals_available = all(
            isinstance(uncertainty.get(name), dict)
            and uncertainty[name].get("lower") is not None
            and uncertainty[name].get("lower") > 0
            for name in h1_values
        )
        results.append({"hypothesis": "H1", "status": "supported" if h1_ok and intervals_available else "inconclusive",
                        "test": "positive negative/risk coefficients with 95% intervals excluding zero",
                        "evidence": {"coefficients": h1_values,
                                     "coefficient_intervals": {name: uncertainty.get(name) for name in h1_values},
                                     "pr_auc": (combined or sentiment).get("pr_auc"),
                                     "robustness_signs": _sign_stability(robustness, h1_features)},
                        "limitation": "без коэффициентных интервалов вывод остается неопределенным; коэффициентная связь не доказывает причинность"})

    if not financials or not combined or "pr_auc" not in financials or "pr_auc" not in combined:
        results.append(_not_evaluable("H2", "PR-AUC/Brier financials versus combined on same test split",
                                      "отсутствуют сопоставимые финансовая и объединенная спецификации"))
    else:
        same_n = financials.get("n") == combined.get("n")
        pr_improved = combined.get("pr_auc") is not None and financials.get("pr_auc") is not None and combined["pr_auc"] > financials["pr_auc"]
        brier_improved = combined.get("brier") is not None and financials.get("brier") is not None and combined["brier"] < financials["brier"]
        paired_pr = _paired_bootstrap_difference(financials, combined, "pr_auc")
        paired_brier = _paired_bootstrap_difference(financials, combined, "brier")
        uncertainty_supports = (paired_pr.get("status") == "available" and paired_pr["lower"] > 0
                                and paired_brier.get("status") == "available" and paired_brier["upper"] < 0)
        status = "supported" if same_n and pr_improved and brier_improved and uncertainty_supports else "inconclusive"
        results.append({"hypothesis": "H2", "status": status,
                        "test": "PR-AUC higher and Brier lower for combined than financials on same test rows",
                        "evidence": {"same_test_n": same_n, "financials": {"pr_auc": financials.get("pr_auc"), "brier": financials.get("brier")},
                                     "combined": {"pr_auc": combined.get("pr_auc"), "brier": combined.get("brier")},
                                     "paired_difference_intervals": {"pr_auc": paired_pr, "brier": paired_brier}},
                        "limitation": "точечных оценок недостаточно: нужны парные интервалы разницы; сравнение прогностическое, не причинное"})

    rating = metrics.get("rating") if isinstance(metrics.get("rating"), dict) else None
    if not rating or rating.get("status") != "ok":
        results.append(_not_evaluable("H3", "rating macro-F1/balanced accuracy and directional classes",
                                      "рейтинговая модель не имеет статуса ok"))
    else:
        results.append({"hypothesis": "H3", "status": "inconclusive",
                        "test": "rating-direction model with downgrade/stable/upgrade classes",
                        "evidence": {"macro_f1": rating.get("macro_f1"),
                                     "balanced_accuracy": rating.get("balanced_accuracy"),
                                     "class_counts": rating.get("class_counts")},
                        "limitation": "метрики сами по себе не устанавливают направленную связь тональности с понижением рейтинга"})

    if not combined or not coefficients:
        results.append(_not_evaluable("H4", "text coefficients in combined model after financial controls",
                                      "отсутствуют коэффициенты объединенной модели"))
    else:
        text_coefficients = {name: value for name, value in coefficients.items() if name.startswith("report_")}
        results.append({"hypothesis": "H4", "status": "inconclusive",
                        "test": "text coefficients estimated jointly with financial predictors",
                        "evidence": {"text_coefficients": text_coefficients,
                                     "combined_model": combined.get("model_type")},
                        "limitation": "эндогенность, пропуски и словарная спецификация не позволяют трактовать знак как причинный эффект"})
    return results


def write_results(results: list[dict[str, Any]], output_json: str | Path, output_csv: str | Path | None = None) -> None:
    Path(output_json).write_text(json.dumps({"protocol_version": "interpretation-1.0", "results": results},
                                             ensure_ascii=False, indent=2), encoding="utf-8")
    if output_csv:
        with Path(output_csv).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=RESULT_COLUMNS, delimiter=";")
            writer.writeheader()
            for result in results:
                writer.writerow({"hypothesis": result["hypothesis"], "status": result["status"],
                                 "test": result["test"], "evidence": json.dumps(result["evidence"], ensure_ascii=False),
                                 "limitation": result["limitation"]})


def main() -> None:
    parser = argparse.ArgumentParser(description="Сформировать таблицу интерпретации гипотез")
    parser.add_argument("--metrics-json", default="model_metrics.json")
    parser.add_argument("--robustness-json", default="robustness_results.json")
    parser.add_argument("--output-json", default="hypothesis_results.json")
    parser.add_argument("--output-csv", default="hypothesis_results.csv")
    args = parser.parse_args()
    metrics_path = Path(args.metrics_json)
    robustness_path = Path(args.robustness_json)
    metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig")) if metrics_path.exists() else None
    robustness = json.loads(robustness_path.read_text(encoding="utf-8-sig")) if robustness_path.exists() else None
    results = evaluate_hypotheses(metrics, robustness)
    write_results(results, args.output_json, args.output_csv)
    print(json.dumps({"protocol_version": "interpretation-1.0", "statuses": {item["hypothesis"]: item["status"] for item in results}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()