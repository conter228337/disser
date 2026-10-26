"""Воспроизводимый запуск базовых моделей PD и рейтинговых переходов.

Скрипт не скачивает данные. Он принимает готовую panel CSV, разделяет наблюдения по датам
публикации, обучает financials/sentiment/combined спецификации и сохраняет метрики. При
отсутствии scikit-learn сообщает понятную инструкцию, но не подменяет результат.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from model_evaluation import (
    bootstrap_emitter_intervals,
    evaluate_default_predictions,
    paired_bootstrap_metric_differences,
)

FINANCIAL_FEATURES = [
    "ROA", "ROE", "Debt_to_EBITDA", "Total_Liabilities_to_Assets", "Current_Ratio",
    "Cash_to_Assets", "Ln_Total_Assets", "total_assets", "equity", "revenue", "net_income",
]
SENTIMENT_FEATURES = [
    "report_positive_share", "report_negative_share", "report_neutral_share", "report_risk_share",
    "report_uncertainty_share", "report_sentiment_score", "report_word_count", "report_ocr_pages",
]


def _require_sklearn():
    try:
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import HistGradientBoostingClassifier
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise RuntimeError("Установите pandas, numpy и scikit-learn для запуска моделей") from exc
    return (ColumnTransformer, HistGradientBoostingClassifier, SimpleImputer, LogisticRegression,
            balanced_accuracy_score, confusion_matrix, f1_score, Pipeline, StandardScaler)


def _numeric_frame(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    values = pd.DataFrame(index=frame.index)
    for feature in features:
        values[feature] = pd.to_numeric(frame.get(feature, pd.Series(index=frame.index)), errors="coerce")
    return values


def _rating_class(value: Any) -> int | float:
    """Свернуть величину рейтингового перехода в downgrade/stable/upgrade."""
    if pd.isna(value):
        return float("nan")
    numeric = float(value)
    return -1 if numeric < 0 else 1 if numeric > 0 else 0


def _logistic_coefficients(model: Any, features: list[str]) -> list[dict[str, Any]]:
    """Извлечь коэффициенты логистической модели вместе с именами индикаторов пропусков."""
    if "model" not in model.named_steps or not hasattr(model.named_steps["model"], "coef_"):
        return []
    imputer = model.named_steps.get("imputer")
    coefficient_model = model.named_steps["model"]
    names = list(features)
    if imputer is not None and hasattr(imputer, "get_feature_names_out"):
        names = list(imputer.get_feature_names_out(features))
    coefficients = coefficient_model.coef_[0].tolist()
    return [{"feature": name, "coefficient": float(value),
             "sign": "positive" if value > 0 else "negative" if value < 0 else "zero"}
            for name, value in zip(names, coefficients)]


def _fit_pd(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: list[str],
    nonlinear: bool = False,
    seed: int = 42,
    winsorize: bool = False,
    target_column: str = "default_12m",
):
    (ColumnTransformer, HistGradientBoostingClassifier, SimpleImputer, LogisticRegression,
     _, _, _, Pipeline, StandardScaler) = _require_sklearn()
    x_train = _numeric_frame(train, features)
    x_test = _numeric_frame(test, features)
    if winsorize:
        for feature in features:
            values = x_train[feature].dropna()
            if len(values) < 2:
                continue
            lower, upper = values.quantile(0.01), values.quantile(0.99)
            x_train[feature] = x_train[feature].clip(lower, upper)
            x_test[feature] = x_test[feature].clip(lower, upper)
    y_train = pd.to_numeric(train[target_column], errors="coerce")
    y_test = pd.to_numeric(test[target_column], errors="coerce")
    valid_train = y_train.isin([0, 1])
    valid_test = y_test.isin([0, 1])
    if not valid_train.all() or not valid_test.all():
        raise ValueError("PD содержит значения кроме 0/1 при observed=1")
    y_train = y_train.astype(int)
    y_test = y_test.astype(int)
    if y_train.nunique() < 2:
        raise ValueError("В обучающем периоде для PD нужен положительный и отрицательный класс")
    if nonlinear:
        model = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                          ("model", HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                                                                        learning_rate=0.05, random_state=seed))])
    else:
        model = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler()),
                          ("model", LogisticRegression(max_iter=1000, random_state=seed))])
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    metrics = evaluate_default_predictions(y_test.tolist(), probability.tolist())
    metrics["model_type"] = "hist_gradient_boosting" if nonlinear else "logistic_regression"
    metrics["features"] = list(features)
    metrics["coefficients"] = [] if nonlinear else _logistic_coefficients(model, features)
    metrics["coefficient_scale"] = "standardized_after_median_imputation" if not nonlinear else None
    metrics["coefficient_interpretation_note"] = (
        "Знак относится к стандартизированному признаку; абсолютные величины не являются единицами исходного поля."
        if not nonlinear else "Модель нелинейная; коэффициенты логистической регрессии неприменимы."
    )
    return model, probability, metrics


def _load_panel(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    if "publication_date" not in frame and "feature_cutoff_date" in frame:
        frame["publication_date"] = frame["feature_cutoff_date"]
    frame["publication_date"] = pd.to_datetime(frame["publication_date"], errors="coerce")
    if frame["publication_date"].isna().all():
        raise ValueError("В panel CSV нет корректных publication_date")
    return frame.sort_values("publication_date").reset_index(drop=True)


def run_pd(
    panel: pd.DataFrame,
    train_end: str,
    test_end: str | None = None,
    seed: int = 42,
    bootstrap_repetitions: int = 200,
    target_column: str = "default_12m",
    observed_column: str | None = None,
    feature_mode: str = "all",
    winsorize: bool = False,
) -> dict[str, Any]:
    observed_column = observed_column or f"{target_column}_observed"
    if target_column not in panel or observed_column not in panel:
        return {"status": "missing_default_target_columns", "train_n": 0, "test_n": 0, "specifications": {}}
    publication_dates = pd.to_datetime(panel["publication_date"], errors="coerce")
    observed = pd.to_numeric(panel[observed_column], errors="coerce")
    target = pd.to_numeric(panel[target_column], errors="coerce")
    date_mask = publication_dates.notna()
    eligible = observed.eq(1) & date_mask & target.isin([0, 1])
    invalid_target_n = int((observed.eq(1) & date_mask & ~target.isin([0, 1])).sum())
    train = panel[(publication_dates <= pd.Timestamp(train_end)) & eligible].copy()
    test = panel[(publication_dates > pd.Timestamp(train_end)) & eligible].copy()
    if test_end:
        test = test[publication_dates.loc[test.index] <= pd.Timestamp(test_end)]
    results: dict[str, Any] = {
        "status": "ok" if not train.empty and not test.empty else "insufficient_temporal_split",
        "train_n": len(train), "test_n": len(test), "specifications": {},
        "invalid_target_n": invalid_target_n,
        "excluded_rows": {f"invalid_{target_column}": invalid_target_n},
        "target_column": target_column,
        "feature_mode": feature_mode,
        "winsorize": winsorize,
        "train_event_rate": float(pd.to_numeric(train[target_column], errors="coerce").mean()) if len(train) else None,
        "test_event_rate": float(pd.to_numeric(test[target_column], errors="coerce").mean()) if len(test) else None,
    }
    if train.empty or test.empty:
        return results
    prediction_cache: dict[str, Any] = {}
    text_features = SENTIMENT_FEATURES
    if feature_mode == "risk_only":
        text_features = ["report_negative_share", "report_risk_share", "report_uncertainty_share"]
    elif feature_mode != "all":
        return {"status": "unknown_feature_mode", "feature_mode": feature_mode,
                "allowed_feature_modes": ["all", "risk_only"], "train_n": len(train), "test_n": len(test)}
    specifications = {
        "financials": FINANCIAL_FEATURES,
        "sentiment": text_features,
        "combined": FINANCIAL_FEATURES + text_features,
    }
    for name, features in specifications.items():
        try:
            model, probability, metrics = _fit_pd(
                train, test, features, seed=seed, winsorize=winsorize, target_column=target_column,
            )
        except ValueError as exc:
            results["specifications"][name] = {"status": "not_run", "reason": str(exc)}
            continue
        if "inn" in test:
            metrics["bootstrap"] = bootstrap_emitter_intervals(
                test[target_column].tolist(), probability.tolist(), test["inn"].tolist(),
                repetitions=bootstrap_repetitions, seed=seed,
            )
        else:
            metrics["bootstrap"] = {"status": "missing_emitter_id", "method": "emitter_cluster_bootstrap"}
        prediction_cache[name] = probability
        results["specifications"][name] = metrics
    try:
        _, probability, metrics = _fit_pd(
            train, test, FINANCIAL_FEATURES + text_features, nonlinear=True, seed=seed,
            winsorize=winsorize, target_column=target_column,
        )
        if "inn" in test:
            metrics["bootstrap"] = bootstrap_emitter_intervals(
                test[target_column].tolist(), probability.tolist(), test["inn"].tolist(),
                repetitions=bootstrap_repetitions, seed=seed,
            )
        else:
            metrics["bootstrap"] = {"status": "missing_emitter_id", "method": "emitter_cluster_bootstrap"}
        prediction_cache["combined_hist_gradient_boosting"] = probability
        results["specifications"]["combined_hist_gradient_boosting"] = metrics
    except ValueError as exc:
        results["specifications"]["combined_hist_gradient_boosting"] = {"status": "not_run", "reason": str(exc)}
    else:
        results["specifications"]["combined_hist_gradient_boosting"] = metrics
    if "financials" in prediction_cache and "combined" in prediction_cache and "inn" in test:
        results["specifications"]["combined"]["paired_comparison"] = paired_bootstrap_metric_differences(
            test[target_column].tolist(), prediction_cache["financials"].tolist(),
            prediction_cache["combined"].tolist(), test["inn"].tolist(),
            repetitions=bootstrap_repetitions, seed=seed,
        )
        # Store metric intervals in the same location consumed by interpret_results.py.
        paired = results["specifications"]["combined"]["paired_comparison"]
        results["specifications"]["combined"]["paired_comparison"] = {
            metric: {**paired.get("intervals", {}).get(metric, {}),
                     "status": paired.get("status"), "method": paired.get("method")}
            for metric in ("pr_auc", "brier")
        }
    return results


def run_rating(panel: pd.DataFrame, train_end: str, test_end: str | None = None, seed: int = 42) -> dict[str, Any]:
    """Модель направления рейтингового перехода: downgrade/stable/upgrade."""
    required = {"publication_date", "rating_observed", "rating_direction"}
    missing = sorted(required - set(panel.columns))
    if missing:
        return {"status": "missing_rating_target_columns", "missing_columns": missing, "train_n": 0, "test_n": 0}
    (_, _, SimpleImputer, LogisticRegression, balanced_accuracy_score,
     confusion_matrix, f1_score, Pipeline, StandardScaler) = _require_sklearn()
    publication_dates = pd.to_datetime(panel["publication_date"], errors="coerce")
    observed = pd.to_numeric(panel["rating_observed"], errors="coerce")
    train = panel[(publication_dates <= pd.Timestamp(train_end)) & observed.eq(1)].copy()
    test = panel[(publication_dates > pd.Timestamp(train_end)) & observed.eq(1)].copy()
    if test_end:
        test = test[publication_dates.loc[test.index] <= pd.Timestamp(test_end)]
    if train.empty or test.empty:
        return {"status": "insufficient_observed_rating_classes", "train_n": len(train), "test_n": len(test)}
    features = FINANCIAL_FEATURES + SENTIMENT_FEATURES
    x_train = _numeric_frame(train, features)
    x_test = _numeric_frame(test, features)
    y_train_delta = pd.to_numeric(train["rating_direction"], errors="coerce")
    y_test_delta = pd.to_numeric(test["rating_direction"], errors="coerce")
    y_train = y_train_delta.map(_rating_class)
    y_test = y_test_delta.map(_rating_class)
    valid_train = y_train.notna(); valid_test = y_test.notna()
    class_labels = [-1, 0, 1]
    class_names = {-1: "downgrade", 0: "stable", 1: "upgrade"}
    train_class_counts = {class_names[label]: int((y_train.loc[valid_train] == label).sum()) for label in class_labels}
    test_class_counts = {class_names[label]: int((y_test.loc[valid_test] == label).sum()) for label in class_labels}
    if not valid_train.any() or not valid_test.any():
        return {"status": "insufficient_observed_rating_rows", "train_n": int(valid_train.sum()),
                "test_n": int(valid_test.sum()), "train_class_counts": train_class_counts,
                "test_class_counts": test_class_counts}
    if y_train.loc[valid_train].nunique() < 2:
        return {"status": "insufficient_observed_rating_classes", "train_n": int(valid_train.sum()),
                "test_n": int(valid_test.sum()), "train_class_counts": train_class_counts,
                "test_class_counts": test_class_counts}
    model = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                      ("scale", StandardScaler()),
                      # Автоматическое определение бинарной/многоклассовой схемы
                      # является поведением LogisticRegression; явный multi_class
                      # удален для совместимости с новыми версиями scikit-learn.
                      ("model", LogisticRegression(max_iter=1000, random_state=seed))])
    model.fit(x_train.loc[valid_train], y_train.loc[valid_train])
    prediction = model.predict(x_test.loc[valid_test])
    actual = y_test.loc[valid_test]
    return {
        "status": "ok", "train_n": int(valid_train.sum()), "test_n": int(valid_test.sum()),
        "class_counts": {"train": train_class_counts, "test": test_class_counts},
        "macro_f1": float(f1_score(actual, prediction, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(actual, prediction)),
        "labels": [{"value": label, "name": class_names[label]} for label in class_labels],
        "confusion_matrix": confusion_matrix(actual, prediction, labels=class_labels).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка временных моделей PD и рейтингов")
    parser.add_argument("panel_csv")
    parser.add_argument("--train-end", required=True, help="Последняя дата обучения, YYYY-MM-DD")
    parser.add_argument("--test-end")
    parser.add_argument("--output-json", default="model_metrics.json")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bootstrap-repetitions", type=int, default=200)
    args = parser.parse_args()
    panel = _load_panel(args.panel_csv)
    result = {"protocol_version": "experiments-1.1", "train_end": args.train_end,
              "test_end": args.test_end, "seed": args.seed,
              "bootstrap_repetitions": args.bootstrap_repetitions,
              "python_version": sys.version.split()[0], "platform": platform.platform(),
              "pandas_version": pd.__version__,
              "numpy_version": __import__("numpy").__version__,
              "sklearn_version": __import__("sklearn").__version__,
              "pd": run_pd(panel, args.train_end, args.test_end, args.seed, args.bootstrap_repetitions),
              "rating": run_rating(panel, args.train_end, args.test_end, args.seed)}
    Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
