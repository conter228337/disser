"""Воспроизводимый запуск базовых моделей PD и рейтинговых переходов.

Скрипт не скачивает данные. Он принимает готовую panel CSV, разделяет наблюдения по датам
публикации, обучает financials/sentiment/combined спецификации и сохраняет метрики. При
отсутствии scikit-learn сообщает понятную инструкцию, но не подменяет результат.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from model_evaluation import evaluate_default_predictions

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


def _fit_pd(train: pd.DataFrame, test: pd.DataFrame, features: list[str], nonlinear: bool = False):
    (ColumnTransformer, HistGradientBoostingClassifier, SimpleImputer, LogisticRegression,
     _, _, _, Pipeline, StandardScaler) = _require_sklearn()
    x_train = _numeric_frame(train, features)
    x_test = _numeric_frame(test, features)
    y_train = pd.to_numeric(train["default_12m"], errors="coerce").astype(int)
    y_test = pd.to_numeric(test["default_12m"], errors="coerce").astype(int)
    if y_train.nunique() < 2:
        raise ValueError("В обучающем периоде для PD нужен положительный и отрицательный класс")
    if nonlinear:
        model = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                          ("model", HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15,
                                                                        learning_rate=0.05, random_state=42))])
    else:
        model = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler()),
                          ("model", LogisticRegression(max_iter=1000, random_state=42))])
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    metrics = evaluate_default_predictions(y_test.tolist(), probability.tolist())
    return model, probability, metrics


def _load_panel(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    if "publication_date" not in frame and "feature_cutoff_date" in frame:
        frame["publication_date"] = frame["feature_cutoff_date"]
    frame["publication_date"] = pd.to_datetime(frame["publication_date"], errors="coerce")
    if frame["publication_date"].isna().all():
        raise ValueError("В panel CSV нет корректных publication_date")
    return frame.sort_values("publication_date").reset_index(drop=True)


def run_pd(panel: pd.DataFrame, train_end: str, test_end: str | None = None) -> dict[str, Any]:
    train = panel[(panel.publication_date <= pd.Timestamp(train_end)) & (panel.default_12m_observed == 1)].copy()
    test = panel[(panel.publication_date > pd.Timestamp(train_end)) & (panel.default_12m_observed == 1)].copy()
    if test_end:
        test = test[test.publication_date <= pd.Timestamp(test_end)]
    if train.empty or test.empty:
        raise ValueError("Временное разбиение не содержит train или test")
    results: dict[str, Any] = {"train_n": len(train), "test_n": len(test), "specifications": {}}
    specifications = {
        "financials": FINANCIAL_FEATURES,
        "sentiment": SENTIMENT_FEATURES,
        "combined": FINANCIAL_FEATURES + SENTIMENT_FEATURES,
    }
    for name, features in specifications.items():
        model, probability, metrics = _fit_pd(train, test, features)
        results["specifications"][name] = metrics
    try:
        _, _, metrics = _fit_pd(train, test, FINANCIAL_FEATURES + SENTIMENT_FEATURES, nonlinear=True)
        results["specifications"]["combined_hist_gradient_boosting"] = metrics
    except ValueError:
        pass
    return results


def run_rating(panel: pd.DataFrame, train_end: str, test_end: str | None = None) -> dict[str, Any]:
    """Многоклассовая модель направления изменения рейтинга {-1, 0, +1}."""
    (_, _, SimpleImputer, LogisticRegression, balanced_accuracy_score,
     confusion_matrix, f1_score, Pipeline, StandardScaler) = _require_sklearn()
    train = panel[(panel.publication_date <= pd.Timestamp(train_end)) & (panel.rating_observed == 1)].copy()
    test = panel[(panel.publication_date > pd.Timestamp(train_end)) & (panel.rating_observed == 1)].copy()
    if test_end:
        test = test[test.publication_date <= pd.Timestamp(test_end)]
    if train.empty or test.empty or train["rating_direction"].dropna().nunique() < 2:
        return {"status": "insufficient_observed_rating_classes", "train_n": len(train), "test_n": len(test)}
    features = FINANCIAL_FEATURES + SENTIMENT_FEATURES
    x_train = _numeric_frame(train, features)
    x_test = _numeric_frame(test, features)
    y_train = pd.to_numeric(train["rating_direction"], errors="coerce")
    y_test = pd.to_numeric(test["rating_direction"], errors="coerce")
    valid_train = y_train.notna(); valid_test = y_test.notna()
    model = Pipeline([("imputer", SimpleImputer(strategy="median", add_indicator=True)),
                      ("scale", StandardScaler()),
                      ("model", LogisticRegression(max_iter=1000, multi_class="auto", random_state=42))])
    model.fit(x_train.loc[valid_train], y_train.loc[valid_train])
    prediction = model.predict(x_test.loc[valid_test])
    actual = y_test.loc[valid_test]
    return {
        "status": "ok", "train_n": int(valid_train.sum()), "test_n": int(valid_test.sum()),
        "macro_f1": float(f1_score(actual, prediction, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(actual, prediction)),
        "labels": sorted(set(actual.tolist()) | set(prediction.tolist())),
        "confusion_matrix": confusion_matrix(actual, prediction).tolist(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка временных моделей PD и рейтингов")
    parser.add_argument("panel_csv")
    parser.add_argument("--train-end", required=True, help="Последняя дата обучения, YYYY-MM-DD")
    parser.add_argument("--test-end")
    parser.add_argument("--output-json", default="model_metrics.json")
    args = parser.parse_args()
    panel = _load_panel(args.panel_csv)
    result = {"protocol_version": "experiments-1.0", "train_end": args.train_end,
              "test_end": args.test_end, "pd": run_pd(panel, args.train_end, args.test_end),
              "rating": run_rating(panel, args.train_end, args.test_end)}
    Path(args.output_json).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
