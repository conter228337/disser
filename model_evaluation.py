"""Короткие метрики и временное разбиение для воспроизводимой оценки моделей."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Iterable, Mapping


def temporal_split(rows: Iterable[Mapping[str, Any]], train_end: str, test_end: str | None = None) -> tuple[list[dict], list[dict]]:
    """Разделить панель по дате публикации, сохраняя порядок времени."""
    train: list[dict] = []
    test: list[dict] = []
    for row in rows:
        stamp = str(row.get("feature_cutoff_date") or row.get("publication_date") or "")[:10]
        if not stamp:
            continue
        if stamp <= train_end:
            train.append(dict(row))
        elif test_end is None or stamp <= test_end:
            test.append(dict(row))
    return train, test


def _valid_pairs(y_true: Iterable[Any], y_score: Iterable[Any]) -> list[tuple[int, float]]:
    pairs = []
    for y, score in zip(y_true, y_score):
        if y in (None, "", "nan") or score in (None, "", "nan"):
            continue
        try:
            pairs.append((int(y), float(score)))
        except (TypeError, ValueError):
            continue
    return pairs


def roc_auc(y_true: Iterable[Any], y_score: Iterable[Any]) -> float | None:
    pairs = _valid_pairs(y_true, y_score)
    positives = [s for y, s in pairs if y == 1]
    negatives = [s for y, s in pairs if y == 0]
    if not positives or not negatives:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in positives for n in negatives)
    return wins / (len(positives) * len(negatives))


def average_precision(y_true: Iterable[Any], y_score: Iterable[Any]) -> float | None:
    pairs = sorted(_valid_pairs(y_true, y_score), key=lambda item: item[1], reverse=True)
    total_positive = sum(y for y, _ in pairs)
    if total_positive == 0:
        return None
    seen_positive = 0
    precision_sum = 0.0
    for rank, (label, _) in enumerate(pairs, start=1):
        if label == 1:
            seen_positive += 1
            precision_sum += seen_positive / rank
    return precision_sum / total_positive


def brier_score(y_true: Iterable[Any], y_score: Iterable[Any]) -> float | None:
    pairs = _valid_pairs(y_true, y_score)
    if not pairs:
        return None
    return sum((score - label) ** 2 for label, score in pairs) / len(pairs)


def binary_accuracy(y_true: Iterable[Any], y_score: Iterable[Any], threshold: float = 0.5) -> float | None:
    pairs = _valid_pairs(y_true, y_score)
    if not pairs:
        return None
    return sum(int((score >= threshold) == bool(label)) for label, score in pairs) / len(pairs)


def calibration_bins(y_true: Iterable[Any], y_score: Iterable[Any], bins: int = 10) -> list[dict[str, float | int]]:
    pairs = _valid_pairs(y_true, y_score)
    groups: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for label, score in pairs:
        index = min(bins - 1, max(0, int(score * bins)))
        groups[index].append((label, score))
    result = []
    for index in range(bins):
        group = groups.get(index, [])
        if not group:
            continue
        result.append({
            "bin": index,
            "n": len(group),
            "mean_predicted": sum(score for _, score in group) / len(group),
            "observed_rate": sum(label for label, _ in group) / len(group),
        })
    return result


def confusion_matrix(y_true: Iterable[Any], y_pred: Iterable[Any], labels: tuple[Any, ...] = (0, 1)) -> dict[str, int]:
    result = {f"true_{a}_pred_{b}": 0 for a in labels for b in labels}
    for actual, predicted in zip(y_true, y_pred):
        if actual in labels and predicted in labels:
            result[f"true_{actual}_pred_{predicted}"] += 1
    return result


def evaluate_default_predictions(y_true: Iterable[Any], y_probability: Iterable[Any]) -> dict[str, Any]:
    """Метрики PD с пропусками и отсутствием положительного класса, обработанными явно."""
    y_true = list(y_true)
    y_probability = list(y_probability)
    return {
        "n": len(_valid_pairs(y_true, y_probability)),
        "roc_auc": roc_auc(y_true, y_probability),
        "pr_auc": average_precision(y_true, y_probability),
        "brier": brier_score(y_true, y_probability),
        "accuracy_at_0_5": binary_accuracy(y_true, y_probability),
        "calibration": calibration_bins(y_true, y_probability),
    }
