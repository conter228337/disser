"""Короткие метрики и временное разбиение для воспроизводимой оценки моделей."""
from __future__ import annotations

import math
import random
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
            label_value, probability = float(y), float(score)
            if label_value not in (0.0, 1.0) or not math.isfinite(probability) or not 0.0 <= probability <= 1.0:
                continue
            label = int(label_value)
            pairs.append((label, probability))
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
    pairs = _valid_pairs(y_true, y_probability)
    labels = [label for label, _ in pairs]
    scores = [score for _, score in pairs]
    predictions = [int(score >= 0.5) for score in scores]
    matrix = confusion_matrix(labels, predictions)
    return {
        "n": len(pairs),
        "n_events": sum(labels),
        "event_rate": sum(labels) / len(labels) if labels else None,
        "roc_auc": roc_auc(y_true, y_probability),
        "pr_auc": average_precision(y_true, y_probability),
        "brier": brier_score(y_true, y_probability),
        "accuracy_at_0_5": binary_accuracy(y_true, y_probability),
        "threshold": 0.5,
        "confusion_matrix_at_0_5": matrix,
        "calibration": calibration_bins(y_true, y_probability),
    }


def bootstrap_emitter_intervals(
    y_true: Iterable[Any],
    y_probability: Iterable[Any],
    groups: Iterable[Any],
    repetitions: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Percentile intervals from a cluster bootstrap resampling emitters with replacement."""
    labels = list(y_true)
    scores = list(y_probability)
    group_values = list(groups)
    if not (len(labels) == len(scores) == len(group_values)):
        raise ValueError("y_true, y_probability и groups должны иметь одинаковую длину")
    valid_indices = [index for index, (label, score) in enumerate(zip(labels, scores))
                     if len(_valid_pairs([label], [score])) == 1]
    if not valid_indices:
        return {"status": "no_valid_test_predictions", "method": "emitter_cluster_bootstrap"}
    by_group: dict[str, list[int]] = {}
    for index in valid_indices:
        value = group_values[index]
        if value is None or str(value).strip() in {"", "nan", "None", "<NA>"}:
            return {"status": "missing_emitter_id", "method": "emitter_cluster_bootstrap",
                    "n_rows_missing_emitter_id": sum(
                        group_values[i] is None or str(group_values[i]).strip() in {"", "nan", "None", "<NA>"}
                        for i in valid_indices
                    )}
        key = str(value).strip()
        by_group.setdefault(key, []).append(index)
    if len(by_group) < 2:
        return {"status": "insufficient_emitter_clusters", "method": "emitter_cluster_bootstrap",
                "n_emitters": len(by_group)}
    if repetitions < 1:
        raise ValueError("repetitions должен быть положительным целым числом")

    rng = random.Random(seed)
    emitter_keys = list(by_group)
    metric_values: dict[str, list[float]] = {name: [] for name in ("pr_auc", "roc_auc", "brier")}
    for _ in range(repetitions):
        sampled_indices: list[int] = []
        for key in (rng.choice(emitter_keys) for _ in emitter_keys):
            sampled_indices.extend(by_group[key])
        metrics = evaluate_default_predictions(
            [labels[index] for index in sampled_indices],
            [scores[index] for index in sampled_indices],
        )
        for name in metric_values:
            value = metrics[name]
            if value is not None and math.isfinite(value):
                metric_values[name].append(float(value))

    def percentile(values: list[float], q: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        position = (len(ordered) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "status": "ok",
        "method": "emitter_cluster_bootstrap",
        "n_emitters": len(by_group),
        "repetitions": repetitions,
        "seed": seed,
        "confidence_level": 0.95,
        "intervals": {
            name: {"lower": percentile(values, 0.025), "upper": percentile(values, 0.975),
                   "valid_replicates": len(values)}
            for name, values in metric_values.items()
        },
    }


def paired_bootstrap_metric_differences(
    y_true: Iterable[Any],
    first_probability: Iterable[Any],
    second_probability: Iterable[Any],
    groups: Iterable[Any],
    repetitions: int = 500,
    seed: int = 42,
) -> dict[str, Any]:
    """Paired cluster bootstrap of model 2 minus model 1 on identical test rows."""
    labels = list(y_true)
    first_scores = list(first_probability)
    second_scores = list(second_probability)
    group_values = list(groups)
    if not (len(labels) == len(first_scores) == len(second_scores) == len(group_values)):
        raise ValueError("paired bootstrap inputs must have identical lengths")
    valid = [index for index, triple in enumerate(zip(labels, first_scores, second_scores))
             if len(_valid_pairs([triple[0]], [triple[1]])) == 1
             and len(_valid_pairs([triple[0]], [triple[2]])) == 1]
    by_group: dict[str, list[int]] = {}
    for index in valid:
        group = group_values[index]
        if group is None or str(group).strip() in {"", "nan", "None", "<NA>"}:
            return {"status": "missing_emitter_id", "method": "paired_emitter_cluster_bootstrap"}
        by_group.setdefault(str(group).strip(), []).append(index)
    if len(by_group) < 2:
        return {"status": "insufficient_emitter_clusters", "method": "paired_emitter_cluster_bootstrap",
                "n_emitters": len(by_group)}
    if repetitions < 1:
        raise ValueError("repetitions должен быть положительным")

    rng = random.Random(seed)
    keys = list(by_group)
    deltas: dict[str, list[float]] = {"pr_auc": [], "roc_auc": [], "brier": []}
    for _ in range(repetitions):
        indices = [index for key in (rng.choice(keys) for _ in keys) for index in by_group[key]]
        actual = [labels[index] for index in indices]
        first = evaluate_default_predictions(actual, [first_scores[index] for index in indices])
        second = evaluate_default_predictions(actual, [second_scores[index] for index in indices])
        for metric in deltas:
            left, right = first[metric], second[metric]
            if left is not None and right is not None:
                deltas[metric].append(float(right - left))

    def quantile(values: list[float], p: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        position = (len(ordered) - 1) * p
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)

    return {
        "status": "ok", "method": "paired_emitter_cluster_bootstrap",
        "contrast": "second_model_minus_first_model", "n_emitters": len(by_group),
        "repetitions": repetitions, "seed": seed, "confidence_level": 0.95,
        "intervals": {metric: {"lower": quantile(values, 0.025), "upper": quantile(values, 0.975),
                               "valid_replicates": len(values)} for metric, values in deltas.items()},
    }
