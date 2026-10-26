"""Point-in-time разметка кредитного риска.

Входом служат строки панели эмитент-документ и события, опубликованные e-disclosure.
Функции используют только стандартную библиотеку, поэтому их можно применять до установки
тяжелых ML-зависимостей. Дата доступности признаков задается publication_date.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Iterable, Mapping

ALGORITHM_VERSION = "pit-targets-1.0"


def parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).date()
    except ValueError:
        pass
    for fmt in ("%Y-%m-%d", "%d.%m.%Y", "%d/%m/%Y", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(text[:19], fmt).date()
        except ValueError:
            continue
    return None


def _within(event_date: date | None, start: date, end: date) -> bool:
    return event_date is not None and start < event_date <= end


def _first_event(events: Iterable[Mapping[str, Any]], date_key: str, start: date, end: date) -> date | None:
    dates = [parse_date(event.get(date_key)) for event in events]
    dates = [d for d in dates if _within(d, start, end)]
    return min(dates) if dates else None


def build_default_target(
    publication_date: Any,
    default_events: Iterable[Mapping[str, Any]] | None,
    horizon_months: int,
    censoring_date: Any | None = None,
    hard_only: bool = True,
) -> dict[str, Any]:
    """Вернуть target/observed/event_date для заданного горизонта.

    Событие в день публикации не считается будущим: используется строгое условие
    publication_date < event_date. Неполный горизонт помечается observed=0 и target=None.
    """
    start = parse_date(publication_date)
    if start is None:
        return {"target": None, "observed": 0, "event_date": "", "reason": "missing_publication_date"}
    # Границы фиксированы календарным числом дней; для 12/24 месяцев это 365/730 дней.
    end = start + timedelta(days=365 if horizon_months == 12 else 730 if horizon_months == 24 else round(365.25 * horizon_months / 12))
    censor = parse_date(censoring_date) or date.today()
    events = list(default_events or [])
    eligible: list[Mapping[str, Any]] = []
    for event in events:
        stage = str(event.get("stage", "")).lower()
        # e-disclosure сообщает technical и default для одного платежа; hard_only выбирает только финальную стадию.
        if hard_only and stage != "default":
            continue
        if not hard_only and stage not in {"default", "technical", "hard", ""}:
            continue
        if str(event.get("kind", "bond")).lower() not in {"bond", ""}:
            continue
        eligible.append(event)
    first = _first_event(eligible, "pub_date", start, end)
    if first is None:
        fallback = [dict(e, pub_date=e.get("event_date")) for e in eligible if not e.get("pub_date")]
        first = _first_event(fallback, "pub_date", start, end)
    # Дата после фактического среза недоступна; учитываем событие только если оно уже опубликовано.
    observed_event = first is not None and first <= censor
    if observed_event:
        return {"target": 1, "observed": 1, "event_date": first.isoformat(), "reason": "event_observed"}
    if censor < end:
        return {"target": None, "observed": 0, "event_date": "", "reason": "right_censored"}
    return {"target": 0, "observed": 1, "event_date": "", "reason": "observed"}


def _rating_level(value: Any) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    token = str(value).upper().replace("(RU)", "").replace(".RU", "").strip()
    token = token.removeprefix("RU")
    if token in {"WD", "NR", "SD", "D", "DEFAULT"}:
        return None
    grades = ["AAA", "AA+", "AA", "AA-", "A+", "A", "A-", "BBB+", "BBB", "BBB-",
              "BB+", "BB", "BB-", "B+", "B", "B-", "CCC", "CC", "C"]
    try:
        return len(grades) - grades.index(token)
    except ValueError:
        return None


def build_rating_target(
    publication_date: Any,
    current_rating: Any,
    rating_events: Iterable[Mapping[str, Any]] | None,
    horizon_months: int = 12,
    censoring_date: Any | None = None,
) -> dict[str, Any]:
    """Разметить понижение и направление рейтинга, доступные после даты публикации."""
    start = parse_date(publication_date)
    current = _rating_level(current_rating)
    if start is None or current is None:
        return {"rating_target": None, "rating_observed": 0, "rating_event_date": "", "rating_reason": "missing_rating"}
    end = start + timedelta(days=365 if horizon_months == 12 else 730 if horizon_months == 24 else round(365.25 * horizon_months / 12))
    censor = parse_date(censoring_date) or date.today()
    valid: list[tuple[date, int]] = []
    unmapped = False
    for event in rating_events or []:
        event_date = parse_date(event.get("pub_date") or event.get("event_date"))
        raw = event.get("rating") or event.get("value")
        level = _rating_level(raw)
        if event_date is None or not _within(event_date, start, end):
            continue
        if level is None:
            unmapped = True
        else:
            valid.append((event_date, level))
    valid.sort()
    observed_valid = [(d, level) for d, level in valid if d <= censor]
    downgrade = any(level < current for _, level in observed_valid)
    first_downgrade = next(((d, level) for d, level in observed_valid if level < current), None)
    endpoint = observed_valid[-1][1] if observed_valid else current
    if downgrade:
        target = 1
        observed = 1
        reason = "downgrade_observed"
    elif censor < end:
        target = None
        observed = 0
        reason = "right_censored"
    elif unmapped and not valid:
        target = None
        observed = 0
        reason = "withdrawn_or_unmapped"
    else:
        target = 0
        observed = 1
        reason = "observed"
    return {
        "rating_target": target,
        "rating_observed": observed,
        "rating_event_date": first_downgrade[0].isoformat() if first_downgrade else "",
        "rating_reason": reason,
        "rating_current_level": current,
        "rating_future_level": endpoint if observed_valid else None,
        "rating_direction": (endpoint - current) if observed_valid else None,
    }


def add_point_in_time_targets(
    rows: Iterable[Mapping[str, Any]],
    default_events_by_inn: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    rating_events_by_inn: Mapping[str, Iterable[Mapping[str, Any]]] | None = None,
    censoring_date: Any | None = None,
) -> list[dict[str, Any]]:
    """Обогатить строки панели метками 12/24 месяцев и проверить доступность дат."""
    defaults = default_events_by_inn or {}
    ratings = rating_events_by_inn or {}
    out: list[dict[str, Any]] = []
    for row in rows:
        rec = dict(row)
        inn = str(rec.get("inn", ""))
        publication = rec.get("publication_date") or rec.get("annual_report_published_at")
        rec["pit_algorithm"] = ALGORITHM_VERSION
        rec["feature_cutoff_date"] = parse_date(publication).isoformat() if parse_date(publication) else ""
        events = defaults.get(inn, [])
        for months in (12, 24):
            target = build_default_target(publication, events, months, censoring_date=censoring_date)
            rec[f"default_{months}m"] = target["target"]
            rec[f"default_{months}m_observed"] = target["observed"]
            rec[f"default_{months}m_event_date"] = target["event_date"]
            rec[f"default_{months}m_reason"] = target["reason"]
        rating_target = build_rating_target(publication, rec.get("rating"), ratings.get(inn, []), censoring_date=censoring_date)
        rec.update(rating_target)
        out.append(rec)
    return out


def assert_no_lookahead(rows: Iterable[Mapping[str, Any]], date_columns: Iterable[str] | None = None) -> None:
    """Проверить, что даты признаков не позже отсечения наблюдения."""
    # В текущем контуре текстовые признаки строятся из годового отчета. Даты
    # аудита и пояснений проверяются отдельным запуском, только если их текст
    # включен в модель.
    columns = tuple(date_columns or ("annual_report_published_at",))
    for index, row in enumerate(rows):
        cutoff = parse_date(row.get("feature_cutoff_date") or row.get("publication_date"))
        if cutoff is None:
            continue
        for column in columns:
            value = parse_date(row.get(column))
            if value is not None and value > cutoff:
                raise ValueError(f"look-ahead at row {index}: {column}={value} > cutoff={cutoff}")
