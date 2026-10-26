"""Извлечение текста и воспроизводимые признаки тональности финансовых документов.

Модуль намеренно не запускает OCR и не скачивает документы сам.  Он принимает локальный
PDF/ZIP/TXT, сначала использует доступный текстовый слой, а OCR включает только для страниц,
на которых текстовый слой отсутствует.  Все словари и версия алгоритма находятся в этом файле,
поэтому результат можно пересчитать и проверить в приложении к диссертации.
"""
from __future__ import annotations

import hashlib
import io
import re
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ALGORITHM_VERSION = "ru-fin-sentiment-lexicon-1.0"
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9-]*")
_SENTENCE_RE = re.compile(r"[^.!?…]+[.!?…]+|[^.!?…]+$", re.S)
_NEGATIONS = {"не", "ни", "без", "никогда", "ничего", "невозможно", "неизбежно"}

# Лексикон содержит основы и наиболее частые словоформы финансовых раскрытий.
# Это прозрачная базовая разметка; модельные результаты нельзя трактовать как экспертную оценку.
POSITIVE_WORDS = frozenset({
    "рост", "росла", "росло", "растет", "растёт", "увеличение", "увеличен",
    "прибыль", "прибыльный", "успешный", "успешно", "улучшение", "улучшен",
    "устойчивый", "устойчивость", "платежеспособность", "платежеспособный",
    "ликвидность", "ликвидный", "сильный", "сильная", "сильное", "положительный",
    "положительная", "положительное", "снижение_долга", "погашение", "своевременно",
    "стабильный", "стабильность", "эффективность", "диверсификация", "резерв"
})
NEGATIVE_WORDS = frozenset({
    "убыток", "убытки", "убыточный", "снижение", "сокращение", "падение", "потеря",
    "долг", "долговая", "задолженность", "просрочка", "дефолт", "неисполнение",
    "банкротство", "банкрот", "ликвидация", "недостаток", "дефицит", "риск",
    "угроза", "проблема", "негативный", "негативная", "неблагоприятный", "штраф",
    "санкции", "санкция", "суд", "иск", "обесценение", "волатильность", "отток",
    "зависимость", "неопределенность", "неопределённость"
})
RISK_WORDS = frozenset({
    "риск", "риски", "угроза", "угрозы", "дефолт", "дефолтный", "просрочка",
    "неисполнение", "банкротство", "банкрот", "санкции", "санкция", "суд", "иск",
    "обесценение", "волатильность", "зависимость", "концентрация", "рефинансирование",
    "ликвидность", "неплатеж", "неплатежеспособность", "неплатёжеспособность"
})
UNCERTAINTY_WORDS = frozenset({
    "возможно", "возможен", "возможна", "возможны", "вероятно", "вероятность",
    "может", "могут", "ожидается", "ожидаемый", "предполагается", "предполагаемый",
    "неопределенность", "неопределённость", "допускается", "сценарий", "прогноз",
    "оценка", "оценочный", "потенциальный", "потенциально", "если", "при",
    "не исключено", "существенно"
})

@dataclass(frozen=True)
class ExtractionResult:
    path: str
    sha256: str
    text: str
    source: str
    pages: int
    ocr_pages: int
    errors: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        value = asdict(self)
        value["errors"] = list(self.errors)
        return value


def _tokens(text: str) -> list[str]:
    return [m.group(0).lower().replace("ё", "е") for m in _TOKEN_RE.finditer(text or "")]


def _is_negated(tokens: list[str], index: int) -> bool:
    return any(tokens[j] in _NEGATIONS for j in range(max(0, index - 3), index))


def analyze_text(text: str) -> dict[str, float | int | str]:
    """Рассчитать признаки тональности без внешних NLP-зависимостей."""
    text = text or ""
    tokens = _tokens(text)
    n = len(tokens)
    positive = negative = risk = uncertainty = negated_positive = 0
    for i, token in enumerate(tokens):
        # Разрешаем простое сопоставление по основе для русских окончаний.
        candidates = {token}
        if len(token) >= 6:
            candidates.add(token[: max(5, len(token) - 3)])
        if candidates & POSITIVE_WORDS:
            positive += 1
            if _is_negated(tokens, i):
                negated_positive += 1
        if candidates & NEGATIVE_WORDS:
            negative += 1
        if candidates & RISK_WORDS:
            risk += 1
        if candidates & UNCERTAINTY_WORDS:
            uncertainty += 1
    denom = float(n or 1)
    sentence_count = len(_SENTENCE_RE.findall(text)) if text.strip() else 0
    sentiment = (positive - negative) / denom
    return {
        "sentiment_algorithm": ALGORITHM_VERSION,
        "text_chars": len(text),
        "word_count": n,
        "sentence_count": sentence_count,
        "positive_count": positive,
        "negative_count": negative,
        "risk_count": risk,
        "uncertainty_count": uncertainty,
        "negated_positive_count": negated_positive,
        "positive_share": positive / denom,
        "negative_share": negative / denom,
        "neutral_share": max(0.0, 1.0 - positive / denom - negative / denom),
        "risk_share": risk / denom,
        "uncertainty_share": uncertainty / denom,
        "sentiment_score": sentiment,
        "sentiment_polarity": "positive" if sentiment > 0 else "negative" if sentiment < 0 else "neutral",
    }


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _pdf_text(data: bytes, use_ocr: bool) -> tuple[str, int, int, list[str]]:
    """Вернуть (текст, число страниц, OCR-страниц, ошибки); PyMuPDF/OCR опциональны."""
    errors: list[str] = []
    try:
        import fitz  # type: ignore
    except Exception:
        return "", 0, 0, ["fitz_unavailable"]
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as exc:
        return "", 0, 0, [f"pdf_open:{type(exc).__name__}"]
    chunks: list[str] = []
    ocr_pages = 0
    for page in doc:
        text = page.get_text("text") or ""
        if text.strip():
            chunks.append(text)
            continue
        if not use_ocr:
            chunks.append("")
            continue
        try:
            import pytesseract  # type: ignore
            from PIL import Image  # type: ignore
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
            image = Image.open(io.BytesIO(pix.tobytes("png")))
            chunks.append(pytesseract.image_to_string(image, lang="rus+eng"))
            ocr_pages += 1
        except Exception as exc:
            errors.append(f"ocr_page:{type(exc).__name__}")
            chunks.append("")
    return "\n".join(chunks), len(doc), ocr_pages, errors


def _extract_blob(data: bytes, name: str, use_ocr: bool) -> tuple[str, str, int, int, list[str]]:
    suffix = Path(name).suffix.lower()
    if suffix in {".txt", ".text", ".csv", ".html", ".htm"}:
        return data.decode("utf-8", errors="replace"), "text", 0, 0, []
    if suffix == ".pdf" or data[:4] == b"%PDF":
        text, pages, ocr_pages, errors = _pdf_text(data, use_ocr)
        return text, "pdf", pages, ocr_pages, errors
    if suffix == ".zip" or data[:2] == b"PK":
        chunks: list[str] = []
        pages = ocr_pages = 0
        errors: list[str] = []
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                for info in archive.infolist():
                    if info.is_dir() or info.file_size > 300 * 1024 * 1024:
                        continue
                    blob = archive.read(info)
                    text, source, p, op, err = _extract_blob(blob, info.filename, use_ocr)
                    if text.strip():
                        chunks.append(f"\n===== {info.filename} ({source}) =====\n{text}")
                    pages += p
                    ocr_pages += op
                    errors.extend(err)
        except Exception as exc:
            errors.append(f"zip_open:{type(exc).__name__}")
        return "\n".join(chunks), "zip", pages, ocr_pages, errors
    return "", "unknown", 0, 0, ["unsupported_format"]


def extract_document(path: str | Path, use_ocr: bool = True) -> ExtractionResult:
    """Извлечь текст локального документа и вернуть контрольные метаданные."""
    file_path = Path(path)
    try:
        data = file_path.read_bytes()
    except Exception as exc:
        return ExtractionResult(str(file_path), "", "", "error", 0, 0, (f"read:{type(exc).__name__}",))
    text, source, pages, ocr_pages, errors = _extract_blob(data, file_path.name, use_ocr)
    return ExtractionResult(str(file_path), _sha256_bytes(data), text, source, pages, ocr_pages, tuple(errors))


def features_from_document(path: str | Path, use_ocr: bool = True) -> dict:
    result = extract_document(path, use_ocr=use_ocr)
    features = analyze_text(result.text)
    features.update({
        "document_path": result.path,
        "document_sha256": result.sha256,
        "extraction_source": result.source,
        "extraction_pages": result.pages,
        "ocr_pages": result.ocr_pages,
        "extraction_errors": "|".join(result.errors),
    })
    return features
