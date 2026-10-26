# Исследовательский контур

Репозиторий содержит сборщики исходных данных и независимый контур диссертационного эксперимента.
Сбор данных с внешних сайтов и тяжёлое обучение запускаются отдельно и в этот короткий smoke-проход не входят.

## Файлы

- `text_features.py` — извлечение TXT/PDF/ZIP, OCR пустых PDF-страниц, SHA-256 и признаки тональности.
- `temporal_targets.py` — point-in-time метки дефолта и понижения рейтинга, правое цензурирование и проверка утечки.
- `build_panel.py` — объединение исходной панели, признаков документов и событий.
- `model_evaluation.py` — метрики PD и калибровочные bins без обучения.
- `run_experiments.py` — временное сравнение financials/sentiment/combined и модель направления рейтинга.
- `robustness.py` — матрица заранее заданных robustness-проверок с явными статусами
  `ok`/`unavailable`, включая горизонты 12/24 месяцев, hard+technical, OCR,
  пороги длины текста, winsorization, отраслевые и минимальные по эмитенту подвыборки.
- `eda.py` — описательная статистика, таблицы пропусков/целей/распределений,
  сравнение документов с событием и без события, OCR-контроль и SVG-графики.
- `interpret_results.py` — таблица H1–H4 по сохраненным метрикам и коэффициентам;
  отсутствие реальных результатов получает `not_evaluable`, а не положительный или
  отрицательный вывод.
- `REFERENCES.md` — проверенные библиографические записи и явно исключенные
  неподтвержденные/отозванные записи.
- `verify_release.py` — preflight-проверка обязательных файлов, UTF-8,
  compileall, тестов и запрет официальных результатов без `panel.csv`.
- `DISSERTATION.md` — основной текст диссертации и протокол эксперимента.
- `DATA_DICTIONARY.md` — кодбук полей.
- `tests/test_pipeline.py` — короткие тесты без сети.
- `company_registry.py` — нормализация реестра ИНН, объединение дубликатов и журнал исключений без сети.
- `audit_financials.py` — аудит машинного финансового CSV: схема, пропуски, числовые ошибки, дубликаты и экстремумы.
- `requirements-data.txt` — зависимости для загрузки и обработки исходных данных.

## Статус выполнения плана

- Этап 0: научный протокол зафиксирован в `research_protocol.md`.
- Этап 1: готов `company_registry.py`; реестр еще не построен — исходный `rfsd_all_inns.csv` и локальный каталог `rfsd_data` отсутствуют.
- Этап 2: готов `audit_financials.py`; аудит реальных данных еще не запускался — финансовый CSV отсутствует.
- Этап 3: контур `annual_reports_ocr.py` и `text_features.py` готов, но реестр документов и ручной аудит еще не сформированы.
- Этап 4: словарный алгоритм реализован и покрыт быстрыми тестами; ручная разметка еще не проведена.
- Этап 5: `build_panel.py` и тесты point-in-time готовы, но `panel.csv` из реальных источников отсутствует.
- Этап 6: `eda.py` готов; запуск на реальной панели ожидает появления `panel.csv`.
- Этап 7: контуры `run_experiments.py` и рейтингового эксперимента готовы; официальный запуск ожидает проверенной панели.
- Этап 8: `robustness.py` и альтернативные point-in-time цели готовы; фактические robustness-результаты ожидают проверенной панели.
- Этап 9: `interpret_results.py` готов; фактические статусы H1–H4 ожидают `model_metrics.json` и robustness-результатов на реальной панели.
- Этап 10: структура диссертации, шаблоны результатов и приложения синхронизированы с кодом; численные поля ожидают реальную панель.
- Этап 11: `REFERENCES.md` содержит проверенные записи Basel/IFRS и научных публикаций; методологии агентств требуют ручной привязки к фактическим сообщениям.
- Этап 12: `verify_release.py` готов; строгий релиз заблокирован отсутствием реальной панели.

Сбор RFSD может скачать несколько больших файлов за каждый год. Перед вызовом сборщика нужно проверить свободное место и выбрать годы/объем выборки; автоматическая полная загрузка здесь не запускалась.

Установка зависимостей контура исходных данных:

```text
python -m pip install -r requirements-data.txt
```


Список ИНН за зарегистрированный период 2018–2025 получается так:

```text
python get_inn.py --min-year 2018 --max-year 2025 --output rfsd_all_inns.csv
```

Для сетевой отладки ограничьте возвращаемую рабочую выборку параметром `--limit N`; диапазон источника при этом остается явно указанным.


## Реестр компаний

После получения списка ИНН из RFSD (`rfsd_all_inns.csv`) сформировать реестр и журнал исключений:

```text
python company_registry.py rfsd_all_inns.csv company_registry.csv --excluded-output company_registry_excluded.csv --source rfsd --min-year 2018 --max-year 2025
```

Перед построением панели рекомендуется выполнить аудит финансового CSV:

```text
python audit_financials.py company_analysis.csv --report-json financial_data_audit.json --unit unspecified
```

Используйте `--unit rubles` или `--unit thousand_rubles` только после сверки с первичным источником. Аудит не удаляет строки и не выполняет winsorization.


`company_registry.csv` является входным реестром этапа сбора данных. `company_registry_excluded.csv` нельзя удалять: он содержит строки с невалидным ИНН, несовместимым периодом или другими причинами исключения.


## Локальный протокол

1. Получить строки `company_analysis_*.csv` и `default_events.csv` существующим сборщиком.
2. Преобразовать события в JSON вида `{ "ИНН": [{"pub_date": "...", "stage": "default", "kind": "bond"}] }`.
3. Построить панель:

   `python build_panel.py company_analysis.csv panel.csv --default-events-json defaults.json`

4. CSV `company_analysis_*.csv` содержит машинные английские имена полей для `build_panel.py`; отдельный файл `company_analysis_display_*.csv` предназначен для просмотра на русском.
5. Сформировать таблицы и SVG-графики описательной статистики:

   `python eda.py panel.csv --output-dir eda_output`

   В `eda_output` сохраняются `yearly_counts_and_targets.csv`, `missingness.csv`,
   `target_summary.csv`, `grouped_targets.csv` (если в панели есть `industry`,
   `sector` или `industry_name`), `numeric_summary.csv`,
   `numeric_distributions.csv`, `event_comparison_default_12m.csv`,
   `correlations.csv`, `ocr_quality.csv`, `eda_summary.json` и SVG-графики.

6. После установки `pandas`, `numpy`, `scikit-learn`, `PyMuPDF`, `Pillow` и `pytesseract` оценить модели:

   `python run_experiments.py panel.csv --train-end 2022-12-31 --test-end 2025-12-31 --output-json model_metrics.json`

   По умолчанию фиксируются `seed=42` и 200 cluster-bootstrap репликаций по
   эмитентам. Их можно изменить явно: `--seed N --bootstrap-repetitions N`.
   JSON содержит размеры и доли событий train/test, PR-AUC, ROC-AUC, Brier,
   калибровку, confusion matrix при пороге 0.5, 95%-интервалы bootstrap,
   версии Python/pandas/numpy/scikit-learn и причины неисполненных спецификаций.
   Классы рейтинга кодируются как `downgrade`, `stable`, `upgrade`.

7. После основного запуска выполнить robustness-матрицу:

   `python robustness.py panel.csv --train-end 2022-12-31 --test-end 2025-12-31 --output-json robustness_results.json`

   Для «hard + technical» панель нужно предварительно построить с
   `build_panel.py ... --include-technical-target`. Если в панели нет отрасли,
   OCR-полей, длины текста или ИНН, соответствующий сценарий получает статус
   `unavailable` с причиной, а не исключается молча. В частности, не
   валидированный альтернативный словарь не выдается за выполненную проверку.

8. Сформировать таблицу интерпретации гипотез:

   `python interpret_results.py --metrics-json model_metrics.json --robustness-json robustness_results.json --output-json hypothesis_results.json --output-csv hypothesis_results.csv`

   Коэффициенты базовой логистической модели сохраняются в стандартизированной
   шкале после median-imputation; знак можно сравнивать, абсолютную величину —
   только с учетом этого ограничения.

9. Выполнить preflight упаковки:

   `python verify_release.py --output-json release_check.json`

   Для финального релиза после появления реальной панели использовать
   `python verify_release.py --strict`; до этого статус ожидаемо остается
   `preflight_ok_no_real_panel`.

Команды предназначены для локального воспроизведения и здесь автоматически не запускаются.

## Правила интерпретации

Нельзя публиковать метрики или вывод о превосходстве тональности до получения реальной панели и завершения временного теста. Цензурированные строки с `*_observed=0` и невалидными целевыми значениями исключаются из обычной бинарной оценки и считаются в журнале исключений; отдельный survival-анализ может быть добавлен позже. Поля `default` и календарный `rating` исходного сборщика используются для описания, пока не проведена датированная point-in-time разметка. Числа из smoke-run на искусственной панели не являются результатами исследования.
