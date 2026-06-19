# Рефакторинг v4.11: устранение дублирования кода

Создан общий модуль `src/text_processing.py` — единственный источник правды для
очистки текста и классификации типов элементов. Дублировавшиеся функции и
регулярные выражения вынесены туда, все модули импортируют их оттуда.

## Что было устранено

| Дубликат | Где жил раньше | Теперь |
|---|---|---|
| `_MATH_STRICT_RE` (регэксп формул) | `chunking.py` **и** `parsing.py` | `text_processing.MATH_STRICT_RE` |
| `_LATEX_RE`, `_CAPTION_RE`, `_NOISE_RE` | `chunking.py` | `text_processing` |
| `sanitize_chunk_content` | `chunking.py`; дублировалась как `_strip_passage_prefix` в `retrieval.py` | `text_processing.sanitize_chunk_content` |
| `is_noise_text`, `is_true_formula_text`, `normalized_element_type` | `chunking.py` | `text_processing` |
| `_markdown_cells`, `_generate_table_summary`, `_generate_formula_summary` | `chunking.py` | `text_processing` |
| Регэкспы имён артефактов (`chunks`/`docstore`/`bm25`) | `parsing.py` **и** `artifacts.py` | `text_processing.artifact_kind` / `is_supported_artifact` |
| Логика определения модели КОИБ | повторялась в `build_ideal_index.py` | `utils.resolve_model` |
| Весь блок очистки в `build_ideal_index.py` | копия функций `chunking.py` | импорт из `text_processing` |

## Принципы

- Имена в местах использования сохранены через алиасы (`as _MATH_STRICT_RE` и т.п.),
  поэтому тела функций ниже по коду не менялись — снижен риск регрессий.
- `text_processing.py` не тянет тяжёлые зависимости (fitz/torch/faiss), поэтому
  `build_ideal_index.py --dry-run` и `export_index.py` работают без них.
- LaTeX-экстракторы (`LATEX_INLINE_RE`/`LATEX_BLOCK_RE`), специфичные для разбора
  PDF, тоже централизованы, но остаются «парсинговыми» по смыслу.

## Проверка

- `python -m py_compile` — все файлы компилируются.
- `pytest -q` — 52 теста проходят.
- Сквозной прогон `load_artifact_chunks` на реальных данных: 0 утёкших префиксов,
  1243 ложные «формулы» переклассифицированы в текст.
