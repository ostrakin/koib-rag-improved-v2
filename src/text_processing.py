# -*- coding: utf-8 -*-
"""
Общие текстовые примитивы KOIB RAG.
====================================
Единственный источник правды для очистки текста, классификации типов
элементов (text/table/formula/figure), детекции артефактов и генерации
саммари таблиц/формул.

Раньше эти функции и регулярные выражения дублировались в chunking.py,
parsing.py, artifacts.py и retrieval.py. Теперь они живут здесь, а все
модули импортируют их отсюда.

Модуль НЕ зависит от тяжёлых библиотек (fitz/torch/faiss/langchain) —
только от .utils, поэтому его можно безопасно импортировать в офлайн-скриптах
(например, в build_ideal_index.py с флагом --dry-run).
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .utils import normalize_ocr_text, strip_embedding_prefix

# ─────────────────────────────────────────────────────────────────────────────
# Регулярные выражения (единый набор для всего проекта)
# ─────────────────────────────────────────────────────────────────────────────
MATH_STRICT_RE = re.compile(
    r"("
    r"\b[A-Za-zА-Яа-я]\s*[=+*/^]\s*[-+]?\d"
    r"|\d+(?:[.,]\d+)?\s*[=+*/^]\s*[-+]?\d"
    r"|\b\d+\s*%\b"
    r"|[∑∫√∞≈≠≤≥±αβγδεζηθλμπρσφψω]"
    r")",
    re.IGNORECASE,
)
# Сильный признак формулы: знак равенства/сравнения, проценты, греческие буквы,
# умножение/степень/сложение между значениями. Деление и дефис между числами
# сюда НЕ входят — это почти всегда номера документов, диапазоны или даты.
STRONG_MATH_RE = re.compile(
    r"([=±≥≤≈≠∑∫√∞]"
    r"|\d\s*%"
    r"|[A-Za-zА-Яа-я0-9]\s*[+*^]\s*[-+]?\d"
    r"|[αβγδεζηθλμπρσφψω])",
    re.IGNORECASE,
)
# «Настоящая» формула: равенство с операндами, сумма/интеграл/корень или
# греческая буква. Само по себе «±», «%» или «+» ничего не значит — это ТТХ,
# допуски, заряд батареи, строки протоколов. Поэтому финальное решение о том,
# что текст — формула, принимается по ЭТОМУ регэкспу, а STRONG_MATH_RE оставлен
# только для совместимости со старым кодом.
TRUE_FORMULA_RE = re.compile(
    r"([∑∫√]"
    r"|[αβγδεζηθλμπρσφψω]"
    r"|[A-Za-zА-Яа-я]\w*\s*=\s*[-+]?\d"
    r"|\d+(?:[.,]\d+)?\s*=\s*[-+]?\d"
    r"|[A-Za-zА-Яа-я0-9_]\s*[+*^]\s*[A-Za-zА-Яа-я0-9_]\s*[=+*/^]"
    r"|(?<![A-Za-zА-Яа-я0-9])[A-Za-zА-Яа-я]\w*\s*=\s*"
    r"(?:[A-Za-zА-Яа-я]\w*\s*[+*/^]"
    r"|[A-Za-zА-Яа-я]\w*\s*\^"
    r"|[-+]?\d+(?:[.,]\d+)?\s*[+*/^])"
    r")"
)
# Допуски/спецификации вида «220±11 В», «210+1мм», «(220±11) В» — НЕ формула,
# а ТТХ. Разделителем может быть ±, + или − (PDF/OCR часто теряют знак ±).
# Единица измерения и скобки необязательны. Если вся строка сводится к числу с
# допуском — это спецификация, а не уравнение.
_UNITS = (r"мм|см|дм|м|кг|г|мг|°|град|вт|в|в\.|а|а\.|гц|кгц|мгц|"
          r"дб|дбм|с|мс|мкс|об|об\s*/\s*мин|rpm|%|°c")
MEASUREMENT_SPEC_RE = re.compile(
    r"^\s*\(?\s*[-]?\d+(?:[.,]\d+)?\s*[±+\u2212-]\s*\d+(?:[.,]\d+)?\s*\)?"
    r"\s*(?:" + _UNITS + r")?\s*\.?\s*$",
    re.IGNORECASE,
)
# Запись «значение с единицей + допуск в скобках»: «напряжение (220±11) В»,
# «ширина 210±1 мм». Строка с таким фрагментом и без настоящего «=» — ТТХ.
MEASUREMENT_PHRASE_RE = re.compile(
    r"\d+(?:[.,]\d+)?\s*(?:" + _UNITS + r")?\s*\(?\s*\d+\s*[±+\u2212-]\s*\d+",
    re.IGNORECASE,
)
# «Процент без оператора» в ТТХ: «0%0/0/0», «заряд 95%», «влажность до 80%» —
# не формула. Процент считается признаком формулы лишь в составе выражения с «=».
BARE_PERCENT_RE = re.compile(r"\b\d+(?:[.,]\d+)?\s*%(?:\s|[/0-9]|$)")
# Юридические/датовые контексты — НЕ формулы (Проблема 6).
LEGAL_REF_RE = re.compile(
    r"(№\s*\d|постановлени|распоряжени|приказ|федеральн\w*\s+закон|"
    r"Российск\w*\s+Федерац|стать[яи]\s+\d|\bст\.\s*\d|\bп\.\s*\d|пункт\s+\d|"
    r"Инструкци|комисси)",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"\b\d{1,2}\s+(?:январ|феврал|март|апрел|ма[яй]|июн|июл|август|сентябр|октябр|ноябр|декабр)\w*\s+\d{4}",
    re.IGNORECASE,
)
DOC_NUMBER_RE = re.compile(r"\b\d{1,5}\s*/\s*\d{1,5}(?:-\d+)?\b")  # 19/204-6, 114/896-8
LATEX_RE = re.compile(r"(\\[a-zA-Z]+|\$[^$]+\$|\$\$.+?\$\$)", re.DOTALL)
LATEX_INLINE_RE = re.compile(r"\$(?!\$)([^$]{1,500})\$")
LATEX_BLOCK_RE = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)
CAPTION_RE = re.compile(r"^\s*(рисунок|рис\.|схема|таблица)\s+\S+", re.IGNORECASE)
NOISE_RE = re.compile(r"^[\W_\d\s.-]{1,20}$", re.UNICODE)

VALID_ELEMENT_TYPES = {"text", "table", "formula", "figure", "heading"}

# Имена старых артефактов индекса/распознавания.
_ARTIFACT_RES = {
    "chunks": re.compile(r"^chunks.*\.(txt|jsonl)$", re.IGNORECASE),
    "docstore": re.compile(r"^docstore.*\.(txt|jsonl)$", re.IGNORECASE),
    "bm25": re.compile(r"^bm25.*\.(txt|jsonl)$", re.IGNORECASE),
}


# ─────────────────────────────────────────────────────────────────────────────
# Очистка текста
# ─────────────────────────────────────────────────────────────────────────────
def sanitize_chunk_content(text: str) -> str:
    """Нормализовать текст чанка, не меняя доменную терминологию.

    normalize_ocr_text уже снимает служебные префиксы, но strip оставлен
    явным для надёжности на дважды-перевыгруженных артефактах.
    """
    return normalize_ocr_text(strip_embedding_prefix(text or ""))


def is_noise_text(text: str, min_alpha: int = 3) -> bool:
    """True для OCR-фрагментов, которые не должны становиться отдельными чанками."""
    if not text:
        return True
    stripped = sanitize_chunk_content(text)
    if not stripped or NOISE_RE.match(stripped):
        return True
    alpha = sum(ch.isalpha() for ch in stripped)
    if alpha < min_alpha and not MATH_STRICT_RE.search(stripped):
        return True
    # Типичный мусор переноса строки: короткий слог с дефисом ("изме-").
    if len(stripped) <= 8 and stripped.endswith("-") and alpha <= 6:
        return True
    return False


def is_true_formula_text(text: str, formula_type: str = "") -> bool:
    """Строгий детектор формул.

    Обычная фраза с '-' или '/' формулой НЕ считается. Юридические номера
    («№ 19/204-6»), даты и ссылки на пункты/постановления — тоже НЕ формулы
    (Проблема 6). Дополнительно отбрасываем ТТХ и допуски: строка вида
    «220±11 В», «210±1мм», «заряд 95%», «0%0/0/0» формулой НЕ является, хотя
    старый STRONG_MATH_RE её пропускал из-за ± и %. Формулой считается только
    настоящий LaTeX, либо наличие равенства с операндами / ∑∫√ / греческой буквы.
    """
    t = sanitize_chunk_content(text)
    if not t or is_noise_text(t, min_alpha=0):
        return False
    if formula_type in {"latex_inline", "latex_block"}:
        return True
    if LATEX_RE.search(t):
        return True
    if CAPTION_RE.match(t):
        return False
    # ТТХ / допуски / «голый» процент — точно не формула, даже если там есть ±/%.
    if MEASUREMENT_SPEC_RE.match(t):
        return False
    # Встроенная спецификация «ширина 210±1 мм», «напряжение (220±11) В» без
    # настоящего равенства — тоже ТТХ, не формула.
    if MEASUREMENT_PHRASE_RE.search(t) and "=" not in t and not TRUE_FORMULA_RE.search(t):
        return False
    if BARE_PERCENT_RE.search(t) and "=" not in t and not TRUE_FORMULA_RE.search(t):
        return False
    # юридический/датовый/номерной контекст без настоящей математики — не формула
    if (LEGAL_REF_RE.search(t) or DATE_RE.search(t) or DOC_NUMBER_RE.search(t)) and not TRUE_FORMULA_RE.search(t):
        return False
    if len(t) > 260 and not TRUE_FORMULA_RE.search(t):
        return False
    if TRUE_FORMULA_RE.search(t):
        return True
    # запасной вариант — старый детектор, но без чистых номеров документов и
    # без ТТХ-подобных строк (± и % уже отсеяны выше).
    return bool(MATH_STRICT_RE.search(t)) and not DOC_NUMBER_RE.search(t) and not MEASUREMENT_SPEC_RE.match(t)


def normalized_element_type(
    element_type: str, content: str, metadata: Optional[Dict[str, Any]] = None
) -> str:
    """Починить тип элемента, выданный старым OCR/PDF-парсингом."""
    metadata = metadata or {}
    etype = (element_type or "text").lower().strip()
    text = sanitize_chunk_content(content)
    if etype == "formula" and not is_true_formula_text(text, str(metadata.get("formula_type", ""))):
        if CAPTION_RE.match(text) and text.lower().startswith(("рис", "рисунок", "схема")):
            return "figure"
        return "text"
    if etype not in VALID_ELEMENT_TYPES:
        return "text"
    return etype


# ─────────────────────────────────────────────────────────────────────────────
# Саммари структурных элементов
# ─────────────────────────────────────────────────────────────────────────────
def markdown_cells(line: str) -> List[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def all_table_cells(markdown: str) -> List[str]:
    cells = [c.strip() for c in sanitize_chunk_content(markdown).replace("\n", "|").split("|")]
    return [c for c in cells if c and c != "---"]


def generate_table_summary(table_markdown: str, metadata: Dict) -> str:
    """Эвристическое саммари таблицы для векторного поиска."""
    table_markdown = sanitize_chunk_content(table_markdown)
    lines = [l for l in table_markdown.split("\n") if l.strip()]
    header_line = lines[0] if lines else ""
    num_rows = int(metadata.get("num_rows", 0) or max(0, len(lines) - 2))
    num_cols = int(metadata.get("num_cols", 0) or (len(markdown_cells(header_line)) if header_line else 0))

    headers = [h for h in markdown_cells(header_line) if h and h != "---"]
    summary_parts = [f"Таблица ({num_rows} строк, {num_cols} столбцов)."]
    if headers:
        summary_parts.append(f"Столбцы: {', '.join(headers[:10])}.")

    data_lines = [l for l in lines[2:] if l.strip() and "---" not in l][:3]
    if data_lines:
        summary_parts.append("Пример данных:")
        for dl in data_lines:
            cells = [c for c in markdown_cells(dl) if c]
            if cells:
                summary_parts.append("  " + " | ".join(cells[:6]))

    if len(summary_parts) == 1 and table_markdown:
        summary_parts.append(table_markdown[:300])
    return sanitize_chunk_content(" ".join(summary_parts))


def generate_formula_summary(formula_content: str, metadata: Dict) -> str:
    """Эвристическое саммари формулы для индексации."""
    formula_content = sanitize_chunk_content(formula_content)
    formula_type = str(metadata.get("formula_type", "unknown"))
    type_desc = {
        "latex_inline": "Формула (LaTeX, строковая)",
        "latex_block": "Формула (LaTeX, блочная)",
        "suspected_formula": "Формула",
        "unknown": "Формула",
    }.get(formula_type, "Формула")
    return f"{type_desc}: {formula_content[:240]}"


# ─────────────────────────────────────────────────────────────────────────────
# Детекция артефактов
# ─────────────────────────────────────────────────────────────────────────────
def artifact_kind(name: str) -> str:
    for kind, rx in _ARTIFACT_RES.items():
        if rx.match(name):
            return kind
    return "unknown"


def is_supported_artifact(path: Path) -> bool:
    return artifact_kind(Path(path).name) != "unknown"
