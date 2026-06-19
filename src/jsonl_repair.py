# -*- coding: utf-8 -*-
"""
Надёжное чтение и починка повреждённых JSONL-артефактов.
=========================================================
Решает «Проблему 1 (Критично)»: в экспортах прошлых пайплайнов встречались
строки без открывающей скобки ``{``, слипшиеся объекты ``}{"id":...`` без
разделителей и мусорные префиксы вроде ``?{"id"...`` или BOM. Обычный
``json.loads`` на таких данных падает или (что хуже) такие строки молча
выбрасываются, и данные теряются.

Этот модуль читает файл потоково и пытается ВОССТАНОВИТЬ как можно больше
валидных объектов:

  * срезает мусор и BOM перед первой ``{``;
  * разрезает несколько объектов, слипшихся в одной строке (``}{``), честным
    сканером глубины скобок, который уважает строки и экранирование;
  * собирает объект, разорванный на несколько строк;
  * гарантирует наличие обязательных ключей (``chunk_id``/``content``/
    ``metadata``); недостающие достраивает (id — из хэша содержимого).

Модуль не зависит от тяжёлых библиотек — только стандартная библиотека,
поэтому его можно использовать и в офлайн-аудите (build_ideal_index --dry-run).
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

logger = logging.getLogger("koib.jsonl_repair")


@dataclass
class RepairReport:
    """Счётчики того, что произошло при чтении файла."""

    lines: int = 0
    objects_ok: int = 0          # распарсились сразу
    objects_repaired: int = 0    # удалось распарсить после починки
    objects_split: int = 0       # извлечено из слипшихся ``}{``
    objects_recovered: int = 0   # собраны из нескольких строк
    keys_backfilled: int = 0     # достроены отсутствующие обязательные ключи
    unrecoverable: int = 0       # не удалось восстановить даже частично
    files: List[str] = field(default_factory=list)

    def merge(self, other: "RepairReport") -> None:
        self.lines += other.lines
        self.objects_ok += other.objects_ok
        self.objects_repaired += other.objects_repaired
        self.objects_split += other.objects_split
        self.objects_recovered += other.objects_recovered
        self.keys_backfilled += other.keys_backfilled
        self.unrecoverable += other.unrecoverable
        self.files.extend(other.files)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "lines": self.lines,
            "objects_ok": self.objects_ok,
            "objects_repaired": self.objects_repaired,
            "objects_split": self.objects_split,
            "objects_recovered": self.objects_recovered,
            "keys_backfilled": self.keys_backfilled,
            "unrecoverable": self.unrecoverable,
            "files": self.files,
        }


def _strip_leading_junk(s: str) -> str:
    """Убрать BOM и любой мусор до первой ``{`` (например ``?{"id"...``)."""
    s = s.lstrip("\ufeff \t\r\n")
    idx = s.find("{")
    if idx > 0:
        s = s[idx:]
    return s


# Строка вида  "chunk_id": "...", ...}  — у неё ОТОРВАНА открывающая ``{``.
_LEADING_KEY_RE = re.compile(r'^[^\{"]{0,6}("(?:[^"\\]|\\.)*"\s*:)')


def _balance_braces(s: str) -> str:
    """Грубо добалансировать фигурные скобки (вне строковых литералов)."""
    depth = 0
    in_str = False
    escape = False
    opens = 0
    closes = 0
    for ch in s:
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            opens += 1
            depth += 1
        elif ch == "}":
            closes += 1
            depth -= 1
    if closes > opens:
        s = "{" * (closes - opens) + s   # не хватает открывающих — это «строка без {»
    elif opens > closes:
        s = s + "}" * (opens - closes)   # незакрытый объект
    return s


def _normalize_candidate(s: str) -> str:
    """Подготовить строку к парсингу.

    Обрабатывает документированные артефакты Проблемы 1:
      * мусор перед ``{`` (``?{"id"...``) — срезается;
      * ОТОРВАННАЯ открывающая ``{`` (строка начинается сразу с ключа
        ``"chunk_id": ...``) — скобка восстанавливается;
      * перекос числа ``{`` и ``}`` — добалансируется.
    """
    s = s.lstrip("\ufeff \t\r\n")
    if not s:
        return s
    if s.startswith("{"):
        return s
    m = _LEADING_KEY_RE.match(s)
    if m:
        # строка без открывающей скобки: ставим '{' перед первой кавычкой ключа
        first_quote = s.find('"')
        s = "{" + s[first_quote:]
        return _balance_braces(s)
    # иначе — обычный мусор перед '{'
    return _strip_leading_junk(s)


def _split_concatenated(s: str) -> List[str]:
    """Разрезать строку, в которой слиплись несколько JSON-объектов (``}{``).

    Использует честный сканер глубины фигурных скобок, который не реагирует на
    скобки внутри строковых литералов и корректно обрабатывает экранирование.
    """
    out: List[str] = []
    depth = 0
    start: Optional[int] = None
    in_str = False
    escape = False
    for i, ch in enumerate(s):
        if in_str:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            if depth > 0:
                depth -= 1
                if depth == 0 and start is not None:
                    out.append(s[start : i + 1])
                    start = None
    return out


def _coerce_record(obj: Dict[str, Any], report: RepairReport) -> Optional[Dict[str, Any]]:
    """Гарантировать обязательные ключи. Вернуть None, если содержимого нет."""
    if not isinstance(obj, dict):
        return None

    content = obj.get("content")
    if content is None:
        # некоторые старые экспорты клали текст в page_content/text
        content = obj.get("page_content") or obj.get("text")
    content = content if isinstance(content, str) else ("" if content is None else str(content))

    meta = obj.get("metadata")
    if not isinstance(meta, dict):
        meta = {}

    chunk_id = obj.get("chunk_id") or obj.get("id") or meta.get("chunk_id")
    backfilled = False
    if not chunk_id:
        chunk_id = "rec_" + hashlib.sha256(content.encode("utf-8")).hexdigest()[:16]
        backfilled = True
    if "metadata" not in obj or not isinstance(obj.get("metadata"), dict):
        backfilled = True
    if backfilled:
        report.keys_backfilled += 1

    if not content.strip() and not meta:
        return None

    record = dict(obj)
    record["chunk_id"] = chunk_id
    record["content"] = content
    record["metadata"] = meta
    record.setdefault("chunk_type", obj.get("chunk_type") or meta.get("chunk_type") or "text")
    return record


def iter_jsonl_records(path: Path, report: Optional[RepairReport] = None) -> Iterator[Dict[str, Any]]:
    """Прочитать JSONL с починкой. Каждый yield — валидный dict-объект.

    Args:
        path: путь к .jsonl/.txt артефакту.
        report: опциональный отчёт для накопления статистики (можно общий).
    """
    report = report if report is not None else RepairReport()
    report.files.append(Path(path).name)
    pending = ""  # буфер для объектов, разорванных между строк

    def emit(raw_obj_str: str, *, source: str) -> Iterator[Dict[str, Any]]:
        try:
            obj = json.loads(raw_obj_str)
        except Exception:
            report.unrecoverable += 1
            return
        rec = _coerce_record(obj, report)
        if rec is None:
            report.unrecoverable += 1
            return
        if source == "ok":
            report.objects_ok += 1
        elif source == "repaired":
            report.objects_repaired += 1
        elif source == "split":
            report.objects_split += 1
        elif source == "recovered":
            report.objects_recovered += 1
        yield rec

    with Path(path).open("r", encoding="utf-8-sig", errors="ignore") as f:
        for line in f:
            report.lines += 1
            line = line.rstrip("\n")
            stripped = line.strip()
            if not stripped:
                continue

            # 1) быстрый путь: целая валидная строка
            if pending == "":
                try:
                    obj = json.loads(stripped)
                    rec = _coerce_record(obj, report)
                    if rec is not None:
                        report.objects_ok += 1
                        yield rec
                        continue
                except Exception:
                    pass

            # 2) починка: восстановить '{', срезать мусор, добалансировать скобки
            candidate = pending + stripped if pending else _normalize_candidate(stripped)

            # 3) попытаться разрезать слипшиеся объекты
            parts = _split_concatenated(candidate)
            if len(parts) >= 2:
                produced = False
                for part in parts:
                    for rec in emit(part, source="split"):
                        produced = True
                        yield rec
                if produced:
                    pending = ""
                    continue

            # 4) ровно один объект найден сканером — это и есть починенный объект
            if len(parts) == 1:
                source = "recovered" if pending else "repaired"
                produced = False
                for rec in emit(parts[0], source=source):
                    produced = True
                    yield rec
                if produced:
                    pending = ""
                    continue

            # 5) объект ещё не закрыт — копим строки до закрытия
            if candidate.count("{") > candidate.count("}"):
                pending = candidate
                continue

            # 6) последняя попытка распарсить как есть
            produced = False
            for rec in emit(candidate, source="recovered" if pending else "repaired"):
                produced = True
                yield rec
            pending = "" if produced else ""
            if not produced:
                report.unrecoverable += 1

    # хвост: незакрытый объект
    if pending.strip():
        for _ in emit(pending, source="recovered"):
            yield _


def validate_jsonl_file(path: Path) -> Dict[str, Any]:
    """Проверить, что КАЖДАЯ строка файла — самостоятельный валидный JSON.

    Возвращает {'total', 'valid', 'invalid', 'invalid_lines'}.
    Используется как пост-проверка собственного экспорта.
    """
    total = valid = 0
    invalid_lines: List[int] = []
    with Path(path).open("r", encoding="utf-8-sig", errors="ignore") as f:
        for i, line in enumerate(f, 1):
            s = line.strip()
            if not s:
                continue
            total += 1
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    valid += 1
                else:
                    invalid_lines.append(i)
            except Exception:
                invalid_lines.append(i)
    return {
        "total": total,
        "valid": valid,
        "invalid": len(invalid_lines),
        "invalid_lines": invalid_lines[:50],
    }


def write_jsonl(records: Iterable[Dict[str, Any]], out_path: Path) -> int:
    """Записать записи в JSONL с гарантией валидности каждой строки.

    Перед записью каждая запись прогоняется через json.dumps и обратно через
    json.loads — строка попадает в файл только если она парсится.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            try:
                line = json.dumps(rec, ensure_ascii=False)
                json.loads(line)  # самопроверка
            except Exception as exc:  # pragma: no cover - запись пропускается
                logger.warning("Пропущена незаписываемая запись: %s", exc)
                continue
            f.write(line + "\n")
            written += 1
    return written
