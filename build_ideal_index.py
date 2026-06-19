# -*- coding: utf-8 -*-
"""
build_ideal_index.py  —  Сборка ИДЕАЛЬНОГО индекса для KOIB RAG
================================================================
Берёт артефакты прошлых попыток (chunks*/docstore*/bm25*/chunks_export_new*),
чинит все дефекты данных и строит production-индекс:

    output/index/text_index.faiss / summary_index.faiss / bm25_fts.db
    output/docstore/docstore.db

Что чинится: утёкший префикс E5 "passage:", гипер-классификация формул,
вырожденные таблицы, шум в поле model, page==0, дубли. bm25 и chunks_export_new
исключены по умолчанию (лемматизированный текст / потеря метаданных → нет цитат).

ВСЯ логика очистки переиспользует общий модуль src.text_processing — здесь нет
дублирующихся функций. Тяжёлые зависимости (torch/faiss) подключаются только на
этапе записи индекса, поэтому --dry-run работает без них.

Запуск из корня репозитория:
    python build_ideal_index.py --artifacts-dir ./data/artifacts --dry-run
    python build_ideal_index.py --artifacts-dir ./data/artifacts --output-dir ./output
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

# Общие примитивы проекта (без fitz/torch) — единственный источник логики очистки.
from src.text_processing import (
    CAPTION_RE,
    all_table_cells,
    artifact_kind,
    generate_formula_summary,
    generate_table_summary,
    is_noise_text,
    is_true_formula_text,
    normalized_element_type,
    sanitize_chunk_content,
)
from src.utils import KNOWN_MODELS, resolve_model, strip_embedding_prefix, text_hash
from src.jsonl_repair import iter_jsonl_records, RepairReport

# отдельный детектор chunks_export_new (в src он относится к семейству "chunks")
import re as _re

_EXPORT_NEW_RE = _re.compile(r"^chunks_export_new.*\.(txt|jsonl)$", _re.IGNORECASE)
_PREFIX_RE = _re.compile(r"^\s*(passage:|query:)\s*", _re.IGNORECASE)


def classify_file(name: str) -> str:
    if _EXPORT_NEW_RE.match(name):
        return "chunks_new"
    return artifact_kind(name)


# ─────────────────────────────────────────────────────────────────────────────
@dataclass
class CleanChunk:
    chunk_id: str
    content: str
    full_content: Optional[str]
    chunk_type: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def quality(self) -> int:
        """Богатство метаданных — для выбора лучшего дубля."""
        q = len(self.content)
        if self.full_content:
            q += 200
        if int(self.metadata.get("page", 0) or 0) > 0:
            q += 500
        if self.metadata.get("model") in KNOWN_MODELS:
            q += 300
        if self.metadata.get("heading"):
            q += 100
        return q


@dataclass
class BuildReport:
    files: Dict[str, int] = field(default_factory=dict)
    records_in: int = 0
    skipped_noise: int = 0
    skipped_empty_table: int = 0
    reclassified_formula: int = 0
    downgraded_table: int = 0
    fixed_prefix: int = 0
    fixed_model: int = 0
    page_zero: int = 0
    deduplicated: int = 0
    kept: int = 0
    # П1: восстановление битого JSONL (раньше такие строки молча терялись)
    jsonl_lines: int = 0
    jsonl_repaired: int = 0
    jsonl_split: int = 0
    jsonl_recovered: int = 0
    jsonl_keys_backfilled: int = 0
    jsonl_unrecoverable: int = 0
    by_type: Counter = field(default_factory=Counter)
    by_model: Counter = field(default_factory=Counter)

    def as_dict(self) -> Dict[str, Any]:
        d = {k: v for k, v in self.__dict__.items() if k not in ("by_type", "by_model")}
        d["by_type"] = dict(self.by_type)
        d["by_model"] = dict(self.by_model)
        return d


def iter_jsonl(path: Path, report: Optional[BuildReport] = None) -> Iterable[Dict[str, Any]]:
    """П1: потоковое чтение JSONL с починкой битых строк.

    Раньше строки, не прошедшие json.loads(), молча отбрасывались (утечка
    данных). Теперь мусор до первой ``{`` срезается, слипшиеся ``}{`` честно
    разрезаются, многострочные объекты собираются. Невосстановимые строки
    считаются, но не роняют сборку.
    """
    rep = RepairReport()
    for obj in iter_jsonl_records(path, rep):
        if isinstance(obj, dict):
            yield obj
    if report is not None:
        report.jsonl_lines += rep.lines
        report.jsonl_repaired += rep.objects_repaired
        report.jsonl_split += rep.objects_split
        report.jsonl_recovered += rep.objects_recovered
        report.jsonl_keys_backfilled += rep.keys_backfilled
        report.jsonl_unrecoverable += rep.unrecoverable


def repair_record(obj: Dict[str, Any], src_file: str, kind: str, report: BuildReport) -> Optional[CleanChunk]:
    raw = obj.get("content")
    raw = raw if isinstance(raw, str) else str(raw or "")
    if _PREFIX_RE.match(raw):
        report.fixed_prefix += 1

    content = sanitize_chunk_content(raw)
    if not content or is_noise_text(content, min_alpha=2):
        report.skipped_noise += 1
        return None

    meta = dict(obj.get("metadata") or {})
    declared = str(obj.get("chunk_type") or meta.get("chunk_type") or "text").lower()
    if kind in ("chunks", "chunks_new", "bm25"):
        declared = declared if declared in {"text", "table", "formula", "figure"} else "text"

    source = str(meta.get("source") or "").strip()
    page = int(meta.get("page", 0) or 0)
    if page == 0:
        report.page_zero += 1
    if kind == "chunks_new" and not source and page == 0:
        meta["no_citation"] = True
        source = "источник_не_определён"

    # ремонт типа: ложные формулы и вырожденные таблицы → текст
    etype = normalized_element_type(declared, content, meta)
    if declared == "formula" and etype != "formula":
        report.reclassified_formula += 1
    if etype == "table":
        cells = all_table_cells(content)
        useful = len(cells) >= 2 and sum(len(c) for c in cells) >= 30
        if not useful:
            if len(cells) == 0:
                report.skipped_empty_table += 1
                return None
            etype = "text"
            report.downgraded_table += 1
    if etype == "formula" and not is_true_formula_text(content, str(meta.get("formula_type", ""))):
        etype = "figure" if CAPTION_RE.match(content) else "text"
        report.reclassified_formula += 1

    # ремонт модели
    raw_model = str(meta.get("model") or "unknown")
    model = resolve_model(raw_model, content, source)
    if model != raw_model:
        report.fixed_model += 1

    meta.update({
        "source": source,
        "page": page,
        "heading": str(meta.get("heading") or "").strip(),
        "model": model,
        "artifact_file": src_file,
        "artifact_kind": kind,
    })
    meta.pop("embedding", None)

    if etype == "table":
        chunk_content, full = generate_table_summary(content, meta), content
    elif etype == "formula":
        chunk_content, full = generate_formula_summary(content, meta), content
    elif etype == "figure":
        chunk_content, full = content, content
    else:
        chunk_content, full = content, None

    chunk_id = f"{etype}_{text_hash('|'.join([source, str(page), etype, content[:1000]]))}"
    return CleanChunk(chunk_id, chunk_content, full, etype, meta)


def load_clean_chunks(artifacts_dir: Path, include_export_new: bool, allow_bm25: bool, report: BuildReport) -> List[CleanChunk]:
    files = [p for p in artifacts_dir.glob("**/*") if p.is_file() and classify_file(p.name) != "unknown"]
    order = {"docstore": 0, "chunks": 1, "chunks_new": 2, "bm25": 3}
    files.sort(key=lambda p: order.get(classify_file(p.name), 9))

    seen: Dict[str, CleanChunk] = {}
    for path in files:
        kind = classify_file(path.name)
        if kind == "bm25" and not allow_bm25:
            continue
        if kind == "chunks_new" and not include_export_new:
            continue
        report.files.setdefault(path.name, 0)
        for obj in iter_jsonl(path, report):
            report.records_in += 1
            report.files[path.name] += 1
            chunk = repair_record(obj, path.name, kind, report)
            if chunk is None:
                continue
            key = text_hash((chunk.full_content or chunk.content).lower()[:2000] + "|" + chunk.chunk_type)
            existing = seen.get(key)
            if existing is None:
                seen[key] = chunk
            else:
                report.deduplicated += 1
                if chunk.quality > existing.quality:
                    seen[key] = chunk

    chunks = list(seen.values())
    for c in chunks:
        report.by_type[c.chunk_type] += 1
        report.by_model[c.metadata.get("model", "unknown")] += 1
    report.kept = len(chunks)
    return chunks


def build_index(chunks: List[CleanChunk], output_dir: Path) -> None:
    try:
        from src.chunking import Chunk
        from src.indexing import IndexBuilder
    except Exception as exc:  # pragma: no cover
        print(
            "\nОШИБКА импорта src.indexing/src.chunking: " + repr(exc) +
            "\n  Запустите из корня репозитория и установите зависимости:\n"
            "    pip install -r requirements.txt\n"
            "  Для аудита без сборки используйте --dry-run.",
            file=sys.stderr,
        )
        sys.exit(2)

    import shutil
    for sub in ("index", "docstore"):
        d = output_dir / sub
        if d.exists():
            shutil.rmtree(d)

    builder = IndexBuilder(output_dir / "index", docstore_path=output_dir / "docstore" / "docstore.db", load_existing=False)
    project_chunks = [Chunk(chunk_id=c.chunk_id, content=c.content, full_content=c.full_content, chunk_type=c.chunk_type, metadata=c.metadata) for c in chunks]
    BATCH = 500
    for i in range(0, len(project_chunks), BATCH):
        builder.add_chunks(project_chunks[i:i + BATCH])
        print(f"  …проиндексировано {min(i + BATCH, len(project_chunks))}/{len(project_chunks)}")
    builder.save()


def print_report(report: BuildReport, dry_run: bool) -> None:
    print("\n" + "═" * 62)
    print("  ОТЧЁТ СБОРКИ ИДЕАЛЬНОГО ИНДЕКСА" + ("  [сухой прогон]" if dry_run else ""))
    print("═" * 62)
    print(f"  Прочитано записей:           {report.records_in}")
    for fn, n in report.files.items():
        print(f"     • {fn}: {n}")
    print(f"  Снято префиксов passage/query:{report.fixed_prefix:>6}")
    print(f"  Формул → текст (ремонт типа): {report.reclassified_formula:>6}")
    print(f"  Таблиц → текст (вырожденные): {report.downgraded_table:>6}")
    print(f"  Исправлено поле модели:       {report.fixed_model:>6}")
    print(f"  Записей с page == 0:          {report.page_zero:>6}")
    print(f"  Отброшено (шум):              {report.skipped_noise:>6}")
    print(f"  Отброшено (пустые таблицы):   {report.skipped_empty_table:>6}")
    print(f"  Удалено дублей:               {report.deduplicated:>6}")
    print("  " + "─" * 58)
    print(f"  ИТОГО чистых чанков:          {report.kept:>6}")
    print(f"     по типам:  {dict(report.by_type)}")
    print(f"     по моделям:{dict(report.by_model)}")
    print("═" * 62)


def main() -> None:
    ap = argparse.ArgumentParser(description="Сборка идеального индекса KOIB RAG из артефактов.")
    ap.add_argument("--artifacts-dir", default="./data/artifacts")
    ap.add_argument("--output-dir", default="./output")
    ap.add_argument("--include-export-new", action="store_true", help="импортировать chunks_export_new (без цитат!)")
    ap.add_argument("--allow-bm25", action="store_true", help="импортировать bm25 (лемматизированный текст!)")
    ap.add_argument("--dry-run", action="store_true", help="только аудит, без сборки и без torch/faiss")
    ap.add_argument("--report-json", default="")
    args = ap.parse_args()

    artifacts_dir = Path(args.artifacts_dir).expanduser()
    if not artifacts_dir.exists():
        print(f"Папка артефактов не найдена: {artifacts_dir}", file=sys.stderr)
        sys.exit(1)

    report = BuildReport()
    chunks = load_clean_chunks(artifacts_dir, args.include_export_new, args.allow_bm25, report)
    print_report(report, args.dry_run)

    if args.report_json:
        Path(args.report_json).write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nОтчёт сохранён: {args.report_json}")

    if args.dry_run:
        print("\n[сухой прогон] Индекс НЕ собирался. Уберите --dry-run для боевой сборки.")
        return

    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nСтрою индекс в {output_dir} …")
    build_index(chunks, output_dir)
    print("\nГотово. Индексы: text_index.faiss + summary_index.faiss + bm25_fts.db + docstore.db")


if __name__ == "__main__":
    main()
