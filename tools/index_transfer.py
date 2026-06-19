# -*- coding: utf-8 -*-
"""
index_transfer.py — проверка, упаковка и перенос индекса КОИБ
=============================================================

Единый инструмент для переноса готового индекса с машины-индексатора на сервер.
Вся логика проверки/экспорта индекса живёт ТОЛЬКО здесь (раньше дублировалась
в export_index.py).

Подкоманды
----------
  verify   — проверить целостность и согласованность индекса в ./output
  export   — выгрузить чанки обратно в JSONL (из docstore или bm25)
  pack     — собрать переносимый архив index_bundle.zip из ./output
             (+ манифест: модель эмбеддингов, префиксы, размерность, контрольные суммы)
  unpack   — распаковать архив в ./output на сервере и проверить совместимость
             манифеста с текущим .env (модель/префиксы ДОЛЖНЫ совпадать)

Примеры
-------
    # на индексаторе:
    python -m tools.index_transfer verify  --output-dir ./output
    python -m tools.index_transfer pack    --output-dir ./output --out index_bundle.zip

    # на сервере:
    python -m tools.index_transfer unpack  --archive index_bundle.zip --output-dir ./output

Зависит только от стандартной библиотеки. faiss подключается опционально.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

# ── Имена файлов индекса (единая точка истины о структуре output/) ────────────
DOCSTORE_REL = Path("docstore") / "docstore.db"
BM25_REL = Path("index") / "bm25_fts.db"
TEXT_FAISS_REL = Path("index") / "text_index.faiss"
TEXT_PKL_REL = Path("index") / "text_index.pkl"
SUMMARY_FAISS_REL = Path("index") / "summary_index.faiss"
SUMMARY_PKL_REL = Path("index") / "summary_index.pkl"
MANIFEST_REL = Path("index_manifest.json")

# что НЕ переносим: рантайм-данные сервера (кэш, логи, метаданные).
# figures переносим — они нужны серверу, чтобы прикладывать рисунки к ответам.
PACK_EXCLUDE_DIRS = {"metadata", "logs"}


# ─────────────────────────────────────────────────────────────────────────────
# Чтение хранилищ
# ─────────────────────────────────────────────────────────────────────────────
def _connect_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def read_docstore(db_path: Path) -> Iterator[Dict[str, Any]]:
    conn = _connect_ro(db_path)
    try:
        cur = conn.execute("SELECT chunk_id, content, chunk_type, metadata FROM docstore")
        for chunk_id, content, chunk_type, metadata in cur:
            meta = json.loads(metadata) if metadata else {}
            yield {
                "chunk_id": chunk_id,
                "content": content,
                "chunk_type": chunk_type or meta.get("chunk_type", "text"),
                "metadata": meta,
            }
    finally:
        conn.close()


def read_bm25(db_path: Path) -> Iterator[Dict[str, Any]]:
    conn = _connect_ro(db_path)
    try:
        cur = conn.execute("SELECT chunk_id, raw_content, chunk_type, metadata FROM chunks_fts")
        for chunk_id, raw_content, chunk_type, metadata in cur:
            meta = json.loads(metadata) if metadata else {}
            yield {
                "chunk_id": chunk_id,
                "content": raw_content,
                "chunk_type": chunk_type or meta.get("chunk_type", "text"),
                "metadata": meta,
            }
    finally:
        conn.close()


def count_rows(db_path: Path, table: str) -> int:
    conn = _connect_ro(db_path)
    try:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    finally:
        conn.close()


def faiss_ntotal(faiss_path: Path) -> Optional[int]:
    if not faiss_path.exists():
        return None
    try:
        import faiss  # type: ignore
    except Exception:
        return None
    try:
        return int(faiss.read_index(str(faiss_path)).ntotal)
    except Exception:
        return None


def _sha256(path: Path, limit_mb: int = 0) -> str:
    h = hashlib.sha256()
    read = 0
    cap = limit_mb * 1024 * 1024 if limit_mb else 0
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
            read += len(block)
            if cap and read >= cap:
                break
    return h.hexdigest()


# ─────────────────────────────────────────────────────────────────────────────
# Проверка
# ─────────────────────────────────────────────────────────────────────────────
def verify(output_dir: Path) -> Tuple[bool, Dict[str, Any]]:
    docstore_db = output_dir / DOCSTORE_REL
    bm25_db = output_dir / BM25_REL
    text_faiss = output_dir / TEXT_FAISS_REL
    summary_faiss = output_dir / SUMMARY_FAISS_REL

    print("═" * 62)
    print("  ПРОВЕРКА ИНДЕКСА:", output_dir)
    print("═" * 62)

    info: Dict[str, Any] = {}
    ok = True
    for label, p in [
        ("docstore.db", docstore_db),
        ("bm25_fts.db", bm25_db),
        ("text_index.faiss", text_faiss),
        ("summary_index.faiss", summary_faiss),
    ]:
        exists = p.exists()
        size_kb = (p.stat().st_size // 1024) if exists else 0
        print(f"  [{'✓' if exists else '✗'}] {label:<22} {size_kb:>8} КБ")
        info[label] = {"exists": exists, "size_kb": size_kb}
        if label in ("docstore.db", "bm25_fts.db") and not exists:
            ok = False

    if not docstore_db.exists():
        print("\n  ОШИБКА: docstore.db не найден — индекс не собран.")
        return False, info

    ds_count = count_rows(docstore_db, "docstore")
    bm_count = count_rows(bm25_db, "chunks_fts") if bm25_db.exists() else 0
    text_vecs = faiss_ntotal(text_faiss)
    sum_vecs = faiss_ntotal(summary_faiss)

    print("  " + "─" * 58)
    print(f"  Чанков в DocStore:            {ds_count}")
    print(f"  Чанков в BM25 (FTS5):         {bm_count}")
    print(f"  Векторов text_index:          {text_vecs if text_vecs is not None else 'faiss не установлен'}")
    print(f"  Векторов summary_index:       {sum_vecs if sum_vecs is not None else 'faiss не установлен'}")
    info.update({"docstore_chunks": ds_count, "bm25_chunks": bm_count,
                 "text_vectors": text_vecs, "summary_vectors": sum_vecs})

    print("  " + "─" * 58)
    if bm25_db.exists() and abs(ds_count - bm_count) > max(5, ds_count * 0.02):
        print(f"  ⚠ DocStore и BM25 расходятся ({ds_count} vs {bm_count}).")
        ok = False
    else:
        print("  ✓ DocStore и BM25 согласованы.")

    if text_vecs is not None and sum_vecs is not None:
        total = text_vecs + sum_vecs
        mark = "✓" if abs(total - ds_count) <= max(5, ds_count * 0.05) else "⚠"
        print(f"  {mark} Векторов всего {total} ≈ DocStore {ds_count}.")

    types: Counter = Counter()
    empty = leaked = 0
    for rec in read_docstore(docstore_db):
        types[rec["chunk_type"]] += 1
        c = (rec["content"] or "").strip()
        if not c:
            empty += 1
        if c.lower().startswith(("passage:", "query:")):
            leaked += 1
    print("  " + "─" * 58)
    print(f"  По типам:   {dict(types)}")
    print(f"  Пустой content: {empty}  |  утёкший префикс: {leaked}")
    info.update({"by_type": dict(types), "empty_content": empty, "leaked_prefix": leaked})
    if empty or leaked:
        print("  ⚠ Найдены пустые/префиксные чанки — стоит пересобрать индекс.")
        ok = False

    print("═" * 62)
    print("  ИТОГ:", "✓ индекс корректен" if ok else "⚠ есть замечания (см. выше)")
    print("═" * 62)
    return ok, info


# ─────────────────────────────────────────────────────────────────────────────
# Экспорт в JSONL
# ─────────────────────────────────────────────────────────────────────────────
def _seq_lookup(docstore_db: Path) -> Dict[str, int]:
    out: Dict[str, int] = {}
    try:
        conn = _connect_ro(docstore_db)
    except Exception:
        return out
    try:
        for chunk_id, seq in conn.execute("SELECT chunk_id, seq FROM chunk_order"):
            if chunk_id is not None and seq is not None:
                out[str(chunk_id)] = int(seq)
    except Exception:
        pass
    finally:
        conn.close()
    return out


def _normalize_for_export(rec: Dict[str, Any], seq_map: Dict[str, int]) -> Dict[str, Any]:
    cid = rec.get("chunk_id") or rec.get("id") or ""
    content = rec.get("content") or rec.get("full_content") or ""
    meta = rec.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    meta.setdefault("model", meta.get("model", "unknown") or "unknown")
    if cid in seq_map and "seq" not in meta:
        meta["seq"] = seq_map[cid]
    return {
        "chunk_id": str(cid),
        "id": str(cid),
        "content": content,
        "chunk_type": rec.get("chunk_type") or meta.get("chunk_type", "text"),
        "metadata": meta,
    }


def _safe_write_line(handle, rec: Dict[str, Any]) -> bool:
    line = json.dumps(rec, ensure_ascii=False)
    try:
        json.loads(line)
    except Exception:
        return False
    handle.write(line + "\n")
    return True


def export_jsonl(output_dir: Path, out_path: Path, source: str, by_type: bool) -> None:
    docstore_db = output_dir / DOCSTORE_REL
    bm25_db = output_dir / BM25_REL
    if source == "bm25":
        if not bm25_db.exists():
            sys.exit("bm25_fts.db не найден.")
        reader = read_bm25(bm25_db)
    else:
        if not docstore_db.exists():
            sys.exit("docstore.db не найден.")
        reader = read_docstore(docstore_db)

    seq_map = _seq_lookup(docstore_db) if docstore_db.exists() else {}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    bad = 0
    if by_type:
        writers: Dict[str, Any] = {}
        counts: Counter = Counter()
        stem = out_path.with_suffix("")
        try:
            for rec in reader:
                rec = _normalize_for_export(rec, seq_map)
                t = rec["chunk_type"] or "text"
                if t not in writers:
                    writers[t] = Path(f"{stem}.{t}.jsonl").open("w", encoding="utf-8")
                counts[t] += 1 if _safe_write_line(writers[t], rec) else 0
        finally:
            for w in writers.values():
                w.close()
        print(f"\nЭкспортировано {sum(counts.values())} чанков (источник: {source}):")
        for t, n in counts.most_common():
            print(f"   • {Path(f'{stem}.{t}.jsonl').name}: {n}")
    else:
        n = 0
        with out_path.open("w", encoding="utf-8") as f:
            for rec in reader:
                if _safe_write_line(f, _normalize_for_export(rec, seq_map)):
                    n += 1
                else:
                    bad += 1
        print(f"\nЭкспортировано {n} чанков (источник: {source}) → {out_path}")
    if bad:
        print(f"  ⚠ Пропущено несериализуемых записей: {bad}")


# ─────────────────────────────────────────────────────────────────────────────
# Манифест совместимости
# ─────────────────────────────────────────────────────────────────────────────
def build_manifest(output_dir: Path) -> Dict[str, Any]:
    """Снимок параметров, при которых собран индекс.

    Эти параметры на сервере ДОЛЖНЫ совпадать, иначе вектор запроса не сойдётся
    с векторами индекса. Манифест позволяет проверить это автоматически.
    """
    try:
        import config  # type: ignore
        emb_model = getattr(config, "LOCAL_EMBEDDING_MODEL", os.getenv("LOCAL_EMBEDDING_MODEL", ""))
        emb_provider = getattr(config, "EMBEDDING_PROVIDER", os.getenv("EMBEDDING_PROVIDER", "local"))
        passage_prefix = getattr(config, "PASSAGE_PREFIX", os.getenv("PASSAGE_PREFIX", ""))
        query_prefix = getattr(config, "QUERY_PREFIX", os.getenv("QUERY_PREFIX", ""))
        version = getattr(config, "APP_VERSION", os.getenv("KOIB_VERSION", ""))
    except Exception:
        emb_model = os.getenv("LOCAL_EMBEDDING_MODEL", "")
        emb_provider = os.getenv("EMBEDDING_PROVIDER", "local")
        passage_prefix = os.getenv("PASSAGE_PREFIX", "")
        query_prefix = os.getenv("QUERY_PREFIX", "")
        version = os.getenv("KOIB_VERSION", "")

    docstore_db = output_dir / DOCSTORE_REL
    text_faiss = output_dir / TEXT_FAISS_REL

    files: Dict[str, Dict[str, Any]] = {}
    for rel in (DOCSTORE_REL, BM25_REL, TEXT_FAISS_REL, TEXT_PKL_REL, SUMMARY_FAISS_REL, SUMMARY_PKL_REL):
        p = output_dir / rel
        if p.exists():
            files[str(rel).replace(os.sep, "/")] = {
                "size": p.stat().st_size,
                "sha256": _sha256(p),
            }

    return {
        "koib_version": version,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedding": {
            "provider": emb_provider,
            "model": emb_model,
            "passage_prefix": passage_prefix,
            "query_prefix": query_prefix,
            "dim": _faiss_dim(text_faiss),
        },
        "counts": {
            "docstore_chunks": count_rows(docstore_db, "docstore") if docstore_db.exists() else 0,
            "text_vectors": faiss_ntotal(text_faiss),
        },
        "files": files,
    }


def _faiss_dim(faiss_path: Path) -> Optional[int]:
    try:
        import faiss  # type: ignore
        return int(faiss.read_index(str(faiss_path)).d)
    except Exception:
        return None


def _iter_pack_files(output_dir: Path) -> Iterator[Path]:
    for p in output_dir.rglob("*"):
        if not p.is_file():
            continue
        parts = p.relative_to(output_dir).parts
        if parts and parts[0] in PACK_EXCLUDE_DIRS:
            continue
        if p.name.startswith(".") or p.suffix == ".zip":
            continue
        yield p


# ─────────────────────────────────────────────────────────────────────────────
# pack / unpack
# ─────────────────────────────────────────────────────────────────────────────
def pack(output_dir: Path, archive: Path) -> None:
    if not (output_dir / DOCSTORE_REL).exists():
        sys.exit(f"Не найден {output_dir / DOCSTORE_REL} — индекс не собран.")
    print("Сборка манифеста совместимости…")
    manifest = build_manifest(output_dir)
    (output_dir / MANIFEST_REL).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  модель: {manifest['embedding']['model']}  | чанков: {manifest['counts']['docstore_chunks']}")

    archive.parent.mkdir(parents=True, exist_ok=True)
    files = list(_iter_pack_files(output_dir))
    total = 0
    print(f"Упаковка {len(files)} файлов → {archive}")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in files:
            arc = Path("output") / p.relative_to(output_dir)
            zf.write(p, arcname=str(arc).replace(os.sep, "/"))
            total += p.stat().st_size
    size_mb = archive.stat().st_size / 1024 / 1024
    print(f"✓ Архив готов: {archive}  ({size_mb:.1f} МБ, исходно {total/1024/1024:.1f} МБ)")
    print("  Перенесите его на сервер (scp/rsync) и распакуйте:")
    print(f"      python -m tools.index_transfer unpack --archive {archive.name} --output-dir ./output")


def _check_compat(manifest: Dict[str, Any]) -> bool:
    """Сверяет манифест с текущим окружением сервера. True = совместимо."""
    emb = manifest.get("embedding", {})
    cur_model = os.getenv("LOCAL_EMBEDDING_MODEL", "")
    cur_pp = os.getenv("PASSAGE_PREFIX", "")
    cur_qp = os.getenv("QUERY_PREFIX", "")
    try:
        import config  # type: ignore
        cur_model = getattr(config, "LOCAL_EMBEDDING_MODEL", cur_model)
        cur_pp = getattr(config, "PASSAGE_PREFIX", cur_pp)
        cur_qp = getattr(config, "QUERY_PREFIX", cur_qp)
    except Exception:
        pass

    ok = True
    print("  " + "─" * 58)
    print("  ПРОВЕРКА СОВМЕСТИМОСТИ с текущим .env сервера:")
    for label, was, now in (
        ("модель эмбеддингов", emb.get("model", ""), cur_model),
        ("passage_prefix", emb.get("passage_prefix", ""), cur_pp),
        ("query_prefix", emb.get("query_prefix", ""), cur_qp),
    ):
        match = (str(was).strip() == str(now).strip())
        print(f"    [{'✓' if match else '✗'}] {label}: индекс='{was}'  сервер='{now}'")
        ok = ok and match
    if not ok:
        print("  ⚠ ВНИМАНИЕ: параметры РАСХОДЯТСЯ. Поиск будет работать плохо!")
        print("    Приведите .env сервера в соответствие с манифестом и перезапустите.")
    else:
        print("  ✓ Параметры совпадают — индекс совместим с сервером.")
    return ok


def _normalize_figure_path(stored: str) -> str:
    """Привести сохранённый image_path к переносимому «figures/<имя>».

    Любой исходный формат (абсолютный Windows-путь, «output/figures/...»,
    «figures/...», голое имя) сводится к «figures/<basename>».
    """
    if not stored:
        return ""
    s = str(stored).strip().replace("\\", "/")
    low = s.lower()
    idx = low.find("figures/")
    if idx >= 0:
        return s[idx:]  # «figures/.../name.png»
    return "figures/" + os.path.basename(s)


def rebase_figure_paths(output_dir: Path) -> int:
    """Перевести image_path в метаданных чанков к виду «figures/...png».

    Старые индексы хранили абсолютный путь машины-индексатора. После распаковки
    на сервере переписываем его на переносимый относительный путь, чтобы рисунки
    разрешались через config.resolve_figure_path на любой машине. Чанки без
    image_path и уже относительные пути остаются как есть (нормализация
    идемпотентна).
    """
    changed = 0
    for db_rel, table, meta_col in (
        (DOCSTORE_REL, "docstore", "metadata"),
        (BM25_REL, "chunks_fts", "metadata"),
    ):
        db_path = output_dir / db_rel
        if not db_path.exists():
            continue
        conn = sqlite3.connect(str(db_path))
        try:
            rows = conn.execute(f"SELECT chunk_id, {meta_col} FROM {table}").fetchall()
            updates = []
            for chunk_id, meta_json in rows:
                if not meta_json:
                    continue
                try:
                    meta = json.loads(meta_json)
                except Exception:
                    continue
                ip = meta.get("image_path")
                if not ip or not str(ip).strip():
                    continue
                normalized = _normalize_figure_path(str(ip))
                if normalized and normalized != str(ip).replace("\\", "/"):
                    meta["image_path"] = normalized
                    updates.append((json.dumps(meta, ensure_ascii=False), chunk_id))
            if updates:
                conn.executemany(f"UPDATE {table} SET {meta_col}=? WHERE chunk_id=?", updates)
                conn.commit()
                changed += len(updates)
        finally:
            conn.close()
    return changed


def unpack(archive: Path, output_dir: Path, force: bool = False) -> None:
    if not archive.exists():
        sys.exit(f"Архив не найден: {archive}")
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = output_dir / DOCSTORE_REL
    if existing.exists() and not force:
        sys.exit(f"В {output_dir} уже есть индекс. Добавьте --force для перезаписи.")

    print(f"Распаковка {archive} → {output_dir.parent}")
    with zipfile.ZipFile(archive) as zf:
        # архив содержит префикс output/...; распаковываем в родителя output_dir
        target_root = output_dir.parent
        for name in zf.namelist():
            # защита от path traversal
            dest = (target_root / name).resolve()
            if not str(dest).startswith(str(target_root.resolve())):
                sys.exit(f"Небезопасный путь в архиве: {name}")
        zf.extractall(target_root)

    # Перенос путей рисунков под текущий OUTPUT_DIR (Проблема 4):
    # старые абсолютные пути машины-индексатора → «figures/...png».
    try:
        rebased = rebase_figure_paths(output_dir)
        if rebased:
            print(f"  Переписано image_path → относительные пути: {rebased} чанков")
    except Exception as exc:
        print(f"  ⚠ Не удалось переписать пути рисунков: {exc}")

    manifest_path = output_dir / MANIFEST_REL
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        print(f"✓ Распаковано. Индекс v{manifest.get('koib_version','?')}, "
              f"чанков: {manifest.get('counts',{}).get('docstore_chunks','?')}")
        _check_compat(manifest)
    else:
        print("✓ Распаковано (без манифеста — проверьте параметры эмбеддингов вручную).")

    print("  " + "─" * 58)
    verify(output_dir)


# ─────────────────────────────────────────────────────────────────────────────
def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description="Проверка, экспорт и перенос индекса КОИБ.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_v = sub.add_parser("verify", help="проверить индекс")
    p_v.add_argument("--output-dir", default="./output")
    p_v.add_argument("--report-json", default="")

    p_e = sub.add_parser("export", help="выгрузить чанки в JSONL")
    p_e.add_argument("--output-dir", default="./output")
    p_e.add_argument("--out", required=True)
    p_e.add_argument("--source", choices=["docstore", "bm25"], default="docstore")
    p_e.add_argument("--by-type", action="store_true")

    p_p = sub.add_parser("pack", help="собрать переносимый архив индекса")
    p_p.add_argument("--output-dir", default="./output")
    p_p.add_argument("--out", default="index_bundle.zip")

    p_u = sub.add_parser("unpack", help="распаковать архив на сервере и проверить совместимость")
    p_u.add_argument("--archive", required=True)
    p_u.add_argument("--output-dir", default="./output")
    p_u.add_argument("--force", action="store_true")

    args = ap.parse_args(argv)

    if args.cmd == "verify":
        ok, info = verify(Path(args.output_dir).expanduser())
        if args.report_json:
            Path(args.report_json).write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
            print("Отчёт сохранён:", args.report_json)
        sys.exit(0 if ok else 2)
    elif args.cmd == "export":
        export_jsonl(Path(args.output_dir).expanduser(), Path(args.out).expanduser(),
                     args.source, args.by_type)
    elif args.cmd == "pack":
        pack(Path(args.output_dir).expanduser(), Path(args.out).expanduser())
    elif args.cmd == "unpack":
        unpack(Path(args.archive).expanduser(), Path(args.output_dir).expanduser(), force=args.force)


if __name__ == "__main__":
    main()
