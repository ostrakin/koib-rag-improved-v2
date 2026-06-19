# -*- coding: utf-8 -*-
"""Batch indexing for KOIB RAG."""
from __future__ import annotations

import concurrent.futures
import hashlib
import json
import logging
import multiprocessing
import shutil
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.getLogger("pymorphy2").setLevel(logging.WARNING)
logging.getLogger("pymorphy2.opencorpora_dict").setLevel(logging.WARNING)

from config import ARTIFACTS_DIR, DOCS_DIR, INGEST_MAX_WORKERS, OUTPUT_DIR, ensure_dirs
from src.artifacts import load_artifact_chunks
from src.chunking import SmartChunker
from src.indexing import IndexBuilder
from src.parsing import parse_document

logger = logging.getLogger("koib.ingest")
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".csv", ".txt", ".md"}


def _sha256_file(path: Path, block_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(block_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except Exception:
        return path.name


def _process_file_task(file_path: Path) -> Tuple[Path, List, bool, str]:
    try:
        elements = parse_document(file_path)
        if not elements:
            return file_path, [], False, "пустой результат парсинга"
        chunks = SmartChunker().chunk_elements(elements)
        if not chunks:
            return file_path, [], False, "не создано чанков"
        # П2 (без LLM, всегда): обогащаем слабые подписи рисунков/формул
        # контекстом соседнего текста той же страницы, чтобы они находились
        # поиском даже без мультимодальной модели.
        try:
            from src.figure_context import enrich_figures_with_context
            enrich_figures_with_context(chunks)  # мутирует chunks на месте
        except Exception:
            logger.exception("Обогащение рисунков контекстом пропущено для %s", file_path)
        # П7: описания изображений генерируются ТОЛЬКО на этапе индексации
        # (тяжёлый Vision-вызов), и только если включён флаг в конфиге. Vision
        # отрабатывает ПОСЛЕ текстового обогащения и перезаписывает слабые подписи.
        try:
            from config import FIGURE_CAPTIONING_ENABLED
            if FIGURE_CAPTIONING_ENABLED:
                from src.figure_captioning import caption_figures_in_chunks
                caption_figures_in_chunks(chunks)  # мутирует chunks на месте
        except Exception:
            logger.exception("Captioning изображений пропущен для %s", file_path)
        return file_path, chunks, True, ""
    except Exception as exc:
        logger.exception("Ошибка обработки %s", file_path)
        return file_path, [], False, str(exc)


class BatchIngester:
    def __init__(
        self,
        docs_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        incremental: bool = True,
        artifacts_dir: Optional[Path] = None,
        ingest_artifacts: bool = False,
    ):
        self.docs_dir = Path(docs_dir or DOCS_DIR)
        self.output_dir = Path(output_dir or OUTPUT_DIR)
        self.artifacts_dir = Path(artifacts_dir or ARTIFACTS_DIR)
        self.ingest_artifacts = ingest_artifacts
        self.incremental = incremental
        self.manifest_path = self.output_dir / "metadata" / "ingest_manifest.json"
        self._manifest: Dict[str, Dict[str, object]] = {}
        ensure_dirs()
        if incremental:
            self._load_manifest()
        else:
            self._reset_indexes()
        self.index_builder = IndexBuilder(
            self.output_dir / "index",
            docstore_path=self.output_dir / "docstore" / "docstore.db",
            load_existing=incremental,
        )

    def _reset_indexes(self) -> None:
        for rel in ["index", "docstore", "figures"]:
            target = self.output_dir / rel
            if target.exists():
                shutil.rmtree(target)
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            self.manifest_path.unlink()

    def _load_manifest(self) -> None:
        if not self.manifest_path.exists():
            self._manifest = {}
            return
        try:
            self._manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Не удалось прочитать manifest, будет полный rebuild: %s", exc)
            self._manifest = {}

    def _save_manifest(self) -> None:
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.manifest_path.write_text(json.dumps(self._manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    def _fingerprint(self, file_path: Path, root: Optional[Path] = None, kind: str = "doc") -> Dict[str, object]:
        stat = file_path.stat()
        root = Path(root or self.docs_dir)
        rel = _safe_rel(file_path, root)
        return {
            "kind": kind,
            "path": rel,
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _sha256_file(file_path),
        }

    def _discover_files(self) -> List[Path]:
        if not self.docs_dir.exists():
            return []
        all_files: List[Path] = []
        for ext in SUPPORTED_EXTENSIONS:
            all_files.extend(self.docs_dir.glob(f"**/*{ext}"))
            all_files.extend(self.docs_dir.glob(f"**/*{ext.upper()}"))
        return sorted({p.resolve(): p for p in all_files if p.is_file()}.values(), key=lambda p: str(p).lower())

    def _discover_artifacts(self) -> List[Path]:
        if not self.ingest_artifacts:
            return []
        from src.artifacts import discover_artifact_files

        return discover_artifact_files(self.artifacts_dir)

    def _current_manifest(self, docs: List[Path], artifacts: List[Path]) -> Dict[str, Dict[str, object]]:
        current: Dict[str, Dict[str, object]] = {}
        for p in docs:
            rel = _safe_rel(p, self.docs_dir)
            current[f"doc:{rel}"] = self._fingerprint(p, self.docs_dir, kind="doc")
        for p in artifacts:
            rel = _safe_rel(p, self.artifacts_dir)
            current[f"artifact:{rel}"] = self._fingerprint(p, self.artifacts_dir, kind="artifact")
        return current

    def _select_files(self, all_files: List[Path], artifact_files: List[Path]) -> Tuple[List[Path], List[Path], bool]:
        if not self.incremental or not self._manifest:
            return all_files, artifact_files, False

        current = self._current_manifest(all_files, artifact_files)
        removed = set(self._manifest) - set(current)
        changed = [key for key, fp in current.items() if key in self._manifest and self._manifest[key].get("sha256") != fp["sha256"]]
        if removed or changed:
            print("  Обнаружены изменённые/удалённые документы или артефакты. Для целостности FAISS будет выполнен полный rebuild.")
            self._reset_indexes()
            self._manifest = {}
            self.index_builder = IndexBuilder(
                self.output_dir / "index",
                docstore_path=self.output_dir / "docstore" / "docstore.db",
                load_existing=False,
            )
            return all_files, artifact_files, True

        new_docs = [p for p in all_files if f"doc:{_safe_rel(p, self.docs_dir)}" not in self._manifest]
        new_artifacts = [p for p in artifact_files if f"artifact:{_safe_rel(p, self.artifacts_dir)}" not in self._manifest]
        return new_docs, new_artifacts, False

    def _process_documents(self, files: List[Path]) -> Tuple[int, int]:
        if not files:
            return 0, 0
        print(f"  Обнаружено документов для индексации: {len(files)}")
        success_count = error_count = 0
        cpu_count = multiprocessing.cpu_count() or 1
        max_workers = INGEST_MAX_WORKERS or min(8, max(2, cpu_count))
        print(f"  Запуск параллельного парсинга в {max_workers} потоков (CPU: {cpu_count})...")

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_file = {executor.submit(_process_file_task, fp): fp for fp in files}
            for i, future in enumerate(concurrent.futures.as_completed(future_to_file), 1):
                file_path = future_to_file[future]
                print(f"  [{i}/{len(files)}] {file_path.name}...", end=" ", flush=True)
                fp, chunks, success, error = future.result()
                if success and chunks:
                    print(f"[{len(chunks)} чанков] ", end="", flush=True)
                    self.index_builder.add_chunks(chunks)
                    key = f"doc:{_safe_rel(fp, self.docs_dir)}"
                    self._manifest[key] = self._fingerprint(fp, self.docs_dir, kind="doc")
                    self._manifest[key]["chunks"] = len(chunks)
                    success_count += 1
                    print("OK")
                else:
                    error_count += 1
                    print(f"ОШИБКА ({error})")
                if i % 10 == 0:
                    self._save_manifest()
        return success_count, error_count

    def _process_artifacts(self, artifact_files: List[Path]) -> Tuple[int, int]:
        if not artifact_files:
            return 0, 0
        print(f"  Импорт артефактов распознавания/старого индекса: {len(artifact_files)} файлов")
        chunks, report = load_artifact_chunks(self.artifacts_dir)
        if not chunks:
            print(f"  Артефакты не дали пригодных чанков: records={report.records}, skipped={report.skipped}")
            return 0, len(artifact_files)
        self.index_builder.add_chunks(chunks)
        for fp in artifact_files:
            key = f"artifact:{_safe_rel(fp, self.artifacts_dir)}"
            self._manifest[key] = self._fingerprint(fp, self.artifacts_dir, kind="artifact")
        print(
            "  Артефакты: imported={imported}, skipped={skipped}, dedup={deduplicated}, "
            "reclassified={reclassified}, types={types}".format(
                imported=report.imported,
                skipped=report.skipped,
                deduplicated=report.deduplicated,
                reclassified=report.reclassified,
                types=report.by_type,
            )
        )
        return 1, 0

    def process_all(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        all_files = self._discover_files()
        artifact_files = self._discover_artifacts()
        if not all_files and not artifact_files:
            print("  Файлы для индексации не найдены.")
            if self.ingest_artifacts:
                print(f"  Проверьте документы в {self.docs_dir} и артефакты в {self.artifacts_dir}.")
            return

        files, artifacts, rebuild = self._select_files(all_files, artifact_files)
        if not files and not artifacts:
            print("  Новых файлов для индексации не найдено.")
            return

        doc_success, doc_errors = self._process_documents(files)
        art_success, art_errors = self._process_artifacts(artifacts)
        self._save_manifest()
        self.index_builder.save()
        mode = "полный rebuild" if rebuild or not self.incremental else "incremental"
        print(f"\nРезультат ({mode}): документы OK={doc_success}, документы errors={doc_errors}, артефакты OK={art_success}, артефакты errors={art_errors}")
        print(f"  Индексы сохранены в {self.output_dir}. Время сборки: {time.time() - t0:.1f}с")
