# -*- coding: utf-8 -*-
"""Import and repair previously generated RAG/OCR artifacts.

The uploaded project has three artifact families:
- ``chunks*.txt``: text chunks exported from FAISS/embedding pipeline;
- ``docstore*.txt``: table/formula/figure chunks with full content;
- ``bm25*.txt``: lemmatized BM25 text. This is not suitable for display and is off by default.

This module converts such JSONL files back to clean ``Chunk`` objects, stripping leaked
``passage:`` prefixes, reclassifying false formulas and deduplicating records.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from config import ARTIFACT_ALLOW_BM25_FALLBACK, ARTIFACT_MIN_CONTENT_CHARS
from .chunking import Chunk
from .text_processing import (
    artifact_kind,
    generate_formula_summary as _generate_formula_summary,
    generate_table_summary as _generate_table_summary,
    is_noise_text,
    is_supported_artifact,
    is_true_formula_text,
    normalized_element_type,
    sanitize_chunk_content,
)
from .utils import detect_model_in_text, normalize_ocr_text, text_hash

logger = logging.getLogger("koib.artifacts")


@dataclass
class ArtifactImportReport:
    files: int = 0
    records: int = 0
    imported: int = 0
    skipped: int = 0
    deduplicated: int = 0
    reclassified: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)

    def add_type(self, chunk_type: str) -> None:
        self.by_type[chunk_type] = self.by_type.get(chunk_type, 0) + 1


def discover_artifact_files(artifacts_dir: Path) -> List[Path]:
    if not artifacts_dir or not Path(artifacts_dir).exists():
        return []
    root = Path(artifacts_dir)
    files = [p for p in root.glob("**/*") if p.is_file() and is_supported_artifact(p)]
    priority = {"chunks": 0, "docstore": 1, "bm25": 2, "unknown": 3}
    return sorted(files, key=lambda p: (priority[artifact_kind(p.name)], str(p).lower()))


def _iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    """П1: чтение JSONL с восстановлением битых строк.

    Делегирует в :mod:`src.jsonl_repair`, чтобы строки с мусором/слипшимися
    объектами не терялись молча, как раньше.
    """
    from .jsonl_repair import iter_jsonl_records, RepairReport

    rep = RepairReport()
    for obj in iter_jsonl_records(path, rep):
        if isinstance(obj, dict):
            yield obj
    if rep.unrecoverable:
        logger.warning(
            "%s: %d строк не удалось восстановить (восстановлено: %d)",
            path.name, rep.unrecoverable, rep.objects_repaired + rep.objects_split + rep.objects_recovered,
        )


def _metadata(obj: Dict[str, Any]) -> Dict[str, Any]:
    meta = obj.get("metadata") or {}
    if not isinstance(meta, dict):
        meta = {}
    # Old chunks sometimes put essential fields at top level or nested metadata.chunk_id.
    if obj.get("chunk_id"):
        meta.setdefault("original_chunk_id", obj.get("chunk_id"))
    if meta.get("chunk_id") and not meta.get("original_chunk_id"):
        meta["original_chunk_id"] = meta.get("chunk_id")
    if obj.get("chunk_type"):
        meta.setdefault("chunk_type", obj.get("chunk_type"))
    return dict(meta)


def _model_for(content: str, metadata: Dict[str, Any]) -> str:
    model = str(metadata.get("model") or "unknown")
    if model and model != "unknown":
        return model
    detected, conf = detect_model_in_text(content)
    return detected if conf > 0.3 else "unknown"


def _table_is_useful(markdown: str) -> bool:
    text = sanitize_chunk_content(markdown)
    if len(text) < ARTIFACT_MIN_CONTENT_CHARS:
        return False
    if "|" not in text:
        return False
    cells = [c.strip() for c in text.replace("\n", "|").split("|")]
    non_empty = [c for c in cells if c and c != "---"]
    return len(non_empty) >= 2 and sum(len(c) for c in non_empty) >= ARTIFACT_MIN_CONTENT_CHARS


def _make_chunk(
    obj: Dict[str, Any],
    artifact_file: Path,
    kind: str,
    report: ArtifactImportReport,
    allow_bm25: bool,
) -> Optional[Chunk]:
    raw_content = obj.get("content") or ""
    if not isinstance(raw_content, str):
        raw_content = str(raw_content)
    content = normalize_ocr_text(raw_content)
    if not content or is_noise_text(content, min_alpha=2):
        report.skipped += 1
        return None

    metadata = _metadata(obj)
    original_type = str(obj.get("chunk_type") or metadata.get("chunk_type") or "text").lower()
    if kind == "chunks":
        original_type = "text"
    if kind == "bm25":
        if not allow_bm25:
            report.skipped += 1
            return None
        metadata["lemmatized_only"] = True
        original_type = "text"

    effective_type = normalized_element_type(original_type, content, metadata)
    if effective_type != original_type:
        report.reclassified += 1
    if effective_type == "table" and not _table_is_useful(content):
        # Bad table OCR is usually more useful as ordinary text than as an empty markdown table.
        effective_type = "text"
        report.reclassified += 1
    if effective_type == "formula" and not is_true_formula_text(content, str(metadata.get("formula_type", ""))):
        effective_type = "text"
        report.reclassified += 1

    metadata.update(
        {
            "source": metadata.get("source") or artifact_file.name,
            "page": int(metadata.get("page") or 0),
            "heading": metadata.get("heading") or "",
            "model": _model_for(content, metadata),
            "artifact_file": artifact_file.name,
            "artifact_kind": kind,
            "artifact_imported": True,
        }
    )
    metadata.pop("embedding", None)

    if effective_type == "table":
        chunk_content = _generate_table_summary(content, metadata)
        full_content: Optional[str] = content
    elif effective_type == "formula":
        chunk_content = _generate_formula_summary(content, metadata)
        full_content = content
    elif effective_type == "figure":
        chunk_content = content
        full_content = content
    else:
        chunk_content = content
        full_content = None

    # Avoid using old ids because some figure ids were duplicated in exported artifacts.
    id_material = "|".join(
        [
            str(metadata.get("source", "")),
            str(metadata.get("page", "")),
            effective_type,
            str(metadata.get("original_chunk_id", "")),
            str(metadata.get("image_path", "")),
            content[:1000],
        ]
    )
    chunk_id = f"{effective_type}_{text_hash(id_material)}"
    return Chunk(chunk_id=chunk_id, content=chunk_content, full_content=full_content, chunk_type=effective_type, metadata=metadata)


def load_artifact_chunks(
    artifacts_dir: Path,
    allow_bm25_fallback: Optional[bool] = None,
) -> Tuple[List[Chunk], ArtifactImportReport]:
    """Load and repair all supported artifacts from a directory."""
    allow_bm25 = ARTIFACT_ALLOW_BM25_FALLBACK if allow_bm25_fallback is None else allow_bm25_fallback
    files = discover_artifact_files(Path(artifacts_dir))
    report = ArtifactImportReport(files=len(files))
    chunks: List[Chunk] = []
    seen: Set[Tuple[str, int, str, str]] = set()
    seen_ids: Set[str] = set()

    for file_path in files:
        kind = artifact_kind(file_path.name)
        for obj in _iter_jsonl(file_path):
            report.records += 1
            chunk = _make_chunk(obj, file_path, kind, report, allow_bm25=allow_bm25)
            if chunk is None:
                continue
            key = (
                str(chunk.metadata.get("source", "")),
                int(chunk.metadata.get("page", 0) or 0),
                chunk.chunk_type,
                text_hash((chunk.full_content or chunk.content)[:2000]),
            )
            if key in seen:
                report.deduplicated += 1
                continue
            seen.add(key)
            if chunk.chunk_id in seen_ids:
                suffix = text_hash(str(len(seen_ids)) + chunk.chunk_id)
                chunk.chunk_id = f"{chunk.chunk_id}_{suffix[:6]}"
            seen_ids.add(chunk.chunk_id)
            chunks.append(chunk)
            report.imported += 1
            report.add_type(chunk.chunk_type)

    return chunks, report
