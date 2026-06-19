# -*- coding: utf-8 -*-
"""KOIB RAG — приложение индексации (запускается на мощном компьютере).

Полный набор тяжёлых операций: парсинг документов, OCR, чанкинг, построение
FAISS + BM25 + DocStore, импорт старых артефактов, оценка качества, а также
проверка и УПАКОВКА готового индекса для переноса на сервер.

Команды:
    ingest      — проиндексировать документы (по умолчанию инкрементально)
    evaluate    — прогнать набор вопросов и посчитать метрики
    verify      — проверить целостность собранного индекса
    export      — выгрузить чанки в JSONL
    pack        — собрать переносимый архив индекса (index_bundle.zip)

Примеры:
    python index_app.py ingest --rebuild
    python index_app.py ingest --ingest-artifacts --artifacts-dir ./data/artifacts
    python index_app.py verify
    python index_app.py pack --out index_bundle.zip
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import (
    APP_VERSION,
    ARTIFACTS_DIR,
    DOCS_DIR,
    FINAL_TOP_K,
    HF_OFFLINE_MODE,
    OUTPUT_DIR,
    ensure_dirs,
)

if HF_OFFLINE_MODE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("koib.index")


def cmd_ingest(args) -> None:
    from batch_ingest import BatchIngester

    BatchIngester(
        Path(args.docs_dir),
        Path(args.output_dir),
        incremental=not args.rebuild,
        artifacts_dir=Path(args.artifacts_dir),
        ingest_artifacts=args.ingest_artifacts,
    ).process_all()
    print("\nИндекс собран. Проверьте и упакуйте его для переноса на сервер:")
    print("    python index_app.py verify")
    print("    python index_app.py pack --out index_bundle.zip")


def cmd_evaluate(args) -> None:
    from src.evaluation import RAGEvaluator, print_report

    path = Path(args.evaluate)
    if not path.exists():
        raise FileNotFoundError(f"Файл оценки не найден: {path}")
    questions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError("Файл оценки должен содержать JSON-массив объектов")
    evaluator = RAGEvaluator()
    out_path = Path(args.output_dir) / "metadata" / "evaluation_results.json"
    results = evaluator.evaluate_batch(questions, save_path=out_path)
    print_report(results)
    print(f"\nРезультаты сохранены: {out_path}")


def cmd_verify(args) -> None:
    from tools.index_transfer import verify

    verify(Path(args.output_dir))


def cmd_export(args) -> None:
    from tools.index_transfer import export_jsonl

    export_jsonl(Path(args.output_dir), Path(args.out), args.source, args.by_type)


def cmd_pack(args) -> None:
    from tools.index_transfer import pack

    pack(Path(args.output_dir), Path(args.out))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"KOIB RAG v{APP_VERSION} — индексация")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("ingest", help="проиндексировать документы")
    pi.add_argument("--docs-dir", default=str(DOCS_DIR))
    pi.add_argument("--output-dir", default=str(OUTPUT_DIR))
    pi.add_argument("--artifacts-dir", default=str(ARTIFACTS_DIR),
                    help="папка со старыми артефактами chunks/docstore/bm25 JSONL")
    pi.add_argument("--ingest-artifacts", action="store_true",
                    help="импортировать очищенные старые артефакты")
    pi.add_argument("--rebuild", action="store_true",
                    help="полностью пересобрать индекс вместо инкрементального")

    pe = sub.add_parser("evaluate", help="оценка качества по JSON-набору вопросов")
    pe.add_argument("evaluate", help="путь к JSON-файлу с вопросами")
    pe.add_argument("--output-dir", default=str(OUTPUT_DIR))
    pe.add_argument("--top-k", type=int, default=FINAL_TOP_K)

    pv = sub.add_parser("verify", help="проверить целостность индекса")
    pv.add_argument("--output-dir", default=str(OUTPUT_DIR))

    px = sub.add_parser("export", help="выгрузить чанки в JSONL")
    px.add_argument("--output-dir", default=str(OUTPUT_DIR))
    px.add_argument("--out", required=True)
    px.add_argument("--source", choices=["docstore", "bm25"], default="docstore")
    px.add_argument("--by-type", action="store_true")

    pp = sub.add_parser("pack", help="собрать переносимый архив индекса")
    pp.add_argument("--output-dir", default=str(OUTPUT_DIR))
    pp.add_argument("--out", default="index_bundle.zip")

    return p


def main() -> None:
    args = build_parser().parse_args()
    ensure_dirs()
    {
        "ingest": cmd_ingest,
        "evaluate": cmd_evaluate,
        "verify": cmd_verify,
        "export": cmd_export,
        "pack": cmd_pack,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
