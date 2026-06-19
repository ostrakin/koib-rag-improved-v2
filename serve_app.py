# -*- coding: utf-8 -*-
"""KOIB RAG — серверное приложение (обслуживание запросов).

Умеет только запускать API-сервер (``serve``) и задавать одиночный
проверочный вопрос (``query``). Индексации здесь НЕТ: готовый ``output/``
переносится с машины-индексатора (см. ``tools/index_transfer.py``).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT))

from config import API_HOST, API_PORT, APP_VERSION, FINAL_TOP_K, HF_OFFLINE_MODE, ensure_dirs

if HF_OFFLINE_MODE:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("koib.serve")


def _warn_if_index_incompatible() -> None:
    """Необязательная, но полезная проверка манифеста при старте через CLI."""
    try:
        from tools.index_transfer import MANIFEST_REL, _check_compat
        from config import OUTPUT_DIR
        import json

        mpath = Path(OUTPUT_DIR) / MANIFEST_REL
        if mpath.exists():
            _check_compat(json.loads(mpath.read_text(encoding="utf-8")))
    except Exception:
        pass


def cmd_query(args) -> None:
    from src.rag_pipeline import RAGPipeline

    async def run() -> None:
        pipeline = RAGPipeline()
        t0 = time.time()
        try:
            result = await pipeline.answer(
                query=args.query,
                user_id="cli",
                k=args.top_k,
                model_filter=args.model_filter,
                use_memory=False,
                validate=True,
            )
            print(f"\nОТВЕТ:\n{result['answer']}")
            if result.get("sources"):
                print("\nИсточники:")
                seen = set()
                for s in result["sources"]:
                    key = f"{s.get('document')}_{s.get('page')}"
                    if key not in seen:
                        seen.add(key)
                        print(f"  - {s.get('document')}, стр. {s.get('page')}")
            print(f"\nСтатус: {result.get('status')} | Время: {time.time() - t0:.2f}с")
        finally:
            await pipeline.close()

    asyncio.run(run())


def cmd_serve(args) -> None:
    import uvicorn

    # Один воркер: модель эмбеддингов и FAISS загружаются один раз на процесс.
    # Несколько воркеров на машине с 2 ГБ ОЗУ приведут к OOM.
    uvicorn.run("api.app:app", host=args.host, port=args.port, log_level="info", workers=1)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=f"KOIB RAG v{APP_VERSION} — сервер")
    p.add_argument("--serve", action="store_true", help="запустить FastAPI-сервер")
    p.add_argument("--query", type=str, default="", help="одиночный проверочный вопрос из CLI")
    p.add_argument("--top-k", type=int, default=FINAL_TOP_K)
    p.add_argument("--model-filter", type=str, default="")
    p.add_argument("--host", type=str, default=API_HOST)
    p.add_argument("--port", type=int, default=API_PORT)
    return p


def main() -> None:
    args = build_parser().parse_args()
    ensure_dirs()
    _warn_if_index_incompatible()
    if args.query:
        cmd_query(args)
    elif args.serve:
        cmd_serve(args)
    else:
        build_parser().print_help()


if __name__ == "__main__":
    main()
