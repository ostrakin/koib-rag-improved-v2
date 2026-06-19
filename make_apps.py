# -*- coding: utf-8 -*-
"""make_apps.py — собрать два разворачиваемых приложения из одного исходника.

В репозитории код хранится в ЕДИНСТВЕННОМ экземпляре (никаких дублей функций).
Этот скрипт раскладывает его на две самодостаточные сборки:

    dist/koib-indexer/   — индексация на мощном компьютере (полный набор)
    dist/koib-server/    — обслуживание запросов на сервере (узкий набор, без OCR/парсинга)

Общие модули попадают в обе сборки, но СОПРОВОЖДАЮТСЯ из одного места (src/),
поэтому дублирующегося исходного кода нет — есть один источник истины.

    python make_apps.py            # собрать обе в ./dist
    python make_apps.py --zip      # дополнительно упаковать в .zip
"""
from __future__ import annotations

import argparse
import shutil
import zipfile
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
DIST = ROOT / "dist"

# Узкий набор src-модулей, нужный СЕРВЕРУ (транзитивное замыкание api/app.py).
SERVER_SRC = [
    "__init__.py", "generation.py", "indexing.py", "preprocessing.py",
    "procedures.py", "quarantine.py", "rag_pipeline.py", "retrieval.py",
    "safety.py", "text_processing.py", "utils.py", "validation.py", "vk_bot.py",
]

COMMON_TOOLS = ["__init__.py", "index_transfer.py"]


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        shutil.copytree(src, dst, dirs_exist_ok=True)
    else:
        shutil.copy2(src, dst)


def build_server(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    # точка входа, конфиг, окружение
    for f in ["serve_app.py", "config.py", ".env.server.example",
              "requirements-server.txt", "requirements-torch-cpu.txt"]:
        _copy(ROOT / f, out / f)
    # api целиком
    _copy(ROOT / "api", out / "api")
    # только нужные src-модули
    (out / "src").mkdir()
    for m in SERVER_SRC:
        _copy(ROOT / "src" / m, out / "src" / m)
    # инструмент проверки/распаковки индекса
    (out / "tools").mkdir()
    for t in COMMON_TOOLS:
        _copy(ROOT / "tools" / t, out / "tools" / t)
    # docs + место под индекс
    _copy(ROOT / "docs" / "VK_BOT_SETUP.md", out / "docs" / "VK_BOT_SETUP.md")
    _placeholder(out / "output")
    _copy(ROOT / "deploy", out)  # Dockerfile/koib.service и т.п., если есть
    (out / "README.md").write_text(_SERVER_README, encoding="utf-8")


def build_indexer(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for f in ["index_app.py", "serve_app.py", "config.py", ".env.indexer.example",
              "requirements-indexer.txt", "batch_ingest.py", "build_ideal_index.py"]:
        _copy(ROOT / f, out / f)
    _copy(ROOT / "api", out / "api")          # чтобы можно было локально тестировать serve
    _copy(ROOT / "src", out / "src")          # полный src
    (out / "tools").mkdir()
    for t in COMMON_TOOLS:
        _copy(ROOT / "tools" / t, out / "tools" / t)
    _copy(ROOT / "docs", out / "docs")
    _placeholder(out / "data" / "docs")
    _placeholder(out / "output")
    (out / "README.md").write_text(_INDEXER_README, encoding="utf-8")


def _placeholder(d: Path) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / ".gitkeep").write_text("", encoding="utf-8")


def _zip(folder: Path) -> Path:
    archive = folder.with_suffix(".zip")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        for p in folder.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(folder.parent))
    return archive


_SERVER_README = """# koib-server — серверная сборка

Только обслуживание запросов. Индексации нет.

1) `cp .env.server.example .env` и заполнить секреты (GigaChat и VK).
2) Установить зависимости:
       pip install -r requirements-torch-cpu.txt
       pip install -r requirements-server.txt
3) Положить индекс: перенесите `index_bundle.zip` с индексатора и распакуйте:
       python -m tools.index_transfer unpack --archive index_bundle.zip --output-dir ./output
4) Запуск:
       python serve_app.py --serve
   Проверка готовности: `GET /ready`.
"""

_INDEXER_README = """# koib-indexer — приложение индексации (мощный компьютер)

1) `cp .env.indexer.example .env` и при необходимости поправить пути/модель.
2) `pip install -r requirements-indexer.txt`  (нужен системный tesseract-ocr +rus).
3) Положить документы в `data/docs/` и собрать индекс:
       python index_app.py ingest --rebuild
4) Проверить и упаковать для сервера:
       python index_app.py verify
       python index_app.py pack --out index_bundle.zip
5) Перенести `index_bundle.zip` на сервер (scp/rsync).
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Сборка приложений koib-indexer и koib-server")
    ap.add_argument("--zip", action="store_true", help="дополнительно упаковать сборки в .zip")
    args = ap.parse_args()

    DIST.mkdir(exist_ok=True)
    build_indexer(DIST / "koib-indexer")
    build_server(DIST / "koib-server")
    print(f"✓ Собрано: {DIST / 'koib-indexer'}")
    print(f"✓ Собрано: {DIST / 'koib-server'}")
    if args.zip:
        print("✓ Архив:", _zip(DIST / "koib-indexer"))
        print("✓ Архив:", _zip(DIST / "koib-server"))


if __name__ == "__main__":
    main()
