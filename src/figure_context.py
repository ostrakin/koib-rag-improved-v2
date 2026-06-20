# -*- coding: utf-8 -*-
"""Обогащение figure/formula-чанков текстовым контекстом той же страницы.

Решает «Проблему 2» без мультимодальной модели: для чанков со «слабой» подписью
(1–3 слова, «Изображение N» и т.п.) добавляет заголовок раздела и ближайший
текст той же страницы/источника, чтобы рисунок/формула стали находимыми
семантическим поиском даже без Vision-модели.

Запускать ВСЕГДА на этапе чанкинга (дёшево, без сети). Если включён Vision
(FIGURE_CAPTIONING_ENABLED=true), он отрабатывает ПОСЛЕ и перезаписывает слабые
подписи настоящими описаниями; флаг ``vision_caption`` защищает уже описанные
рисунки от повторной обработки, а ``context_enriched`` помечает текстовое
обогащение.
"""
from __future__ import annotations

import logging
from typing import List

logger = logging.getLogger("koib.figure_context")

try:  # переиспользуем единый критерий «слабой» подписи
    from src.figure_captioning import _is_weak_caption
except Exception:  # pragma: no cover - запасной критерий, если модуль недоступен
    def _is_weak_caption(caption: str) -> bool:
        return not caption or len(caption.strip()) <= 40


# Сколько символов соседнего текста подмешивать в подпись. Достаточно, чтобы
# попали ключевые термины страницы, но не настолько много, чтобы figure-чанк
# превратился в копию текстового.
CONTEXT_MAX_CHARS = 320


def enrich_figures_with_context(chunks: List) -> int:
    """Дописать в слабые figure/formula-чанки контекст соседнего текста.

    Меняет ``content`` (и ``full_content``, если он тоже слабый) у figure/formula
    чанков. Возвращает число обновлённых чанков. Безопасно вызывать всегда:
    если подходящего контекста нет — чанк остаётся прежним.
    """
    # индекс текстовых чанков по (source, page)
    by_page: dict = {}
    for c in chunks:
        if getattr(c, "chunk_type", "") != "text":
            continue
        md = getattr(c, "metadata", {}) or {}
        key = (md.get("source"), md.get("page"))
        by_page.setdefault(key, []).append(c)

    updated = 0
    for c in chunks:
        if getattr(c, "chunk_type", "") not in ("figure", "formula"):
            continue
        md = getattr(c, "metadata", {}) or {}
        if md.get("vision_caption"):  # уже описано Vision-моделью — не трогаем
            continue
        caption = (getattr(c, "content", "") or "").strip()
        if not _is_weak_caption(caption):
            continue

        heading = (md.get("heading") or "").strip()
        neighbours = by_page.get((md.get("source"), md.get("page")), [])
        ctx = ""
        if neighbours:
            ctx = max((n.content for n in neighbours), key=len, default="")[:CONTEXT_MAX_CHARS]

        parts = [p for p in (heading, caption, ctx) if p]
        if not parts:
            continue
        # убираем дубли, сохраняя порядок (heading -> caption -> context)
        new_content = " — ".join(dict.fromkeys(parts))
        c.content = new_content
        full = getattr(c, "full_content", None)
        if full is None or _is_weak_caption(full or ""):
            c.full_content = new_content
        md["context_enriched"] = True
        c.metadata = md
        updated += 1

    if updated:
        logger.info("Контекст страницы добавлен в %s figure/formula-чанков", updated)
    return updated
