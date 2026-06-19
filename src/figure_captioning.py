# -*- coding: utf-8 -*-
"""
Генерация текстовых описаний (caption) для рисунков/схем КОИБ.
==============================================================
Решает «Проблему 7 (Средний приоритет)»: раньше для figure-чанков сохранялись
только image_path и размеры, поэтому без мультимодальной LLM система не могла
ответить на вопросы вида «Как правильно опечатать накопитель?».

Здесь на ЭТАПЕ ИНДЕКСАЦИИ (мощная машина) каждое изображение прогоняется через
лёгкую Vision-модель, а сгенерированное описание кладётся в поле ``content``
чанка. На боевом сервере с ограниченными ресурсами Vision уже не нужен —
описания «запечены» в индекс.

Провайдеры (config.FIGURE_CAPTION_PROVIDER):
  * "openai" — gpt-4o-mini (vision) по OPENAI_API_KEY;
  * "local"  — Ollama-совместимый сервер (например, qwen2-vl / llava) по
               LOCAL_LLM_URL /api/generate с полем images;
  * "none"   — отключено (по умолчанию), возвращается пустая строка.

Модуль устойчив к ошибкам: при любой проблеме возвращается пустая строка, и
исходный текст figure-чанка остаётся прежним.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger("koib.captioning")

CAPTION_PROMPT = (
    "Ты технический иллюстратор документации КОИБ (комплекс обработки "
    "избирательных бюллетеней). Кратко (1–3 предложения, по-русски) опиши, что "
    "изображено на рисунке: какой узел/экран/действие показаны, какие подписи, "
    "кнопки или индикаторы видны. Не выдумывай того, чего не видно. Без вступлений."
)

# Порог «слабой» подписи: по такой подписи семантический поиск не сработает, и
# её выгоднее заменить настоящим Vision-описанием (Проблема 4). Раньше слабыми
# считались только пустые подписи и «Изображение N», но в индексе полно обрывков
# фраз в 1–2 слова: «Внешний вид», «»). Для», «Сообщение о».
WEAK_CAPTION_MAX_CHARS = 40
WEAK_CAPTION_FRAGMENTS = (
    ").", "). для", "сообщение о", "внешний вид", "общий вид", "рис.", "см.",
)


def _is_weak_caption(caption: str) -> bool:
    """True, если подпись слишком скудна для поиска и её стоит заменить Vision-описанием."""
    if not caption or not caption.strip():
        return True
    stripped = caption.strip()
    if stripped.lower().startswith("изображение"):
        return True
    text = stripped.lower()
    # Развёрнутые подписи (длиннее порога) всегда считаются содержательными —
    # даже если они начинаются со слов «общий вид» и т.п.
    if len(stripped) > WEAK_CAPTION_MAX_CHARS:
        return False
    # короткий обрывок фразы
    if caption.count(" ") <= 3:
        return True
    # типичные обрывки из вёрстки PDF для коротких подписей
    if any(text.startswith(frag) for frag in WEAK_CAPTION_FRAGMENTS):
        return True
    # начинается со служебных символов (скобка, точка) — артефакт разбиения
    if stripped[:1] in {")", "(", ".", ",", ";"}:
        return True
    return False


def _read_image_b64(image_path: Path) -> Optional[str]:
    try:
        data = Path(image_path).read_bytes()
        return base64.b64encode(data).decode("ascii")
    except Exception as exc:
        logger.debug("Не удалось прочитать изображение %s: %s", image_path, exc)
        return None


def _caption_openai(image_path: Path, model: str, max_tokens: int) -> str:
    from config import OPENAI_API_KEY

    if not OPENAI_API_KEY:
        return ""
    b64 = _read_image_b64(image_path)
    if not b64:
        return ""
    try:
        from openai import OpenAI

        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": CAPTION_PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                    ],
                }
            ],
            max_tokens=max_tokens,
            temperature=0.1,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.debug("OpenAI vision caption error: %s", exc)
        return ""


def _caption_local(image_path: Path, model: str, max_tokens: int) -> str:
    import requests  # локальный провайдер опционален

    from config import LOCAL_LLM_URL

    b64 = _read_image_b64(image_path)
    if not b64:
        return ""
    try:
        resp = requests.post(
            f"{LOCAL_LLM_URL}/api/generate",
            json={
                "model": model,
                "prompt": CAPTION_PROMPT,
                "images": [b64],
                "stream": False,
                "options": {"num_predict": max_tokens, "temperature": 0.1},
            },
            timeout=120,
        )
        if resp.status_code != 200:
            logger.debug("Local vision caption HTTP %s", resp.status_code)
            return ""
        return (resp.json().get("response") or "").strip()
    except Exception as exc:
        logger.debug("Local vision caption error: %s", exc)
        return ""


def caption_image(image_path: Path) -> str:
    """Сгенерировать описание изображения выбранным провайдером."""
    from config import (
        FIGURE_CAPTION_MAX_TOKENS,
        FIGURE_CAPTION_MODEL,
        FIGURE_CAPTION_PROVIDER,
        FIGURE_CAPTIONING_ENABLED,
    )

    if not FIGURE_CAPTIONING_ENABLED:
        return ""
    image_path = Path(image_path)
    if not image_path.exists():
        return ""
    provider = (FIGURE_CAPTION_PROVIDER or "none").lower().strip()
    if provider == "openai":
        return _caption_openai(image_path, FIGURE_CAPTION_MODEL, FIGURE_CAPTION_MAX_TOKENS)
    if provider == "local":
        return _caption_local(image_path, FIGURE_CAPTION_MODEL, FIGURE_CAPTION_MAX_TOKENS)
    return ""


def caption_figures_in_chunks(chunks: List) -> int:
    """Пост-обработка: дописать описания во все figure-чанки.

    Меняет content (и full_content) у figure-чанков со слабым описанием
    («Изображение N» или пустым). Возвращает число обновлённых чанков.
    Безопасно вызывать всегда: если captioning выключен — ничего не делает.
    """
    from config import FIGURE_CAPTIONING_ENABLED, FIGURE_CAPTION_MAX_IMAGES

    if not FIGURE_CAPTIONING_ENABLED:
        return 0

    updated = 0
    processed = 0
    for chunk in chunks:
        if getattr(chunk, "chunk_type", "") != "figure":
            continue
        meta = getattr(chunk, "metadata", {}) or {}
        stored_path = meta.get("image_path")
        # Путь к рисунку хранится относительно OUTPUT_DIR; разрешаем его в
        # абсолютный на текущей машине. Старые индексы с абсолютными путями тоже
        # разрезолвятся — см. config.resolve_figure_path.
        from config import resolve_figure_path
        image_path = resolve_figure_path(stored_path)
        if not image_path:
            continue
        current = (getattr(chunk, "content", "") or "").strip()
        # Слабая подпись: пустая, «Изображение N» или обрывок фразы, по которому
        # семантический поиск всё равно не сработает (Проблема 4). Такие заменяем
        # настоящим Vision-описанием; развёрнутые подписи оставляем как есть.
        if not _is_weak_caption(current):
            continue
        if FIGURE_CAPTION_MAX_IMAGES and processed >= FIGURE_CAPTION_MAX_IMAGES:
            break
        processed += 1
        caption = caption_image(Path(image_path))
        if not caption:
            continue
        prefix = (current + ". ") if current and not current.lower().startswith("изображение") else ""
        new_content = (prefix + caption).strip()
        chunk.content = new_content
        chunk.full_content = new_content
        meta["vision_caption"] = True
        chunk.metadata = meta
        updated += 1
    if updated:
        logger.info("Vision: сгенерировано описаний для %s рисунков", updated)
    return updated
