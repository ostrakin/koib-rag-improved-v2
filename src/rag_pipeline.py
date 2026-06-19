# -*- coding: utf-8 -*-
"""Высокоуровневый RAG-пайплайн: memory -> retrieval -> generation -> validation."""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict

from config import (
    ANSWER_REFINE_ENABLED,
    ANSWER_REFINE_MAX_TOKENS,
    ANSWER_REFINE_TEMPERATURE,
    EMBEDDING_PROVIDER,
    MAX_CONCURRENT_GENERATIONS,
    MAX_FIGURES_PER_ANSWER,
    PROCEDURAL_REMINDER_ENABLED,
    QUERY_PREFIX,
    SEND_FIGURES_ENABLED,
    USE_USHAPED_CONTEXT,
)
from .generation import LLMClient, build_prompt, build_refine_prompt, REFINE_SYSTEM_PROMPT
from .procedures import ensure_procedural_reminder
from .indexing import get_global_embeddings
from .retrieval import HybridRetriever, SemanticCache, reorder_u_shape
from .utils import ConversationMemory, rewrite_query
from .validation import AnswerValidator, get_blocked_response

logger = logging.getLogger("koib.rag_pipeline")
CONTEXT_PRONOUNS = {
    "он", "она", "оно", "они", "его", "её", "их", "нему", "ней", "ними",
    "этом", "этот", "тот", "такой", "там", "это", "неё", "него", "у", "него", "неё",
}


class RAGPipeline:
    def __init__(self):
        self.retriever = HybridRetriever()
        self.llm = LLMClient()
        self.semantic_cache = SemanticCache()
        self.memory = ConversationMemory()
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT_GENERATIONS)

    async def close(self) -> None:
        await self.llm.close()

    async def answer(
        self,
        query: str,
        user_id: str = "anonymous",
        k: int = 4,
        model_filter: str = "",
        use_memory: bool = True,
        validate: bool = True,
        strict_model: bool = None,
    ) -> Dict[str, Any]:
        query = (query or "").strip()
        t0 = time.time()
        if not query:
            return {"answer": "Пустой запрос.", "sources": [], "status": "review", "latency": 0.0}

        async with self._semaphore:
            history = await self.memory.get_history(user_id) if use_memory and user_id != "anonymous" else []
            search_query = query
            if history and any(w.strip(",.!?").lower() in CONTEXT_PRONOUNS for w in query.split()):
                search_query = await rewrite_query(query, history, self.llm)

            query_embedding = None
            try:
                emb = get_global_embeddings()
                embedding_text = (QUERY_PREFIX + search_query) if EMBEDDING_PROVIDER == "local" else search_query
                query_embedding = await asyncio.to_thread(emb.embed_query, embedding_text)
            except Exception as exc:
                logger.warning("Embedding недоступен, будет использован только BM25/FTS: %s", exc)

            if query_embedding:
                cached = self.semantic_cache.get(search_query, query_embedding)
                if cached:
                    answer = cached["answer"]
                    if PROCEDURAL_REMINDER_ENABLED:
                        answer = ensure_procedural_reminder(answer, query)
                    if use_memory and user_id != "anonymous":
                        await self.memory.add_message(user_id, "user", query)
                        await self.memory.add_message(user_id, "assistant", answer[:500])
                    return {
                        "answer": answer,
                        "sources": cached.get("sources", []),
                        "figures": self._collect_figures(cached.get("sources", [])),
                        "status": "approved",
                        "latency": time.time() - t0,
                    }

            results = await asyncio.to_thread(
                self.retriever.search, search_query, k=k,
                model_filter=model_filter, strict_model=strict_model,
            )
            if not results:
                answer = "По вашему запросу не найдено релевантных фрагментов в официальной документации."
                if PROCEDURAL_REMINDER_ENABLED:
                    answer = ensure_procedural_reminder(answer, query)
                if use_memory and user_id != "anonymous":
                    await self.memory.add_message(user_id, "user", query)
                    await self.memory.add_message(user_id, "assistant", answer[:500])
                return {"answer": answer, "sources": [], "status": "review", "latency": time.time() - t0}

            if USE_USHAPED_CONTEXT and len(results) > 2:
                results = reorder_u_shape(results)

            prompt = build_prompt(search_query, results)
            answer = await self.llm.generate_async(prompt)

            # Второй проход: переписать черновик понятным языком (вопрос + ответ).
            if ANSWER_REFINE_ENABLED and answer and not answer.startswith("Ошибка"):
                try:
                    refined = await self.llm.generate_async(
                        build_refine_prompt(query, answer),
                        system_prompt=REFINE_SYSTEM_PROMPT,
                        max_tokens=ANSWER_REFINE_MAX_TOKENS,
                        temperature=ANSWER_REFINE_TEMPERATURE,
                    )
                    refined = (refined or "").strip()
                    # Защита: refine не должен терять факты — принимаем результат,
                    # только если он осмысленный и сохранил регламентный блок (если был).
                    reminder_kept = ("Регламентное уведомление" not in answer) or ("Регламентное уведомление" in refined)
                    if refined and not refined.startswith("Ошибка") and len(refined) >= 40 and reminder_kept:
                        answer = refined
                except Exception as exc:
                    logger.warning("Answer refine error (используем черновик): %s", exc)

            status = "approved"
            if answer.startswith("Ошибка"):
                status = "review"
            elif validate:
                try:
                    validation_result = AnswerValidator().validate(answer, results, query)
                    if validation_result.status == "rejected":
                        status = "rejected"
                        answer = get_blocked_response()
                    elif validation_result.status == "review":
                        status = "review"
                except Exception as exc:
                    logger.warning("Validation error: %s", exc)

            if PROCEDURAL_REMINDER_ENABLED:
                answer = ensure_procedural_reminder(answer, query)

            if use_memory and user_id != "anonymous":
                await self.memory.add_message(user_id, "user", query)
                await self.memory.add_message(user_id, "assistant", answer[:500])

            sources = []
            for r in results:
                src = {
                    "document": r.source,
                    "page": r.page,
                    "heading": r.heading,
                    "chunk_type": r.chunk_type,
                    "score": r.score,
                }
                # Для рисунков сохраняем путь к изображению и подпись — бот сможет
                # приложить картинку к ответу. Хранится и в семантическом кэше.
                if r.chunk_type == "figure" and isinstance(r.metadata, dict):
                    image_path = str(r.metadata.get("image_path") or "")
                    if image_path:
                        src["image_path"] = image_path
                        src["caption"] = (r.content or "")[:300]
                sources.append(src)

            figures = self._collect_figures(sources)
            if status == "approved" and query_embedding:
                self.semantic_cache.set(search_query, query_embedding, answer, sources)

            return {"answer": answer, "sources": sources, "status": status,
                    "figures": figures, "latency": time.time() - t0,
                    "model_filter": model_filter}

    @staticmethod
    def _collect_figures(sources) -> list:
        """Собрать из sources рисунки с существующими файлами (для отправки пользователю).

        image_path хранится относительно OUTPUT_DIR («figures/x.png») и разрешается
        через config.resolve_figure_path — это работает и на сервере, и на
        индексаторе, и совместимо со старыми абсолютными путями.
        """
        if not SEND_FIGURES_ENABLED:
            return []
        from config import resolve_figure_path

        figures = []
        seen = set()
        for src in sources or []:
            if not isinstance(src, dict):
                continue
            path = resolve_figure_path(src.get("image_path") or "")
            if not path or path in seen:
                continue
            seen.add(path)
            figures.append({
                "image_path": path,
                "caption": str(src.get("caption") or ""),
                "document": str(src.get("document") or ""),
                "page": src.get("page", 0),
            })
            if len(figures) >= MAX_FIGURES_PER_ANSWER:
                break
        return figures
