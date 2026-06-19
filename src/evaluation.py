# -*- coding: utf-8 -*-
"""
Koib-V-4.6 — Модуль оценки качества RAG
★ ИСПРАВЛЕНО: context содержит реальный контент чанков, а не только имена файлов
"""
import json
import re
import logging
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field, asdict
from config import LLM_PROVIDER

logger = logging.getLogger("koib.evaluation")


@dataclass
class EvalResult:
    question_id: str
    question: str
    category: str = ""
    koib_model: str = ""
    answer: str = ""
    reference_answer: str = ""
    context_chunks: int = 0
    faithfulness: float = 0.0
    answer_relevancy: float = 0.0
    context_precision: float = 0.0
    context_recall: float = 0.0
    token_f1: float = 0.0
    has_reference: bool = False
    error: Optional[str] = None
    latency_sec: float = 0.0

    @property
    def rag_score(self) -> float:
        vals = [self.faithfulness, self.answer_relevancy,
                self.context_precision, self.context_recall]
        vals = [v for v in vals if v > 0]
        return round(sum(vals) / len(vals), 3) if vals else 0.0


PROMPT_FAITHFULNESS = """Ты — строгий судья качества AI-ответов. Оцени ВЕРНОСТЬ ответа относительно контекста.

ВОПРОС: {question}

КОНТЕКСТ (извлечённые фрагменты документации):
{context}

ОТВЕТ СИСТЕМЫ:
{answer}

Критерий — Faithfulness (Верность):
Содержит ли ответ ТОЛЬКО информацию из контекста? Нет ли в нём домыслов?
Оцени по шкале от 0 до 10:
10 — ответ полностью основан на контексте
5 — частично из контекста, частично домыслы
0 — ответ полностью придуман

Ответь ТОЛЬКО одним числом от 0 до 10."""

PROMPT_ANSWER_RELEVANCY = """Ты — строгий судья качества AI-ответов. Оцени РЕЛЕВАНТНОСТЬ ответа вопросу.

ВОПРОС: {question}

ОТВЕТ СИСТЕМЫ:
{answer}

Критерий — Answer Relevancy (Релевантность):
Отвечает ли ответ напрямую на поставленный вопрос?
Оцени по шкале от 0 до 10:
10 — ответ точно и полно отвечает на вопрос
5 — частично отвечает, много лишнего
0 — ответ не по теме

Ответь ТОЛЬКО одним числом от 0 до 10."""

PROMPT_CONTEXT_PRECISION = """Ты — строгий судья качества AI-ответов. Оцени ТОЧНОСТЬ найденного контекста.

ВОПРОС: {question}

НАЙДЕННЫЕ ФРАГМЕНТЫ ДОКУМЕНТАЦИИ:
{context}

Критерий — Context Precision (Точность контекста):
Какая доля фрагментов действительно нужна для ответа на вопрос?
Оцени по шкале от 0 до 10:
10 — все фрагменты релевантны
5 — примерно половина по теме
0 — все нерелевантны

Ответь ТОЛЬКО одним числом от 0 до 10."""

PROMPT_CONTEXT_RECALL = """Ты — строгий судья качества AI-ответов. Оцени ПОЛНОТУ найденного контекста.

ВОПРОС: {question}

ЭТАЛОННЫЙ ОТВЕТ: {reference}

НАЙДЕННЫЕ ФРАГМЕНТЫ ДОКУМЕНТАЦИИ:
{context}

Критерий — Context Recall (Полнота контекста):
Содержит ли найденный контекст достаточно информации для полного ответа?
Оцени по шкале от 0 до 10:
10 — контекст содержит всё необходимое
5 — контекст содержит часть нужной информации
0 — контекст совсем не помогает

Ответь ТОЛЬКО одним числом от 0 до 10."""


def _normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return text


def token_f1(prediction: str, reference: str) -> float:
    pred_tokens = set(_normalize_text(prediction).split())
    ref_tokens = set(_normalize_text(reference).split())
    if not ref_tokens:
        return 0.0
    common = pred_tokens & ref_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(ref_tokens)
    return round(2 * precision * recall / (precision + recall), 3)


def _extract_score(text: str) -> float:
    nums = re.findall(r"\b(\d+(?:\.\d+)?)\b", text)
    for n in nums:
        val = float(n)
        if 0 <= val <= 10:
            return round(val / 10.0, 3)
    return 0.0


class LLMJudge:
    def __init__(self, provider: Optional[str] = None):
        from .generation import LLMClient
        self.llm = LLMClient(provider=provider or LLM_PROVIDER)

    def score(self, prompt: str) -> float:
        try:
            response = self.llm.generate(prompt, max_tokens=50)
            return _extract_score(response)
        except Exception as exc:
            logger.warning(f"Ошибка LLM-судьи: {exc}")
            return 0.0


class RAGEvaluator:
    def __init__(self, judge_provider: Optional[str] = None):
        self.judge = LLMJudge(provider=judge_provider)

    def evaluate_one(
        self, question: str, answer: str, context: str,
        reference: str = "", question_id: str = "",
        category: str = "", koib_model: str = "",
    ) -> EvalResult:
        result = EvalResult(
            question_id=question_id,
            question=question,
            category=category,
            koib_model=koib_model,
            answer=answer,
            reference_answer=reference,
            has_reference=bool(reference),
        )
        logger.info(f"Оценка [{question_id}]: {question[:80]}...")

        result.faithfulness = self.judge.score(
            PROMPT_FAITHFULNESS.format(question=question, context=context, answer=answer)
        )
        result.answer_relevancy = self.judge.score(
            PROMPT_ANSWER_RELEVANCY.format(question=question, answer=answer)
        )
        result.context_precision = self.judge.score(
            PROMPT_CONTEXT_PRECISION.format(question=question, context=context)
        )
        if reference:
            result.context_recall = self.judge.score(
                PROMPT_CONTEXT_RECALL.format(
                    question=question, reference=reference, context=context
                )
            )
            result.token_f1 = token_f1(answer, reference)

        logger.info(
            f"  RAG={result.rag_score:.3f} F={result.faithfulness:.2f} "
            f"AR={result.answer_relevancy:.2f} CP={result.context_precision:.2f} "
            f"CR={result.context_recall:.2f}"
        )
        return result

    def evaluate_batch(
        self,
        questions: List[Dict[str, Any]],
        save_path: Optional[Path] = None,
    ) -> List[EvalResult]:
        results: List[EvalResult] = []
        for i, q in enumerate(questions):
            try:
                result = self.evaluate_one(
                    question=q.get("question", ""),
                    answer=q.get("answer", ""),
                    # ★ ИСПРАВЛЕНО: контекст из реальных чанков
                    context=q.get("context", q.get("context_text", "")),
                    reference=q.get("reference", q.get("reference_answer", "")),
                    question_id=q.get("id", q.get("question_id", str(i))),
                    category=q.get("category", ""),
                    koib_model=q.get("koib_model", ""),
                )
                results.append(result)
            except Exception as exc:
                logger.error(f"Ошибка оценки вопроса {i}: {exc}")
                results.append(EvalResult(
                    question_id=str(i),
                    question=q.get("question", ""),
                    error=str(exc),
                ))
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump([asdict(r) for r in results], f,
                          ensure_ascii=False, indent=2)
            logger.info(f"Результаты сохранены: {save_path}")
        return results


def print_report(results: List[EvalResult]) -> None:
    ok = [r for r in results if r.error is None]
    if not ok:
        print("\nНет успешных результатов.")
        return

    def avg(attr):
        vals = [getattr(r, attr) for r in ok]
        return round(sum(vals) / len(vals), 3) if vals else 0

    print("\n" + "═" * 65)
    print("  ИТОГОВЫЙ ОТЧЁТ КАЧЕСТВА RAG-СИСТЕМЫ")
    print("═" * 65)
    print(f"  Вопросов обработано : {len(ok)}/{len(results)}")
    print(f"  Среднее время ответа: {avg('latency_sec')} сек")
    print()
    print(f"  Faithfulness       : {avg('faithfulness'):.3f}")
    print(f"  Answer Relevancy   : {avg('answer_relevancy'):.3f}")
    print(f"  Context Precision  : {avg('context_precision'):.3f}")
    print(f"  Context Recall     : {avg('context_recall'):.3f}")
    total_rag = round(sum(r.rag_score for r in ok) / len(ok), 3)
    print(f"  {'─' * 40}")
    print(f"  Итоговый RAG Score : {total_rag:.3f}")
    print("═" * 65)
