# -*- coding: utf-8 -*-
"""Фильтрация опасного контента и политически/юридически чувствительных запросов."""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import List, Tuple

logger = logging.getLogger("koib.safety")


@dataclass(frozen=True)
class SensitiveTopic:
    category: str
    patterns: List[str]
    description: str


# Технические сбои КОИБ НЕ блокируются: это основной сценарий RAG по документации.
# Блокируются только темы, где бот не должен давать процедурные/юридические решения.
SENSITIVE_TOPICS = [
    SensitiveTopic(
        "invalid_ballot",
        [r"\bнедействительн(?:ый|ые|ых)\s+бюллетен", r"\bаннулировани(?:е|я)\s+бюллетен"],
        "Вопросы о статусе бюллетеней должны решаться по официальной процедуре УИК/ТИК.",
    ),
    SensitiveTopic(
        "complaint",
        [r"\bжалоб(?:а|ы|у)\b", r"\bпожаловаться\b", r"\bпротест\b", r"\bапелляци(?:я|и|ю)\b"],
        "Жалобы и апелляции требуют обращения к ответственным лицам/официальной процедуре.",
    ),
    SensitiveTopic(
        "security",
        [r"\bвзлом\b", r"\bнесанкционированный\s+доступ\b", r"\bутечка\s+данных\b", r"\bфальсификаци"],
        "Вопросы безопасности и фальсификаций должны передаваться ответственным службам.",
    ),
]

FORBIDDEN_ANSWER_PATTERNS = [
    r"<\s*script",
    r"javascript:",
    r"data:text/html",
]


def check_query_safety(query: str) -> Tuple[bool, str]:
    query_lower = (query or "").lower()
    for topic in SENSITIVE_TOPICS:
        for pattern in topic.patterns:
            if re.search(pattern, query_lower, re.IGNORECASE):
                logger.info("Обнаружена чувствительная тема: %s", topic.category)
                return False, topic.description
    return True, ""


def check_answer_safety(answer: str) -> Tuple[bool, str]:
    for pattern in FORBIDDEN_ANSWER_PATTERNS:
        if re.search(pattern, answer or "", re.IGNORECASE):
            return False, f"Forbidden pattern: {pattern}"
    if len(answer or "") > 10000:
        return False, "Answer too long"
    return True, ""


def sanitize_answer(answer: str) -> str:
    answer = re.sub(r"<\s*script[^>]*>.*?</script>", "", answer or "", flags=re.IGNORECASE | re.DOTALL)
    answer = re.sub(r"javascript:", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"data:text/html", "", answer, flags=re.IGNORECASE)
    return answer.strip()
