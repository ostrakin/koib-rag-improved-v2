# -*- coding: utf-8 -*-
import re, json, logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from config import UNCERTAINTY_MIN_LENGTH, VALIDATION_USE_LLM_JUDGE, VALIDATION_CHECK_CITATIONS

logger = logging.getLogger("koib.validation")
CITATION_PATTERN = re.compile(r"\[Документ:\s*([^,\]]+?),\s*стр(?:аница)?\.?\s*(\d+)\]", re.IGNORECASE)
PROMPT_FACTUALITY_CHECK = """Ты — строгий валидатор RAG. Проверь, основан ли ОТВЕТ исключительно на КОНТЕКСТЕ.
<retrieved_context>{context}</retrieved_context>
<generated_answer>{answer}</generated_answer>
Критерии: 1. Галлюцинации? 2. Противоречия? 3. Слова неуверенности как свои (не цитаты)?
Ответь СТРОГО JSON: {{"is_factual": true/false, "issues": ["список"]}}"""

@dataclass
class ValidationCheck:
    name: str; passed: bool; details: str = ""; severity: str = "info"; semantic_similarity: float = 0.0

@dataclass
class ValidationResult:
    status: str = "approved"; checks: List[ValidationCheck] = field(default_factory=list)
    requires_review_reasons: List[str] = field(default_factory=list)
    def add_check(self, check: ValidationCheck):
        self.checks.append(check)
        if not check.passed:
            if check.severity == "critical": self.status = "rejected"
            elif check.severity == "warning" and self.status != "rejected": self.status = "review"; self.requires_review_reasons.append(check.details)
    def to_dict(self): return {"status": self.status, "checks": [{"name": c.name, "passed": c.passed, "details": c.details, "severity": c.severity} for c in self.checks], "requires_review_reasons": self.requires_review_reasons}

class AnswerValidator:
    def __init__(self, embeddings=None, similarity_threshold: float = 0.75):
        self.embeddings = embeddings; self.similarity_threshold = similarity_threshold; self._llm_judge = None
    def _get_llm_judge(self):
        if not self._llm_judge:
            try: from .generation import LLMClient; self._llm_judge = LLMClient()
            except Exception: pass
        return self._llm_judge

    def validate(self, answer: str, context_chunks: List[Any], query: str = "") -> ValidationResult:
        result = ValidationResult()
        if len(answer.strip()) < UNCERTAINTY_MIN_LENGTH: return result
        if VALIDATION_USE_LLM_JUDGE:
            check = self._check_factuality_llm(answer, context_chunks)
            result.add_check(check)
            if not check.passed: return result
        result.add_check(self._check_sources(answer))
        if VALIDATION_CHECK_CITATIONS and context_chunks: result.add_check(self._check_citations_authenticity(answer, context_chunks))
        return result

    def _check_factuality_llm(self, answer, chunks) -> ValidationCheck:
        judge = self._get_llm_judge()
        if not judge: return ValidationCheck("factuality", True, "Судья недоступен", "info")
        ctx = "\n".join(c.to_context_string() if hasattr(c, 'to_context_string') else str(c) for c in chunks[:5])
        try:
            resp = judge.generate(PROMPT_FACTUALITY_CHECK.format(context=ctx[:6000], answer=answer[:3000]), max_tokens=300, temperature=0.01)
            m = re.search(r'\{[^{}]*"is_factual"[^{}]*\}', resp, re.DOTALL)
            if m:
                v = json.loads(m.group(0))
                if not v.get("is_factual", True): return ValidationCheck("factuality", False, "Галлюцинации", "critical")
            return ValidationCheck("factuality", True, "ОК", "info")
        except Exception: return ValidationCheck("factuality", True, "Ошибка парсинга", "info")

    def _check_sources(self, answer) -> ValidationCheck:
        c = CITATION_PATTERN.findall(answer)
        return ValidationCheck("sources", bool(c), f"Найдено {len(c)}", "info" if c else "warning")

    def _check_citations_authenticity(self, answer, chunks) -> ValidationCheck:
        real = {(c.source.split('/')[-1].lower(), c.page) for c in chunks if hasattr(c, 'source') and c.source}
        fake = []
        for doc, page in CITATION_PATTERN.findall(answer):
            try: p = int(page)
            except ValueError: continue
            if not any(p == rp and (doc.lower() in rs or rs in doc.lower()) for rs, rp in real): fake.append(doc)
        return ValidationCheck("citations", not fake, f"Фейковые: {fake}", "info" if not fake else "warning")

def get_blocked_response() -> str:
    return "По вашему запросу не найдено точного ответа в официальных источниках."
