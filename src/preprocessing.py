# -*- coding: utf-8 -*-
"""
Предобработка OCR-текста для KOIB RAG.
======================================
Решает «Проблему 2 (Критично)»: мусорные строки, слипшиеся слова и искажения
распознавания, которые ломают лексический (BM25) поиск и зашумляют контекст.

Состав:
  * OCR_REPLACEMENTS  — словарь доменных замен (komb→КОИБ, toto→того и т.п.);
  * fix_homoglyphs    — чинит латиницу, затесавшуюся в кириллические слова
                        (классическая ошибка OCR: 'c','o','a','p','e','x'…);
  * is_garbage_line   — отсекает строки с аномальным соотношением пробелов и
                        букв («ю отл л еге оте кн рь а д ёво л нж а е…»);
  * split_glued_words — аккуратно разрезает слипшиеся слова
                        («количествадопустимых» → «количества допустимых»)
                        динамическим программированием по частотному словарю;
  * preprocess_light  — дешёвая чистка (замены+гомоглифы), пригодна везде;
  * deep_clean        — полная чистка (+отсев мусора +разбиение слов) для
                        этапа индексации на мощной машине.

Модуль использует только стандартную библиотеку. Частотный словарь для
разбиения слов строится лениво из pymorphy3(если установлен) плюс доменный
список терминов КОИБ; при отсутствии pymorphy3 разбиение работает на доменном
словаре и не ломает текст (консервативно: режет только при уверенной сегментации).
"""
from __future__ import annotations

import logging
import re
import threading
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("koib.preprocessing")

_CYR = "\u0400-\u04FF"
_VOWELS = set("аеёиоуыэюяaeiouy")

# ─────────────────────────────────────────────────────────────────────────────
# 1. Доменный словарь замен частых OCR-ошибок
#    (паттерн, замена). Применяется как regex с границами слова, без учёта
#    регистра там, где это безопасно. Порядок важен — сначала длинные формы.
# ─────────────────────────────────────────────────────────────────────────────
OCR_REPLACEMENTS: List[Tuple[re.Pattern, str]] = [
    # Латинизированные/искажённые написания КОИБ и КЭГ
    (re.compile(r"\bkomb\b", re.IGNORECASE), "КОИБ"),
    (re.compile(r"\bko[il1]b\b", re.IGNORECASE), "КОИБ"),
    (re.compile(r"\bкоиб\b", re.IGNORECASE), "КОИБ"),
    (re.compile(r"\bkəг\b", re.IGNORECASE), "КЭГ"),
    # Частые «слова-обрывки», возникавшие при распознавании
    (re.compile(r"\btoto\b", re.IGNORECASE), "того"),
    (re.compile(r"\bустановяич\w*", re.IGNORECASE), "установочный"),
    (re.compile(r"\bсенсорнь[!1|]й\b", re.IGNORECASE), "сенсорный"),
    (re.compile(r"\bбюллетен[ьb]\b", re.IGNORECASE), "бюллетень"),
    # Цифро-буквенные подмены внутри цифровых кодов (l/I→1, O→0) — только
    # в окружении цифр, чтобы не портить слова.
    (re.compile(r"(?<=\d)[lI](?=\d)"), "1"),
    (re.compile(r"(?<=\d)[O](?=\d)"), "0"),
    (re.compile(r"(?<=\d)[З](?=\d)"), "3"),
    # «N°»/«No»/«No.» → № (нормализация номера)
    (re.compile(r"\bN[°o]\.?\b"), "№"),
]

# Латинские буквы → кириллические двойники (для починки гомоглифов в словах)
_LAT2CYR = {
    "a": "а", "c": "с", "e": "е", "o": "о", "p": "р", "x": "х", "y": "у",
    "A": "А", "B": "В", "C": "С", "E": "Е", "H": "Н", "K": "К", "M": "М",
    "O": "О", "P": "Р", "T": "Т", "X": "Х", "Y": "У",
    "k": "к", "m": "м", "t": "т", "n": "п", "h": "н", "b": "ь",
}
# Кириллические → латинские (для починки латинских кодов/аббревиатур)
_CYR2LAT = {
    "А": "A", "В": "B", "С": "C", "Е": "E", "Н": "H", "К": "K", "М": "M",
    "О": "O", "Р": "P", "Т": "T", "Х": "X", "У": "Y",
    "а": "a", "с": "c", "е": "e", "о": "o", "р": "p", "х": "x", "у": "y",
}

_TOKEN_SPLIT_RE = re.compile(r"(\s+|[^\w°№$+\-=±%/.,:;()«»\"'])", re.UNICODE)
_WORD_RE = re.compile(rf"[{_CYR}A-Za-z]+", re.UNICODE)
_LONG_ALPHA_RUN_RE = re.compile(rf"[{_CYR}]{{14,}}", re.UNICODE)


def apply_replacements(text: str) -> str:
    """Применить доменный словарь OCR-замен."""
    if not text:
        return ""
    for pattern, repl in OCR_REPLACEMENTS:
        text = pattern.sub(repl, text)
    return text


def _is_mostly_cyrillic(word: str) -> bool:
    letters = [c for c in word if c.isalpha()]
    if not letters:
        return False
    cyr = sum(1 for c in letters if "\u0400" <= c <= "\u04FF")
    return cyr >= max(1, int(len(letters) * 0.5))


def fix_homoglyphs(text: str) -> str:
    """Починить смешение латиницы и кириллицы внутри одного слова.

    Если слово преимущественно кириллическое, отдельные латинские буквы-двойники
    («c»,«o»,«p»,«e»,«a»,«x»…) заменяются на кириллические. Чисто латинские слова
    (англоязычные термины, коды) не трогаются.
    """
    if not text:
        return ""

    def fix_word(word: str) -> str:
        if not word or word.isascii():
            return word  # чисто латинское/цифровое — не трогаем
        has_cyr = any("\u0400" <= c <= "\u04FF" for c in word)
        has_lat = any(("a" <= c.lower() <= "z") for c in word)
        if not (has_cyr and has_lat):
            return word
        if _is_mostly_cyrillic(word):
            return "".join(_LAT2CYR.get(c, c) for c in word)
        return "".join(_CYR2LAT.get(c, c) for c in word)

    return _WORD_RE.sub(lambda m: fix_word(m.group(0)), text)


def is_garbage_line(line: str) -> bool:
    """True для строк с аномальным соотношением пробелов/одиночных букв.

    Отлавливает OCR-мусор вида «ю отл л еге оте кн рь а д ёво л нж а е»:
    много очень коротких «слов», низкая доля гласных, низкая средняя длина токена.
    Не трогает короткие осмысленные строки и таблицы.
    """
    if not line:
        return False
    s = line.strip()
    if not s or s.startswith("|"):
        return False
    letters = [c for c in s if c.isalpha()]
    if len(letters) < 12:
        return False  # слишком коротко, чтобы судить
    tokens = [t for t in s.split() if any(c.isalpha() for c in t)]
    if len(tokens) < 6:
        return False
    short = sum(1 for t in tokens if len(t) <= 2)
    short_ratio = short / len(tokens)
    avg_len = sum(len(t) for t in tokens) / len(tokens)
    vowels = sum(1 for c in letters if c.lower() in _VOWELS)
    vowel_ratio = vowels / len(letters)
    # Мусор: больше половины «слов» длиной ≤2, ИЛИ крайне короткая средняя длина
    # вместе с низкой долей гласных (нечитаемый набор согласных).
    if short_ratio >= 0.55 and avg_len <= 3.2:
        return True
    if avg_len <= 2.6 and vowel_ratio < 0.30:
        return True
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Частотный словарь и разбиение слипшихся слов
# ─────────────────────────────────────────────────────────────────────────────
_DOMAIN_WORDS = [
    "количество", "количества", "количеств", "допустимых", "допустимый", "бюллетень",
    "бюллетеня", "бюллетеней", "бюллетени", "сканирующее", "устройство", "устройства",
    "накопитель", "накопителя", "печатающего", "комплекс", "обработки", "избирательных",
    "результаты", "голосования", "оператор", "оператора", "комиссии", "участковой",
    "председатель", "сенсорный", "экран", "кнопка", "кнопку", "питание", "питания",
    "режим", "режиме", "тестирование", "тестирования", "контрольный", "соотношение",
    "значение", "значения", "параметр", "параметры", "нештатная", "ситуация", "ситуации",
    "руководство", "эксплуатации", "установка", "установочный", "модель", "модели",
    "сканер", "сканера", "протокол", "протокола", "данные", "данных", "проверка",
    "избирательной", "комиссия", "число", "числа", "голосов", "ошибка", "ошибки",
    "включить", "выключить", "нажмите", "нажать", "вставить", "извлечь", "загрузка",
    "контрольное", "контрольных", "допустимое", "допустимое",
]


@lru_cache(maxsize=1)
def _word_costs() -> Dict[str, float]:
    """Словарь стоимости слов для DP-сегментации (меньше — вероятнее)."""
    words: Dict[str, float] = {}
    # доменные слова — самые «дешёвые»
    for w in _DOMAIN_WORDS:
        words[w] = 1.0
    # расширяем общерусским словарём из pymorphy3, если он доступен
    try:  # pragma: no cover - зависит от окружения
        import pymorphy3  # type: ignore

        morph = pymorphy3.MorphAnalyzer()
        dawg = getattr(morph.dictionary, "words", None)
        if dawg is not None:
            for i, key in enumerate(dawg.iterkeys()):
                w = key.lower().replace("ё", "е")
                if 2 <= len(w) <= 24 and w.isalpha():
                    words.setdefault(w, 3.0)
                if i > 200000:
                    break
    except Exception as exc:
        logger.debug("pymorphy3 недоступен для словаря разбиения: %s", exc)
    return words


def _segment(token: str, costs: Dict[str, float]) -> Optional[List[str]]:
    """DP-сегментация одного длинного токена. None, если уверенно не разбить."""
    n = len(token)
    low = token.lower().replace("ё", "е")
    INF = float("inf")
    best = [INF] * (n + 1)
    back = [0] * (n + 1)
    best[0] = 0.0
    max_word = 24
    for i in range(1, n + 1):
        for j in range(max(0, i - max_word), i):
            seg = low[j:i]
            c = costs.get(seg)
            if c is None:
                continue
            # штраф за слишком короткие сегменты, чтобы не плодить «а»,«и»
            penalty = 0.0 if len(seg) >= 3 else 2.0
            cand = best[j] + c + penalty
            if cand < best[i]:
                best[i] = cand
                back[i] = j
    if best[n] == INF:
        return None
    # восстановить разбиение
    parts: List[str] = []
    i = n
    while i > 0:
        j = back[i]
        parts.append(token[j:i])
        i = j
    parts.reverse()
    # принимаем только если получилось ≥2 осмысленных куска
    if len(parts) < 2 or any(len(p) < 2 for p in parts):
        return None
    return parts


def split_glued_words(text: str, costs: Optional[Dict[str, float]] = None) -> str:
    """Разрезать слипшиеся длинные кириллические токены по словарю.

    Консервативно: трогает только токены длиной ≥14 без пробелов, и только если
    найдено чистое словарное разбиение. Иначе оставляет токен как есть.
    """
    if not text or not _LONG_ALPHA_RUN_RE.search(text):
        return text
    costs = costs or _word_costs()

    def repl(m: re.Match) -> str:
        token = m.group(0)
        if token.lower().replace("ё", "е") in costs:
            return token  # это валидное длинное слово
        seg = _segment(token, costs)
        return " ".join(seg) if seg else token

    return _LONG_ALPHA_RUN_RE.sub(repl, text)


# ─────────────────────────────────────────────────────────────────────────────
# Публичные оркестраторы
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_light(text: str) -> str:
    """Дешёвая предобработка: замены + починка гомоглифов. Безопасна везде."""
    if not text:
        return ""
    text = apply_replacements(text)
    text = fix_homoglyphs(text)
    return text


def deep_clean(text: str, drop_garbage: bool = True, split_words: bool = True) -> str:
    """Полная предобработка для этапа индексации (мощная машина).

    1) доменные замены и починка гомоглифов;
    2) построчный отсев мусорных строк;
    3) разбиение слипшихся слов.
    """
    if not text:
        return ""
    text = preprocess_light(text)
    if drop_garbage:
        kept = [ln for ln in text.split("\n") if not is_garbage_line(ln)]
        text = "\n".join(kept)
    if split_words:
        text = split_glued_words(text)
    return text
