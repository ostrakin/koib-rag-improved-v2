# -*- coding: utf-8 -*-
"""Production-ready VK Callback бот для KOIB RAG.

Модуль не зависит от FastAPI напрямую: route только принимает webhook и
передаёт событие сюда. Это упрощает тестирование и не смешивает HTTP-слой,
VK API, rate limiting и RAG-логику.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import secrets
import sqlite3
import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Tuple

import aiohttp

from config import (
    APP_VERSION,
    FINAL_TOP_K,
    METADATA_DIR,
    MODEL_FILTER_STRICT,
    PROCEDURAL_REMINDER_ENABLED,
    SEND_FIGURES_ENABLED,
    VK_ACCESS_TOKEN,
    VK_ADMIN_IDS,
    VK_API_VERSION,
    VK_BOT_MENTION_ALIASES,
    VK_CONFIRM_CODE,
    VK_DEDUP_TTL_SECONDS,
    VK_ERROR_MESSAGE,
    VK_GLOBAL_RATE_LIMIT_PER_MINUTE,
    VK_GROUP_ID,
    VK_MAX_INCOMING_CHARS,
    VK_MAX_OUTGOING_CHARS,
    VK_OUTBOUND_TIMEOUT,
    VK_RATE_LIMIT_PER_MINUTE,
    VK_REPLY_IN_GROUP_CHATS,
    VK_SECRET_KEY,
    VK_SEND_TYPING,
)
from src.procedures import detect_incident, ensure_procedural_reminder
from src.safety import check_answer_safety, check_query_safety, sanitize_answer
from src.utils import KNOWN_MODELS, detect_model_in_text, model_label
from src.validation import get_blocked_response

logger = logging.getLogger("koib.vk_bot")

VK_API_URL = "https://api.vk.com/method/{method}"
_GROUP_CHAT_PEER_ID = 2_000_000_000

# Порог уверенности, при котором модель из ТЕКСТА сообщения считается явной.
# detect_model_in_text даёт 0.95 для сильных сигналов и 0.3+ для явных
# упоминаний «КОИБ-2010 / 2017а». Этого достаточно, чтобы привязать запрос.
_MODEL_TEXT_MIN_CONFIDENCE = 0.3
# Спецзначение префы «любая модель» (искать без жёсткого фильтра).
_MODEL_ANY = "any"


@dataclass(frozen=True)
class VKIncomingMessage:
    """Нормализованное входящее сообщение VK."""

    event_key: str
    user_id: int
    peer_id: int
    text: str
    message_id: int = 0
    conversation_message_id: int = 0
    date: int = 0
    payload: str = ""

    @property
    def is_group_chat(self) -> bool:
        return self.peer_id >= _GROUP_CHAT_PEER_ID


class SlidingWindowRateLimiter:
    """Простой in-memory sliding-window limiter без внешних зависимостей."""

    def __init__(self, limit: int, window_seconds: int = 60):
        self.limit = max(1, int(limit))
        self.window_seconds = max(1, int(window_seconds))
        self._buckets: Dict[str, Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, key: str) -> bool:
        now = time.time()
        with self._lock:
            bucket = self._buckets[key]
            while bucket and now - bucket[0] > self.window_seconds:
                bucket.popleft()
            if len(bucket) >= self.limit:
                return False
            bucket.append(now)
            return True


class VKDedupStore:
    """SQLite-хранилище обработанных callback-событий.

    VK может повторить один и тот же webhook при сетевых проблемах. Поэтому
    подтверждаем callback быстро, но повторно не запускаем RAG для того же event.
    """

    def __init__(self, path: Optional[Path] = None, ttl_seconds: int = VK_DEDUP_TTL_SECONDS):
        self.path = path or (METADATA_DIR / "vk_callback_events.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ttl_seconds = max(60, int(ttl_seconds))
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        with self.conn:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS vk_events "
                "(event_key TEXT PRIMARY KEY, seen_at REAL NOT NULL)"
            )

    def mark_seen(self, event_key: str) -> bool:
        """Вернуть True, если событие уже было обработано раньше."""
        now = time.time()
        with self._lock:
            cur = self.conn.execute("SELECT 1 FROM vk_events WHERE event_key = ?", (event_key,))
            if cur.fetchone():
                return True
            with self.conn:
                self.conn.execute(
                    "INSERT INTO vk_events(event_key, seen_at) VALUES (?, ?)",
                    (event_key, now),
                )
            return False

    def purge_old(self) -> int:
        cutoff = time.time() - self.ttl_seconds
        with self._lock:
            with self.conn:
                cur = self.conn.execute("DELETE FROM vk_events WHERE seen_at < ?", (cutoff,))
                return int(cur.rowcount or 0)

    def close(self) -> None:
        with self._lock:
            self.conn.close()


class VKUserPrefStore:
    """Хранит выбранную пользователем модель КОИБ и «отложенный» вопрос.

    Это ключевое для бота: инструкции КОИБ-2010 (физические кнопки ДА/НЕТ) и
    КОИБ-2017А/Б (сенсорный экран) кардинально различаются, поэтому ответы
    жёстко фильтруются по модели. Префа сохраняется между сообщениями, чтобы
    не спрашивать модель на каждый вопрос.

    Значения model:
        "koib2010" | "koib2017a" | "koib2017b" — конкретная модель;
        "any"  — пользователь выбрал «любая / не знаю» (фильтр не жёсткий);
        строка отсутствует — пользователь ещё не выбирал (нужно спросить).
    """

    def __init__(self, path: Optional[Path] = None):
        self.path = path or (METADATA_DIR / "vk_user_prefs.db")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        with self.conn:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS vk_user_prefs ("
                "user_id TEXT PRIMARY KEY, model TEXT, pending_query TEXT, updated_at REAL)"
            )

    def get_model(self, user_id: str) -> Optional[str]:
        with self._lock:
            cur = self.conn.execute("SELECT model FROM vk_user_prefs WHERE user_id = ?", (user_id,))
            row = cur.fetchone()
        if not row or row[0] is None:
            return None
        return str(row[0])

    def set_model(self, user_id: str, model: str) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO vk_user_prefs(user_id, model, pending_query, updated_at) "
                "VALUES (?, ?, COALESCE((SELECT pending_query FROM vk_user_prefs WHERE user_id = ?), ''), ?) "
                "ON CONFLICT(user_id) DO UPDATE SET model = excluded.model, updated_at = excluded.updated_at",
                (user_id, model, user_id, time.time()),
            )

    def clear_model(self, user_id: str) -> None:
        with self._lock, self.conn:
            self.conn.execute("DELETE FROM vk_user_prefs WHERE user_id = ?", (user_id,))

    def set_pending(self, user_id: str, query: str) -> None:
        with self._lock, self.conn:
            self.conn.execute(
                "INSERT INTO vk_user_prefs(user_id, model, pending_query, updated_at) "
                "VALUES (?, (SELECT model FROM vk_user_prefs WHERE user_id = ?), ?, ?) "
                "ON CONFLICT(user_id) DO UPDATE SET pending_query = excluded.pending_query, "
                "updated_at = excluded.updated_at",
                (user_id, user_id, query, time.time()),
            )

    def pop_pending(self, user_id: str) -> str:
        with self._lock, self.conn:
            cur = self.conn.execute("SELECT pending_query FROM vk_user_prefs WHERE user_id = ?", (user_id,))
            row = cur.fetchone()
            pending = str(row[0]) if row and row[0] else ""
            if pending:
                self.conn.execute("UPDATE vk_user_prefs SET pending_query = '' WHERE user_id = ?", (user_id,))
        return pending

    def close(self) -> None:
        with self._lock:
            self.conn.close()


def build_model_keyboard() -> str:
    """VK inline-клавиатура выбора модели КОИБ. Возвращает JSON-строку."""
    def button(label: str, model: str, color: str) -> Dict[str, Any]:
        return {
            "action": {
                "type": "text",
                "label": label,
                "payload": json.dumps({"cmd": "set_model", "model": model}, ensure_ascii=False),
            },
            "color": color,
        }

    keyboard = {
        "inline": True,
        "buttons": [
            [button("КОИБ-2010", "koib2010", "primary"),
             button("КОИБ-2017А", "koib2017a", "primary")],
            [button("КОИБ-2017Б", "koib2017b", "primary"),
             button("Не знаю / любая", _MODEL_ANY, "secondary")],
        ],
    }
    return json.dumps(keyboard, ensure_ascii=False)


class VKBotService:
    """Сервисная логика VK-бота: validation, commands, RAG, VK API send."""

    def __init__(self, pipeline_factory: Callable[[], Any]):
        self.pipeline_factory = pipeline_factory
        self.user_limiter = SlidingWindowRateLimiter(VK_RATE_LIMIT_PER_MINUTE)
        self.global_limiter = SlidingWindowRateLimiter(VK_GLOBAL_RATE_LIMIT_PER_MINUTE)
        self.dedup = VKDedupStore()
        self.prefs = VKUserPrefStore()
        self._last_purge = 0.0

    def validate_callback(self, raw_data: Dict[str, Any]) -> bool:
        """Проверить secret/group_id. Confirmation также должен пройти эту проверку."""
        if VK_SECRET_KEY and raw_data.get("secret") != VK_SECRET_KEY:
            logger.warning("VK callback отклонён: неверный secret")
            return False
        if VK_GROUP_ID:
            try:
                if int(raw_data.get("group_id", 0)) != int(VK_GROUP_ID):
                    logger.warning("VK callback отклонён: неверный group_id")
                    return False
            except Exception:
                logger.warning("VK callback отклонён: group_id не является числом")
                return False
        return True

    @staticmethod
    def is_confirmation(raw_data: Dict[str, Any]) -> bool:
        return raw_data.get("type") == "confirmation"

    @staticmethod
    def confirmation_code() -> str:
        return VK_CONFIRM_CODE

    def parse_message(self, raw_data: Dict[str, Any]) -> Optional[VKIncomingMessage]:
        if raw_data.get("type") != "message_new":
            return None

        obj = raw_data.get("object") or {}
        message = obj.get("message") or obj
        if not isinstance(message, dict):
            return None

        from_id = int(message.get("from_id") or 0)
        if from_id <= 0:
            # Игнорируем сообщения от групп/сервисных отправителей.
            return None

        peer_id = int(message.get("peer_id") or from_id)
        text = str(message.get("text") or "").strip()
        payload = str(message.get("payload") or "").strip()
        # Текст может быть пустым, если это нажатие кнопки (приходит только payload).
        if not text and not payload:
            return None

        message_id = int(message.get("id") or 0)
        conversation_message_id = int(message.get("conversation_message_id") or 0)
        date = int(message.get("date") or 0)
        event_key = self._build_event_key(raw_data, from_id, peer_id, text, message_id, conversation_message_id, date)
        return VKIncomingMessage(
            event_key=event_key,
            user_id=from_id,
            peer_id=peer_id,
            text=text,
            message_id=message_id,
            conversation_message_id=conversation_message_id,
            date=date,
            payload=payload,
        )

    @staticmethod
    def _build_event_key(
        raw_data: Dict[str, Any],
        user_id: int,
        peer_id: int,
        text: str,
        message_id: int,
        conversation_message_id: int,
        date: int,
    ) -> str:
        raw_event_id = raw_data.get("event_id")
        if raw_event_id:
            return str(raw_event_id)
        group_id = raw_data.get("group_id", "")
        base = f"{group_id}:{raw_data.get('type')}:{user_id}:{peer_id}:{message_id}:{conversation_message_id}:{date}:{text[:200]}"
        return hashlib.sha256(base.encode("utf-8")).hexdigest()

    def should_process(self, message: VKIncomingMessage) -> Tuple[bool, str]:
        """Дешёвые проверки до постановки background task."""
        now = time.time()
        if now - self._last_purge > 3600:
            self._last_purge = now
            try:
                purged = self.dedup.purge_old()
                if purged:
                    logger.info("Очищено старых VK event_id: %s", purged)
            except Exception as exc:
                logger.debug("Не удалось очистить VK dedup store: %s", exc)

        if message.is_group_chat and not VK_REPLY_IN_GROUP_CHATS:
            return False, "group_chat_disabled"
        if self.dedup.mark_seen(message.event_key):
            return False, "duplicate"
        return True, "ok"

    async def process_message(self, message: VKIncomingMessage, session: aiohttp.ClientSession) -> None:
        """Обработать сообщение: payload-кнопки -> команды -> safety -> RAG -> send."""
        user_key = str(message.user_id)
        if not self.global_limiter.allow("global") or not self.user_limiter.allow(user_key):
            await self.send_message(message.peer_id, "Слишком много сообщений. Повторите запрос позже.", session)
            return

        # 1) Нажатие кнопки выбора модели (приходит payload). Обрабатываем до всего.
        if message.payload:
            handled = await self._handle_model_payload(message, session)
            if handled:
                return

        query = self._prepare_text(message.text, message.is_group_chat)
        if not query:
            return

        command_response = await self._handle_command(query, message, session)
        if command_response is not None:
            if command_response:  # пустая строка => команда уже сама всё отправила
                await self.send_message(message.peer_id, command_response, session)
            return

        # Small talk: приветствия/благодарности не гоняем через RAG —
        # отвечаем по-человечески и сразу.
        smalltalk = self._handle_smalltalk(query)
        if smalltalk:
            await self.send_message(message.peer_id, smalltalk, session)
            return

        if len(query) > VK_MAX_INCOMING_CHARS:
            await self.send_message(
                message.peer_id,
                f"Запрос слишком длинный. Сократите его до {VK_MAX_INCOMING_CHARS} символов и отправьте снова.",
                session,
            )
            return

        is_safe, reason = check_query_safety(query)
        if not is_safe:
            await self.send_message(message.peer_id, self._blocked_topic_response(reason), session)
            return

        # 2) Определяем модель: текст сообщения -> сохранённая префа -> спросить.
        model_filter, ask_needed = self._resolve_user_model(user_key, query)
        if ask_needed:
            # Запоминаем вопрос и просим выбрать модель кнопками.
            self.prefs.set_pending(user_key, query)
            await self.send_message(
                message.peer_id,
                "Для точного ответа уточните модель КОИБ — инструкции для разных моделей "
                "(физические кнопки у КОИБ-2010 против сенсорного экрана у КОИБ-2017) различаются.\n"
                "Выберите модель кнопкой ниже (можно сменить позже командой /model):",
                session,
                keyboard=build_model_keyboard(),
            )
            return

        await self._answer_query(message, query, model_filter, session)

    async def _answer_query(
        self,
        message: VKIncomingMessage,
        query: str,
        model_filter: str,
        session: aiohttp.ClientSession,
    ) -> None:
        """Прогнать вопрос через RAG с выбранной моделью и отправить ответ."""
        user_key = str(message.user_id)
        if VK_SEND_TYPING:
            await self.send_activity(message.peer_id, session)

        # model_filter == "" означает «искать без жёсткого фильтра» (выбор «любая»).
        strict = bool(model_filter) and MODEL_FILTER_STRICT
        figures: List[Dict[str, Any]] = []
        try:
            result = await self.pipeline_factory().answer(
                query=query,
                user_id=user_key,
                k=FINAL_TOP_K,
                model_filter=model_filter,
                use_memory=True,
                validate=True,
                strict_model=strict,
            )
            answer = self._format_rag_answer(query, result, model_filter)
            figures = result.get("figures") or []
        except Exception as exc:
            logger.exception("Ошибка обработки VK сообщения через RAG: %s", exc)
            answer = VK_ERROR_MESSAGE

        is_answer_safe, _ = check_answer_safety(answer)
        if not is_answer_safe:
            answer = sanitize_answer(answer)

        # Прикрепляем релевантные рисунки (загрузка фото в VK) к ответу.
        attachment = ""
        if figures and SEND_FIGURES_ENABLED:
            try:
                attachment = await self._upload_figures(message.peer_id, figures, session)
            except Exception as exc:
                logger.debug("Не удалось приложить рисунки: %s", exc)

        await self.send_message(message.peer_id, answer, session, attachment=attachment)

    async def _handle_model_payload(self, message: VKIncomingMessage, session: aiohttp.ClientSession) -> bool:
        """Обработать нажатие кнопки выбора модели. True, если payload распознан."""
        try:
            data = json.loads(message.payload)
        except Exception:
            return False
        if not isinstance(data, dict) or data.get("cmd") != "set_model":
            return False

        user_key = str(message.user_id)
        model = str(data.get("model") or "").strip()
        if model not in KNOWN_MODELS and model != _MODEL_ANY:
            return False

        self.prefs.set_model(user_key, model)
        label = "любая модель" if model == _MODEL_ANY else model_label(model)
        confirm = f"Принято. Модель: {label}. Сменить — командой /model."
        await self.send_message(message.peer_id, confirm, session)

        # Если был отложенный вопрос — отвечаем на него сразу с выбранной моделью.
        pending = self.prefs.pop_pending(user_key)
        if pending:
            model_filter = "" if model == _MODEL_ANY else model
            await self._answer_query(message, pending, model_filter, session)
        return True

    def _resolve_user_model(self, user_key: str, query: str) -> Tuple[str, bool]:
        """Вернуть (model_filter, ask_needed).

        Приоритет: явная модель в ТЕКСТЕ вопроса -> сохранённая префа -> спросить.
        model_filter == "" значит «без жёсткого фильтра» (любая модель).
        """
        detected = detect_model_in_text(query)
        if detected.name in KNOWN_MODELS and detected.confidence >= _MODEL_TEXT_MIN_CONFIDENCE:
            # Пользователь сам назвал модель в вопросе — используем и запоминаем.
            self.prefs.set_model(user_key, detected.name)
            return detected.name, False

        stored = self.prefs.get_model(user_key)
        if stored is None:
            return "", True  # ещё не выбирал — спросим
        if stored == _MODEL_ANY:
            return "", False  # выбрал «любая»
        if stored in KNOWN_MODELS:
            return stored, False
        return "", False

    def _prepare_text(self, text: str, is_group_chat: bool) -> str:
        text = (text or "").strip()
        text = re.sub(r"^\[club\d+\|[^\]]+\]\s*[,;:—-]?\s*", "", text, flags=re.IGNORECASE)
        if is_group_chat:
            lowered = text.lower().strip()
            for alias in VK_BOT_MENTION_ALIASES:
                pattern = rf"^(?:@?{re.escape(alias)})\s*[,;:—-]?\s*"
                text = re.sub(pattern, "", text, flags=re.IGNORECASE).strip()
                if text.lower().strip() != lowered:
                    break
        return text.strip()

    async def _handle_command(self, query: str, message: VKIncomingMessage,
                              session: aiohttp.ClientSession) -> Optional[str]:
        command = query.lower().strip()
        command = re.sub(r"\s+", " ", command)
        if command in {"/start", "start", "начать"}:
            return self._start_text()
        if command in {"/help", "help", "помощь", "что ты умеешь"}:
            return self._help_text()
        if command in {"/model", "model", "модель", "сменить модель", "выбрать модель"}:
            user_key = str(message.user_id)
            stored = self.prefs.get_model(user_key)
            if stored == _MODEL_ANY:
                current = "сейчас выбрана: любая модель"
            elif stored in KNOWN_MODELS:
                current = f"сейчас выбрана: {model_label(stored)}"
            else:
                current = "модель пока не выбрана"
            await self.send_message(
                message.peer_id,
                f"Выбор модели КОИБ ({current}).\nОтветы фильтруются по модели, "
                "так как интерфейсы 2010 и 2017 различаются. Выберите кнопкой:",
                session,
                keyboard=build_model_keyboard(),
            )
            return ""  # уже отправили сами
        if command in {"/reset", "reset", "сброс", "очистить историю"}:
            try:
                await self.pipeline_factory().memory.clear_history(str(message.user_id))
            except Exception:
                # Если pipeline ещё тяжело грузится, очищаем память напрямую.
                from src.utils import ConversationMemory

                await ConversationMemory().clear_history(str(message.user_id))
            return "История диалога очищена. Следующий вопрос будет обработан без предыдущего контекста."
        if command in {"/health", "health", "статус"}:
            if str(message.user_id) not in VK_ADMIN_IDS:
                return "Команда /health доступна только администратору бота."
            return f"KOIB RAG VK Bot работает. Версия: {APP_VERSION}."
        return None

    @staticmethod
    def _handle_smalltalk(query: str) -> Optional[str]:
        """Короткие человеческие ответы на приветствия/благодарности (без RAG).

        Срабатывает только на КОРОТКИХ сообщениях, чтобы не перехватывать
        реальные вопросы вида «Привет, что делать если КОИБ не включается?».
        """
        text = query.lower().strip(" .!?,")
        words = text.split()
        if len(words) > 4:
            return None

        greetings = {"привет", "здравствуйте", "здравствуй", "добрый", "доброе",
                     "хай", "приветствую", "доброго", "ку", "здрасте", "салют"}
        thanks = {"спасибо", "благодарю", "спс", "благодарствую", "пасибо", "thanks", "thx"}
        farewells = {"пока", "досвидания", "свидания", "бывай", "до"}

        first = words[0] if words else ""
        if first in greetings:
            return (
                "Здравствуйте! 👋 Я помогу с технической эксплуатацией КОИБ.\n"
                "Опишите, что произошло или что нужно сделать — например: "
                "«КОИБ не принимает бюллетень» или «как опечатать накопитель». "
                "Если знаете модель (КОИБ-2010 / 2017А / 2017Б), укажите её — так ответ будет точнее."
            )
        if first in thanks or any(w in thanks for w in words):
            return "Пожалуйста! Рад помочь. Если появятся ещё вопросы по КОИБ — пишите."
        if first in farewells or text in {"до свидания", "всего доброго", "всего хорошего"}:
            return "Всего доброго! Обращайтесь, если возникнут вопросы по работе с КОИБ."
        return None


        return (
            "Здравствуйте. Я VK-бот KOIB RAG: помогаю искать ответы по технической документации КОИБ.\n\n"
            "Задайте вопрос обычным текстом, например: «Что делать, если КОИБ не включается?»\n"
            "Важно: интерфейсы моделей различаются (у КОИБ-2010 — физические кнопки ДА/НЕТ, "
            "у КОИБ-2017А/Б — сенсорный экран), поэтому ответы фильтруются по модели. "
            "Укажите модель прямо в вопросе или выберите её командой /model.\n\n"
            "Команды: /help — справка, /model — выбрать модель КОИБ, /reset — очистить историю диалога.\n\n"
            "При нештатных ситуациях я дополнительно напомню о процессуальной обязанности: "
            "проинформировать председателя участковой комиссии и сообщить об инциденте на горячую линию технической поддержки."
        )

    @staticmethod
    def _help_text() -> str:
        return (
            "Как пользоваться:\n"
            "1. Пишите короткий конкретный вопрос по КОИБ.\n"
            "2. Указывайте модель (КОИБ-2010 / 2017А / 2017Б), симптом и этап работы, если известно.\n"
            "3. Модель можно задать кнопками: команда /model. Если не указать — спрошу один раз и запомню.\n"
            "4. Для продолжения диалога можно писать уточнения: бот учитывает последние сообщения.\n"
            "5. /reset очищает историю диалога; /model меняет выбранную модель.\n\n"
            "Я отвечаю по найденным фрагментам документации и показываю источники. "
            "По юридическим решениям, жалобам и статусу бюллетеней обращайтесь к председателю комиссии и официальному регламенту."
        )

    @staticmethod
    def _blocked_topic_response(reason: str) -> str:
        return (
            "Я не должен давать процедурное или юридическое решение по этому вопросу в VK-боте. "
            f"Причина: {reason}\n\n"
            "Обратитесь к председателю участковой комиссии и действуйте по официальному регламенту. "
            "Если одновременно есть технический инцидент с КОИБ, сообщите о нём на горячую линию технической поддержки."
        )

    def _format_rag_answer(self, query: str, result: Dict[str, Any], model_filter: str = "") -> str:
        status = result.get("status", "approved")
        answer = str(result.get("answer") or "Не удалось сгенерировать ответ.").strip()
        if status == "rejected":
            answer = get_blocked_response()
        elif status == "review" and not answer.startswith("Требуется проверка"):
            answer = "Требуется проверка по официальной документации.\n\n" + answer

        sources = result.get("sources") or []
        answer = self._append_sources_if_needed(answer, sources)
        if PROCEDURAL_REMINDER_ENABLED:
            answer = ensure_procedural_reminder(answer, query, force=detect_incident(query))

        # Шапка с выбранной моделью — чтобы пользователь видел контекст ответа.
        effective = model_filter or result.get("model_filter") or ""
        if effective in KNOWN_MODELS:
            header = f"🗳 Модель: {model_label(effective)}\n\n"
        else:
            header = "🗳 Модель: любая (для точности уточните через /model)\n\n"
        return (header + answer).strip()

    @staticmethod
    def _append_sources_if_needed(answer: str, sources: Iterable[Dict[str, Any]]) -> str:
        if "[Документ:" in answer:
            return answer
        unique: List[str] = []
        seen = set()
        for src in sources:
            doc = str(src.get("document") or "").strip()
            page = src.get("page", "")
            if not doc:
                continue
            key = (doc, page)
            if key in seen:
                continue
            seen.add(key)
            unique.append(f"{doc}, стр. {page}")
            if len(unique) >= 3:
                break
        if not unique:
            return answer
        return answer.rstrip() + "\n\nИсточники, найденные системой:\n" + "\n".join(f"- {item}" for item in unique)

    async def _upload_figures(self, peer_id: int, figures: List[Dict[str, Any]],
                              session: aiohttp.ClientSession) -> str:
        """Загрузить рисунки в VK и вернуть строку attachment вида 'photo<owner>_<id>,...'.

        Протокол VK: photos.getMessagesUploadServer -> POST файла на upload_url ->
        photos.saveMessagesPhoto. Любая ошибка по отдельному файлу не ломает ответ.
        """
        if not VK_ACCESS_TOKEN:
            return ""
        attachments: List[str] = []
        for fig in figures[:MAX_FIGURES_PER_ANSWER]:
            path = fig.get("image_path")
            if not path:
                continue
            try:
                attach = await self._upload_single_photo(peer_id, Path(path), session)
                if attach:
                    attachments.append(attach)
            except Exception as exc:
                logger.debug("VK upload figure error (%s): %s", path, exc)
        return ",".join(attachments)

    async def _upload_single_photo(self, peer_id: int, path: Path,
                                   session: aiohttp.ClientSession) -> str:
        if not path.is_file():
            return ""
        # 1) получить upload server
        async with session.post(
            VK_API_URL.format(method="photos.getMessagesUploadServer"),
            data={"peer_id": peer_id, "access_token": VK_ACCESS_TOKEN, "v": VK_API_VERSION},
            timeout=aiohttp.ClientTimeout(total=VK_OUTBOUND_TIMEOUT),
        ) as resp:
            payload = await resp.json(content_type=None)
        upload_url = (payload.get("response") or {}).get("upload_url")
        if not upload_url:
            return ""

        # 2) загрузить файл
        form = aiohttp.FormData()
        form.add_field("photo", path.read_bytes(), filename=path.name,
                       content_type="image/png")
        async with session.post(
            upload_url, data=form,
            timeout=aiohttp.ClientTimeout(total=VK_OUTBOUND_TIMEOUT),
        ) as resp:
            up = await resp.json(content_type=None)
        if not up.get("photo") or up.get("photo") == "[]":
            return ""

        # 3) сохранить фото
        async with session.post(
            VK_API_URL.format(method="photos.saveMessagesPhoto"),
            data={
                "photo": up.get("photo"),
                "server": up.get("server"),
                "hash": up.get("hash"),
                "access_token": VK_ACCESS_TOKEN,
                "v": VK_API_VERSION,
            },
            timeout=aiohttp.ClientTimeout(total=VK_OUTBOUND_TIMEOUT),
        ) as resp:
            saved = await resp.json(content_type=None)
        items = saved.get("response") or []
        if not items:
            return ""
        photo = items[0]
        return f"photo{photo['owner_id']}_{photo['id']}"

    async def send_activity(self, peer_id: int, session: aiohttp.ClientSession) -> bool:
        if not VK_ACCESS_TOKEN:
            return False
        payload = {
            "peer_id": peer_id,
            "type": "typing",
            "access_token": VK_ACCESS_TOKEN,
            "v": VK_API_VERSION,
        }
        try:
            async with session.post(
                VK_API_URL.format(method="messages.setActivity"),
                data=payload,
                timeout=aiohttp.ClientTimeout(total=min(5, VK_OUTBOUND_TIMEOUT)),
            ) as resp:
                data = await resp.json(content_type=None)
                if "error" in data:
                    logger.debug("VK setActivity error: %s", data["error"])
                    return False
                return True
        except Exception as exc:
            logger.debug("Не удалось отправить VK typing activity: %s", exc)
            return False

    async def send_message(self, peer_id: int, text: str, session: aiohttp.ClientSession,
                           keyboard: Optional[str] = None, attachment: str = "") -> bool:
        if not VK_ACCESS_TOKEN:
            logger.warning("VK_ACCESS_TOKEN не задан, сообщение не отправлено")
            return False
        chunks = split_vk_message(text, max_chars=VK_MAX_OUTGOING_CHARS)
        ok = True
        last = len(chunks) - 1
        for i, chunk in enumerate(chunks):
            # Клавиатуру и вложение прикрепляем только к последнему сообщению серии.
            kb = keyboard if i == last else None
            att = attachment if i == last else ""
            sent = await self._send_one_message(peer_id, chunk, session, keyboard=kb, attachment=att)
            ok = ok and sent
            if len(chunks) > 1:
                await asyncio.sleep(0.2)
        return ok

    async def _send_one_message(self, peer_id: int, text: str, session: aiohttp.ClientSession,
                                keyboard: Optional[str] = None, attachment: str = "") -> bool:
        data = {
            "peer_id": peer_id,
            "message": text,
            "access_token": VK_ACCESS_TOKEN,
            "v": VK_API_VERSION,
            "random_id": secrets.randbits(31),
        }
        if keyboard:
            data["keyboard"] = keyboard
        if attachment:
            data["attachment"] = attachment
        transient_codes = {6, 9, 10}
        for attempt in range(2):
            try:
                async with session.post(
                    VK_API_URL.format(method="messages.send"),
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=VK_OUTBOUND_TIMEOUT),
                ) as resp:
                    payload = await resp.json(content_type=None)
                    if "error" not in payload:
                        return True
                    error = payload["error"]
                    code = int(error.get("error_code", 0) or 0)
                    logger.warning("VK messages.send error: %s", error)
                    if code in transient_codes and attempt == 0:
                        await asyncio.sleep(0.7)
                        continue
                    return False
            except Exception as exc:
                logger.warning("Ошибка отправки VK сообщения: %s", exc)
                if attempt == 0:
                    await asyncio.sleep(0.7)
                    continue
                return False
        return False

    def close(self) -> None:
        self.dedup.close()
        try:
            self.prefs.close()
        except Exception:
            pass


def split_vk_message(text: str, max_chars: int = VK_MAX_OUTGOING_CHARS) -> List[str]:
    """Разбить длинный ответ под лимит VK, стараясь сохранять абзацы."""
    text = (text or "").strip()
    max_chars = max(100, int(max_chars))
    if not text:
        return [""]
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    remaining = text
    while remaining:
        if len(remaining) <= max_chars:
            chunks.append(remaining.strip())
            break
        cut = remaining.rfind("\n\n", 0, max_chars)
        if cut < max_chars * 0.5:
            cut = remaining.rfind("\n", 0, max_chars)
        if cut < max_chars * 0.5:
            cut = remaining.rfind(". ", 0, max_chars)
            if cut != -1:
                cut += 1
        if cut < max_chars * 0.5:
            cut = max_chars
        chunks.append(remaining[:cut].strip())
        remaining = remaining[cut:].strip()
    return [chunk for chunk in chunks if chunk]
