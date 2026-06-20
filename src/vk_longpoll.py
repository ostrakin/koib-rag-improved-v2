# -*- coding: utf-8 -*-
"""VK Bots Long Poll — приёмник событий для бота КОИБ.

В отличие от Callback API (нужен публичный HTTPS-эндпоинт), Long Poll сам
опрашивает сервер VK длинными запросами. Сервер с ботом может стоять за NAT,
без белого IP и сертификата — нужен только исходящий доступ в интернет.

Формат события Long Poll (API 5.x) совпадает с Callback API:
    {"type":"message_new","object":{"message":{...}},"group_id":..,"event_id":".."}
поэтому разбор, дедупликация и обработка переиспользуются из VKBotService
без изменений.

Протокол:
  1) groups.getLongPollServer  → {server, key, ts}
  2) GET {server}?act=a_check&key=..&ts=..&wait=25  → {ts, updates:[...]}
     или {failed: 1|2|3} — обновляем ts/ключ/сервер и продолжаем.
Каждое событие обрабатывается отдельной задачей, чтобы долгая генерация
GigaChat не блокировала опрос. Параллелизм генераций ограничен семафором в
RAGPipeline (MAX_CONCURRENT_GENERATIONS).
"""
from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Dict, Optional, Set

import aiohttp

from config import APP_VERSION, VK_ACCESS_TOKEN, VK_API_VERSION, VK_GROUP_ID
from src.vk_bot import VKBotService, VK_API_URL

logger = logging.getLogger("koib.vk_longpoll")

# wait дольше 25 с VK не рекомендует; таймаут клиента берём с запасом.
VK_LONGPOLL_WAIT = int(os.getenv("VK_LONGPOLL_WAIT", "25"))
VK_LONGPOLL_HTTP_TIMEOUT = VK_LONGPOLL_WAIT + 15


class VKLongPollRunner:
    """Опрашивает Bots Long Poll и передаёт события в VKBotService."""

    def __init__(
        self,
        service: VKBotService,
        *,
        group_id: Optional[str] = None,
        token: Optional[str] = None,
        api_version: Optional[str] = None,
        wait: Optional[int] = None,
    ) -> None:
        self.service = service
        self.group_id = str(group_id or VK_GROUP_ID).strip()
        self.token = (token or VK_ACCESS_TOKEN).strip()
        self.api_version = str(api_version or VK_API_VERSION)
        self.wait = int(wait or VK_LONGPOLL_WAIT)

        self._server: Optional[str] = None
        self._key: Optional[str] = None
        self._ts: Optional[str] = None
        self._session: Optional[aiohttp.ClientSession] = None
        self._stop = asyncio.Event()
        self._tasks: Set[asyncio.Task] = set()

    # ─────────────────────────── HTTP ───────────────────────────
    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=50, ttl_dns_cache=300)
            self._session = aiohttp.ClientSession(connector=connector)
        return self._session

    async def _api(self, method: str, **params: Any) -> Any:
        """Вызов метода VK API с базовой обработкой ошибок."""
        session = await self._ensure_session()
        params.setdefault("access_token", self.token)
        params.setdefault("v", self.api_version)
        async with session.post(
            VK_API_URL.format(method=method),
            data=params,
            timeout=aiohttp.ClientTimeout(total=30),
        ) as resp:
            data = await resp.json(content_type=None)
        if "error" in data:
            raise RuntimeError(f"VK API {method}: {data['error']}")
        return data["response"]

    async def _refresh_server(self, keep_ts: bool = False) -> None:
        """Получить (обновить) адрес Long Poll сервера, ключ и ts."""
        resp = await self._api("groups.getLongPollServer", group_id=self.group_id)
        self._server = resp["server"]
        self._key = resp["key"]
        if not keep_ts or self._ts is None:
            self._ts = resp["ts"]

    async def _check(self) -> Dict[str, Any]:
        session = await self._ensure_session()
        params = {"act": "a_check", "key": self._key, "ts": self._ts, "wait": self.wait}
        async with session.get(
            self._server,  # type: ignore[arg-type]
            params=params,
            timeout=aiohttp.ClientTimeout(total=VK_LONGPOLL_HTTP_TIMEOUT),
        ) as resp:
            return await resp.json(content_type=None)

    # ─────────────────────────── основной цикл ───────────────────────────
    async def run(self) -> None:
        if not self.token or not self.group_id:
            raise SystemExit("Нужны VK_ACCESS_TOKEN и VK_GROUP_ID (см. .env).")

        await self._refresh_server()
        logger.info(
            "VK Long Poll запущен | group_id=%s | версия бота %s | wait=%sс",
            self.group_id, APP_VERSION, self.wait,
        )

        backoff = 1
        while not self._stop.is_set():
            try:
                data = await self._check()
            except (aiohttp.ClientError, asyncio.TimeoutError) as exc:
                logger.warning("Сетевая ошибка long poll: %s (повтор через %sс)", exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
                continue
            except Exception:
                logger.exception("Непредвиденная ошибка опроса, переинициализация сервера")
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30)
                try:
                    await self._refresh_server()
                except Exception:
                    pass
                continue

            backoff = 1

            failed = data.get("failed")
            if failed is not None:
                await self._handle_failed(int(failed), data)
                continue

            self._ts = data.get("ts", self._ts)
            for update in data.get("updates", []):
                self._spawn(self._dispatch(update))

        await self._shutdown()

    async def _handle_failed(self, failed: int, data: Dict[str, Any]) -> None:
        if failed == 1:
            # История событий частично устарела — VK прислал актуальный ts.
            self._ts = data.get("ts", self._ts)
        elif failed == 2:
            logger.info("Long poll: ключ устарел (failed=2), обновляю ключ")
            await self._refresh_server(keep_ts=True)
        elif failed == 3:
            logger.info("Long poll: ts потерян (failed=3), полная переинициализация")
            await self._refresh_server(keep_ts=False)
        else:
            logger.warning("Long poll: неизвестный failed=%s, переинициализация", failed)
            await self._refresh_server(keep_ts=False)

    # ─────────────────────────── обработка события ───────────────────────────
    def _spawn(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    async def _dispatch(self, update: Dict[str, Any]) -> None:
        try:
            if not isinstance(update, dict) or update.get("type") != "message_new":
                return
            message = self.service.parse_message(update)
            if message is None:
                return
            ok, reason = self.service.should_process(message)
            if not ok:
                logger.debug("Событие пропущено: %s", reason)
                return
            session = await self._ensure_session()
            await self.service.process_message(message, session)
        except Exception:
            logger.exception("Ошибка обработки события Long Poll")

    # ─────────────────────────── остановка ───────────────────────────
    def stop(self) -> None:
        self._stop.set()

    async def _shutdown(self) -> None:
        logger.info("Останавливаюсь, дожидаюсь %d активных задач…", len(self._tasks))
        if self._tasks:
            await asyncio.gather(*list(self._tasks), return_exceptions=True)
        if self._session is not None and not self._session.closed:
            await self._session.close()
        logger.info("VK Long Poll остановлен.")
