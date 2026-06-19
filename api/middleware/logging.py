# -*- coding: utf-8 -*-
"""
Koib-V-4.5 — Промежуточное ПО логирования
=============================================
Логирование входящих HTTP-запросов: метод, путь, время обработки.
"""
import time
import logging

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("koib.api.middleware")


class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Промежуточное ПО для логирования HTTP-запросов.

    Для каждого запроса записывает:
      - HTTP-метод и путь
      - Время обработки
      - Код ответа
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        start_time = time.time()

        # Вызов следующего обработчика
        response = await call_next(request)

        # Логирование
        process_time = time.time() - start_time
        logger.info(
            f"{request.method} {request.url.path} "
            f"→ {response.status_code} "
            f"({process_time:.3f}с)"
        )

        # Добавляем заголовок с временем обработки
        response.headers["X-Process-Time"] = f"{process_time:.3f}"

        return response
