import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Request
from starlette.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware

from ..prometheus.metrics import ERRORS_TOTAL
from .logging_config import setup_api_logger

LOGGER = setup_api_logger()


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request_id = str(uuid.uuid4())
        started_at = time.perf_counter()
        route = request.url.path
        client_host = request.client.host if request.client else None
        LOGGER.info(
            "inicio_de_requisicao",
            extra={
                "request_id": request_id,
                "metodo_http": request.method,
                "rota": route,
            },
        )

        request.state.request_id = request_id
        try:
            response = await call_next(request)
        except Exception:
            ERRORS_TOTAL.inc()
            LOGGER.exception(
                "erro_de_requisicao",
                extra={
                    "request_id": request_id,
                    "metodo_http": request.method,
                    "rota": route,
                    "status_code": 500,
                    "duracao": time.perf_counter() - started_at,
                    "ip_cliente": client_host,
                },
            )
            raise

        duration = time.perf_counter() - started_at
        LOGGER.info(
            "fim_de_requisicao",
            extra={
                "request_id": request_id,
                "metodo_http": request.method,
                "rota": route,
                "status_code": response.status_code,
                "duracao": duration,
                "ip_cliente": client_host,
            },
        )

        if response.status_code >= 400:
            ERRORS_TOTAL.inc()

        response.headers["X-Request-ID"] = request_id
        return response
