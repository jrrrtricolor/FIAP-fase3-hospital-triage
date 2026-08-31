import time
import uuid

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

from src.hospital_triage.config.logging_config import setup_api_logger
from src.hospital_triage.prometheus.metrics import ERRORS_TOTAL


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logger = setup_api_logger()
        request_id = str(uuid.uuid4())
        start_time = time.time()
        logger.info(
            "inicio_de_requisicao",
            extra={
                "request_id": request_id,
                "metodo_http": request.method,
                "url": str(request.url),
            },
        )

        request.state.request_id = request_id
        response = await call_next(request)

        duration = time.time() - start_time
        logger.info(
            "fim_de_requisicao",
            extra={
                "request_id": request_id,
                "metodo_http": request.method,
                "url": str(request.url),
                "status_code": response.status_code,
                "duracao": duration,
                "ip_cliente": request.client.host,
            },
        )

        ERRORS_TOTAL.inc() if response.status_code >= 400 else None

        response.headers["X-Request-ID"] = request_id

        return response
