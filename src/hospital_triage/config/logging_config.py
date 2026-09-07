import logging
import sys
from datetime import UTC, datetime

from pythonjsonlogger import json


class JsonFormatter(json.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super().add_fields(log_record, record, message_dict)
        log_record["timestamp"] = datetime.now(UTC).isoformat()
        log_record["level"] = record.levelname
        log_record["logger"] = record.name
        log_record["servico"] = "hospital_triage_api"

        if not log_record.get("mensagem"):
            log_record["mensagem"] = record.getMessage()


def setup_api_logger() -> logging.Logger:
    """Configura o logger estruturado uma única vez por processo."""
    logger = logging.getLogger("hospital_triage_api")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
    return logger
