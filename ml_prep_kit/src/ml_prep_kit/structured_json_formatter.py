"""Formatador reutilizável para logs estruturados em JSON."""

import json
import logging

CAMPOS_TECNICOS = {
    "args",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "taskName",
    "thread",
    "threadName",
}


class StructuredJsonFormatter(logging.Formatter):
    """Formata registros de log como JSON.

    Use esta classe para padronizar logs estruturados em scripts, notebooks ou
    pipelines. Os campos principais usam português brasileiro.

    Exemplo:
        handler = logging.StreamHandler()
        handler.setFormatter(StructuredJsonFormatter())

        logger = logging.getLogger(__name__)
        logger.addHandler(handler)
        logger.info(
            "Treino iniciado.",
            extra={"evento": "treino_iniciado"},
        )
    """

    def format(self, registro: logging.LogRecord) -> str:
        """Retorna uma linha de log estruturada em JSON.

        Exemplo:
            registro = logging.LogRecord(
                name="treino",
                level=logging.INFO,
                pathname="",
                lineno=1,
                msg="Treino iniciado.",
                args=(),
                exc_info=None,
            )
            registro.evento = "treino_iniciado"

            linha = StructuredJsonFormatter().format(registro)
        """
        # Monta os campos principais que todo log deve ter.
        conteudo = {
            "momento": self.formatTime(registro),
            "nivel": registro.levelname,
            "origem": registro.name,
            "mensagem": registro.getMessage(),
        }

        # Adiciona os campos extras enviados pelo projeto em LOGGER.info.
        for campo, valor in registro.__dict__.items():
            if campo not in CAMPOS_TECNICOS:
                conteudo[campo] = valor

        # Converte o dicionário em uma linha JSON.
        return json.dumps(conteudo, ensure_ascii=False, default=str)
