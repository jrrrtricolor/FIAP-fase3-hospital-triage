"""Configuração reutilizável para logging estruturado."""

import logging
import sys

from ml_prep_kit.structured_json_formatter import StructuredJsonFormatter


class StructuredLoggingConfigurator:
    """Configura logs estruturados para scripts e pipelines.

    Use esta classe quando quiser que a aplicação escreva logs em JSON usando
    o formatador padrão do ``ml_prep_kit``.

    Exemplo:
        StructuredLoggingConfigurator.configure()

        logger = logging.getLogger(__name__)
        logger.info(
            "Treino iniciado.",
            extra={"evento": "treino_iniciado"},
        )
    """

    @classmethod
    def configure(cls, nivel: int = logging.INFO) -> None:
        """Configura o logger raiz para emitir logs estruturados.

        Exemplo:
            StructuredLoggingConfigurator.configure(logging.INFO)
        """
        manipulador = logging.StreamHandler(sys.stdout)
        manipulador.setFormatter(StructuredJsonFormatter())

        logger_raiz = logging.getLogger()
        logger_raiz.handlers.clear()
        logger_raiz.addHandler(manipulador)
        logger_raiz.setLevel(nivel)
