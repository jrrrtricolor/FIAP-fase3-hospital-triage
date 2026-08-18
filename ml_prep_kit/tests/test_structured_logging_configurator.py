"""Testes da configuração de logging estruturado."""

import logging
import unittest
from io import StringIO

from ml_prep_kit.structured_json_formatter import StructuredJsonFormatter

from ml_prep_kit import StructuredLoggingConfigurator


class StructuredLoggingConfiguratorTest(unittest.TestCase):
    """Valida a configuração reutilizável de logging.

    Exemplo:
        StructuredLoggingConfigurator.configure()
        logging.getLogger(__name__).info(
            "Treino iniciado.",
            extra={"evento": "treino_iniciado"},
        )
    """

    def test_configure_adds_structured_json_formatter(self) -> None:
        """O logger raiz deve usar o formatter estruturado.

        Exemplo:
            StructuredLoggingConfigurator.configure()
        """
        StructuredLoggingConfigurator.configure()

        logger_raiz = logging.getLogger()

        self.assertEqual(logger_raiz.level, logging.INFO)
        self.assertIsInstance(
            logger_raiz.handlers[0].formatter,
            StructuredJsonFormatter,
        )

    def test_configured_logger_writes_structured_message(self) -> None:
        """O logger configurado deve escrever uma mensagem JSON.

        Exemplo:
            logger.info(
                "Logging estruturado ativo.",
                extra={"evento": "logging_validado"},
            )
        """
        saida = StringIO()
        manipulador = logging.StreamHandler(saida)
        manipulador.setFormatter(StructuredJsonFormatter())

        logger = logging.getLogger("teste_logging_estruturado")
        logger.handlers.clear()
        logger.addHandler(manipulador)
        logger.setLevel(logging.INFO)
        logger.propagate = False

        logger.info(
            "Logging estruturado ativo.",
            extra={"evento": "logging_validado"},
        )

        mensagem = saida.getvalue()

        self.assertIn('"mensagem": "Logging estruturado ativo."', mensagem)
        self.assertIn('"evento": "logging_validado"', mensagem)


if __name__ == "__main__":
    unittest.main()
