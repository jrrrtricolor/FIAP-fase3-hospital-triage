"""Funções utilitárias reutilizáveis."""


def format_currency(value: float) -> str:
    """Formata valores financeiros estimados em dólar.

    Exemplo:
        text = format_currency(1500)
    """
    return f"US$ {value:,.0f}"


def format_percent(value: float) -> str:
    """Formata valores percentuais de forma simples.

    Exemplo:
        text = format_percent(0.153)
    """
    return f"{value:.1%}"
