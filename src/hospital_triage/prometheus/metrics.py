from prometheus_client import Counter, Histogram

PREDICTIONS_TOTAL = Counter(
    "total_de_predicoes",
    "Total de predições realizadas",
)
ERRORS_TOTAL = Counter(
    "total_de_erros_na_api",
    "Total de respostas HTTP com erro e exceções não tratadas",
)
PREDICTION_DURATION = Histogram(
    "duracao_da_predicao_segundos",
    "Duração do processamento da predição em segundos",
)
PREDICTION_CONFIDENCE = Histogram(
    "confianca_da_predicao",
    "Maior probabilidade retornada em cada predição",
    buckets=(0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0),
)
