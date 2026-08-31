from prometheus_client import Counter, Histogram

PREDICTIONS_TOTAL = Counter(
    "total_de_predicoes", 
    "Total de predições realizadas"
)
ERRORS_TOTAL = Counter(
    "total_de_erros_na_api", 
    "Total de erros encontrados na API"
)
PREDICTION_DURATION = Histogram(
    "duracao_da_predicao_segundos", 
    "Duração do processamento da predição em segundos"
)
AVG_CONFIDENCE = Histogram(
    "confianca_media_da_predicao", 
    "Confiança média das predições"

)
