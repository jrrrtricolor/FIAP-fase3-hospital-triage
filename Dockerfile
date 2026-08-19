# syntax=docker/dockerfile:1

# Builder e runtime reutilizam as mesmas camadas da imagem Python.
FROM python:3.12-slim AS python-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app


# O builder resolve as dependências sem levar Poetry para a imagem final.
FROM python-base AS builder

ENV POETRY_VERSION=2.4.1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry-cache

RUN python -m pip install --no-cache-dir "poetry==$POETRY_VERSION"

# Copiar primeiro os manifests permite reutilizar o cache das dependências.
COPY pyproject.toml poetry.lock ./

RUN poetry install --only main --no-root --no-ansi \
    && rm -rf "$POETRY_CACHE_DIR"


# A imagem final recebe somente o ambiente virtual e o código de inferência.
FROM python-base AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app:/app/src"

RUN groupadd --system app \
    && useradd --system --gid app --home-dir /app app

COPY --from=builder /app/.venv /app/.venv
COPY --chown=app:app src ./src
COPY --chown=app:app ml_prep_kit ./ml_prep_kit

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

# Poetry não é necessário no runtime; o uvicorn vem da .venv do builder.
CMD ["uvicorn", "hospital_triage.api:app", "--host", "0.0.0.0", "--port", "8000"]
