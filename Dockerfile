# syntax=docker/dockerfile:1

# Builder e runtime reutilizam as mesmas camadas da imagem Python.
FROM python:3.12-slim AS python-base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

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


# A imagem final recebe somente o runtime ONNX e o código de inferência.
FROM python-base AS runtime

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH="/app:/app/src" \
    MODEL_PATH="/app/model/hospital_triage_model.onnx" \
    MODEL_VERSION="onnx-nhamcs-2021-v1"

RUN apt-get update \
    && apt-get install --yes --no-install-recommends locales \
    && sed -i 's/^# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
    && locale-gen \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system app \
    && useradd --system --gid app --home-dir /app app

ENV LANG=en_US.UTF-8 \
    LC_ALL=en_US.UTF-8

COPY --from=builder /app/.venv /app/.venv
COPY --chown=app:app src ./src
COPY --chown=app:app ml_prep_kit ./ml_prep_kit
COPY --chown=app:app model/hospital_triage_model.onnx ./model/

USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

# Poetry não é necessário no runtime; o uvicorn vem da .venv do builder.
CMD ["uvicorn", "hospital_triage.api:app", "--host", "0.0.0.0", "--port", "8000"]
