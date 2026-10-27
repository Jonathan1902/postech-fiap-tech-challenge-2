# syntax=docker/dockerfile:1

# ---- poetry: ferramenta de build compartilhada pelos dois venvs ----
# Python igual ao do ambiente local: pickles (model.pkl) idênticos no host e no container.
# Poetry na mesma versão que gerou o poetry.lock.
FROM python:3.13-slim AS poetry

ENV POETRY_VIRTUALENVS_CREATE=false \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN pip install "poetry==2.5.1" && python -m venv /opt/venv
WORKDIR /build
COPY pyproject.toml poetry.lock ./

# ---- builder: dependências de produção (mlflow server e API) ----
# VIRTUAL_ENV faz o Poetry instalar dentro de /opt/venv.
# --no-root: o código entra depois (via PYTHONPATH), então mudar o src não invalida esta camada.
FROM poetry AS builder
RUN VIRTUAL_ENV=/opt/venv poetry install --no-root --only main

# ---- builder-trainer: produção + grupo "pipeline" (dvc) ----
FROM poetry AS builder-trainer
RUN VIRTUAL_ENV=/opt/venv poetry install --no-root --only main,pipeline

# ---- base: runtime enxuto, sem Poetry nem compiladores ----
FROM python:3.13-slim AS base

ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# UID 1000 costuma coincidir com o usuário do host, evitando arquivos de root nos bind mounts.
RUN useradd --uid 1000 --create-home --shell /usr/sbin/nologin app
WORKDIR /app

# ---- mlflow: servidor de tracking/registry (só precisa do venv) ----
FROM base AS mlflow
COPY --from=builder /opt/venv /opt/venv
# O volume nomeado herda o dono deste diretório, permitindo que o usuário não-root grave nele.
RUN mkdir /mlflow && chown app:app /mlflow
USER app
EXPOSE 5000
# --workers 1: o padrão (4) consome ~1,2 GB e faz o trainer sofrer OOM em VMs pequenas.
# --serve-artifacts: o servidor recebe os artefatos por HTTP, então os clientes
# não precisam montar o volume nem conhecer o caminho do disco.
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000", "--workers", "1", \
     "--backend-store-uri", "sqlite:////mlflow/mlflow.db", \
     "--artifacts-destination", "/mlflow/artifacts", "--serve-artifacts", \
     "--allowed-hosts", "mlflow:5000,localhost:5000,127.0.0.1:5000"]

# ---- api: serviço de inferência ----
FROM base AS api
COPY --from=builder /opt/venv /opt/venv
COPY --chown=app:app src ./src
USER app
EXPOSE 8000
CMD ["uvicorn", "shopper_intent.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# ---- trainer: roda o pipeline DVC (dvc + git) ----
# Não copia o código: o compose monta o repositório inteiro em /app (src, dvc.yaml,
# params.yaml, .dvc e data), e os outputs do pipeline voltam para o host.
FROM base AS trainer
RUN apt-get update \
    && apt-get install -y --no-install-recommends git \
    && rm -rf /var/lib/apt/lists/*
COPY --from=builder-trainer /opt/venv /opt/venv
USER app
CMD ["dvc", "repro"]
