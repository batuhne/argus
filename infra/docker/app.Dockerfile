FROM python:3.11-slim AS builder

COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /bin/uv

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

COPY pyproject.toml uv.lock ./
# Runtime dependencies only; the dev and pipeline groups stay out of the image.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-default-groups --no-install-project


FROM python:3.11-slim AS runtime

# libgomp1 backs the xgboost/lightgbm/catboost wheels; the pip upgrade clears the base image CVEs.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir --upgrade pip setuptools wheel \
    && useradd --uid 1000 --no-create-home --home-dir /tmp argus

WORKDIR /app
COPY --from=builder --chown=1000:1000 /app/.venv /app/.venv
COPY --chown=1000:1000 src ./src
COPY --chown=1000:1000 params.yaml ./
# feature_logic reads the frozen V set at import; k8s has no feature_repo mount.
COPY --chown=1000:1000 feature_repo/v_selected.json ./feature_repo/v_selected.json

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp \
    BENTOML_HOME=/tmp/bentoml

USER 1000

# Default serving command; the probe targets serving readiness on 3001.
HEALTHCHECK --interval=10s --timeout=5s --start-period=40s --retries=12 \
    CMD ["python", "-c", "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:3001/readyz').status==200 else 1)"]

CMD ["bentoml", "serve", "src/fraud/serving/service.py:FraudService", "--host=0.0.0.0", "--port=3001"]
