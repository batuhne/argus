FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /bin/uv

# libgomp1 is required at runtime by the xgboost and lightgbm wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY src ./src
COPY params.yaml ./

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src
