FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.16 /uv /bin/uv

# libgomp1 is required at runtime by the xgboost and lightgbm wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Patch the base image's system pip/setuptools/wheel so the scanner does not
# flag their bundled CVEs; the app itself runs from the uv-managed .venv.
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app
ENV UV_PROJECT_ENVIRONMENT=/app/.venv \
    UV_COMPILE_BYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY pyproject.toml uv.lock ./
# Only project (runtime) dependencies; the dev and pipeline groups stay out.
RUN uv sync --frozen --no-default-groups --no-install-project

COPY src ./src
COPY params.yaml ./
# feature_logic reads the frozen V set at import; k8s has no feature_repo mount.
COPY feature_repo/v_selected.json ./feature_repo/v_selected.json

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONPATH=/app/src
