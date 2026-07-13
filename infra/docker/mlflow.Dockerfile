# Base pinned by digest for a reproducible image.
FROM python:3.11-slim@sha256:b27df5841f3355e9473f9a516d38a6783b6c8dfeacaf2d14a240f443b368ddb6

# psycopg2-binary for the Postgres backend store, boto3 for the MinIO/S3
# artifact store. Pinned so the tracking server image is reproducible.
RUN pip install --no-cache-dir \
    mlflow==3.14.0 \
    psycopg2-binary==2.9.10 \
    boto3==1.35.36 \
    && useradd --uid 1000 --create-home mlflow

USER 1000

EXPOSE 5000

# Runnable default; compose overrides with the Postgres/S3 config.
CMD ["mlflow", "server", "--host", "0.0.0.0", "--port", "5000"]
