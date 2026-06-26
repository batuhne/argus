FROM python:3.11-slim

# psycopg2-binary for the Postgres backend store, boto3 for the MinIO/S3
# artifact store. Pinned so the tracking server image is reproducible.
RUN pip install --no-cache-dir \
    mlflow==3.1.4 \
    psycopg2-binary==2.9.10 \
    boto3==1.35.36 \
    && useradd --uid 1000 --create-home mlflow

USER 1000

EXPOSE 5000
