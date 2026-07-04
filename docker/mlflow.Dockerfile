FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN pip install --no-cache-dir mlflow==2.17.2 psycopg2-binary==2.9.10

RUN useradd --create-home --shell /usr/sbin/nologin mlflow \
    && mkdir -p /mlflow/artifacts \
    && chown -R mlflow:mlflow /mlflow
USER mlflow

EXPOSE 5000

CMD ["sh", "-c", "mlflow server \
    --host 0.0.0.0 \
    --port 5000 \
    --backend-store-uri postgresql+psycopg2://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${MLFLOW_DB} \
    --artifacts-destination /mlflow/artifacts \
    --serve-artifacts"]
