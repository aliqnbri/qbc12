FROM apache/airflow:2.10.3-python3.12

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*
USER airflow

# The pipeline dependencies (SQLAlchemy 2.x, MLflow, Polars...) conflict with
# Airflow's own pins, so they live in an isolated virtualenv executed through
# ExternalPythonOperator.
COPY docker/requirements-airflow.txt /tmp/requirements-airflow.txt
RUN python -m venv /opt/project-venv \
    && /opt/project-venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/project-venv/bin/pip install --no-cache-dir -r /tmp/requirements-airflow.txt
