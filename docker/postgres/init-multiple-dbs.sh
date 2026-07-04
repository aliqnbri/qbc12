#!/bin/bash
# Creates the additional databases (Airflow metadata, MLflow backend) next to
# the main application database on first container start.
set -euo pipefail

create_db() {
    local db_name="$1"
    echo "Ensuring database '${db_name}' exists"
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
        SELECT 'CREATE DATABASE ${db_name}'
        WHERE NOT EXISTS (SELECT FROM pg_database WHERE datname = '${db_name}')\gexec
EOSQL
}

create_db "${AIRFLOW_DB:-airflow}"
create_db "${MLFLOW_DB:-mlflow}"
