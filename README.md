# Group04 capstone starter

This starter implements the required FastAPI contract, Prometheus metrics, API
test, and an Airflow workflow skeleton. The baseline score uses purchase-time
inputs only: delivery outcome, review, and actual-delivery fields are excluded
to prevent temporal leakage.

## Local setup and tests

Create the project virtual environment from the shared offline wheelhouse:

```bash
cd ~/project
python3 -m venv .venv
.venv/bin/python -m pip install --no-index \
  --find-links /opt/QBC12/MlOps/project/capstone_stack/wheelhouse \
  -r requirements.txt
.venv/bin/python -m pytest -q tests
```

Run the API manually when required:

```bash
.venv/bin/uvicorn app.main:app --host 0.0.0.0 --port 8114
```

The deployed Group04 API is available at `http://127.0.0.1:8114`; use
`/health` for availability, `/metrics` for Prometheus metrics, and `/predict`
for a prediction request. MLflow for Group04 is available locally at
`http://127.0.0.1:5314`.

## Version control and submission

Use GitLab for source control and submission:

```bash
git clone ssh://git@185.50.38.163:2224/qbc12-mlops-capstone/group04-project.git
```

Commit your source, tests, documentation, and dependency manifest to your own
group project. Do not place credentials, `.venv`, data exports, or generated
artifacts in Git.

Grafana is available at `http://185.50.38.163:3010`. Use the Group04 account
issued by the course administrator to view the Group04 organization and its
Prometheus datasource.
