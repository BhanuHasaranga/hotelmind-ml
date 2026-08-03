FROM ghcr.io/mlflow/mlflow:v2.17.2

# The stock mlflow image ships without a Postgres driver, but this stack's
# MLflow backend store is Postgres (mlflow-db service) per Phase 6's design
# decision to use a real DB backend rather than local SQLite. Add the driver.
RUN pip install --no-cache-dir "psycopg2-binary>=2.9,<3.0"
