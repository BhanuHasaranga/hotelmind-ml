FROM apache/airflow:2.9.3-python3.12

COPY requirements-training.txt /requirements-training.txt

# Install against Airflow's own constraints file to avoid dependency-resolution
# fights with Airflow's pinned transitive deps (Airflow images are picky about
# this - see https://airflow.apache.org/docs/apache-airflow/stable/installation/installing-from-pypi.html).
RUN pip install --no-cache-dir -r /requirements-training.txt \
    --constraint "https://raw.githubusercontent.com/apache/airflow/constraints-2.9.3/constraints-3.12.txt"
