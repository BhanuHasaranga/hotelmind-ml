FROM apache/airflow:2.9.3-python3.12

COPY requirements-training.txt /requirements-training.txt

# NOTE: deliberately NOT installed against Airflow's own constraints file.
# That constraints file pins pandas==2.1.4 (Airflow's own transitive pin),
# which conflicts with this project's pandas>=2.2 requirement used by the
# ML training/feature code. requirements-training.txt only adds packages
# Airflow itself does not depend on (pandas/numpy/scikit-learn/xgboost/
# prophet/mlflow/evidently/etc.), so an unconstrained install here is safe
# and won't touch any package Airflow's core needs.
RUN pip install --no-cache-dir -r /requirements-training.txt
