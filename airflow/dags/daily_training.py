"""Daily conditional retraining DAG (M6).

Reuses `src.mlops.pipelines.retrain_trigger.check_and_retrain`, which already
encapsulates the drift-score + accuracy-drop decision logic (M12) — this DAG
does not duplicate that logic, it just wires it to a daily schedule.

Requires PYTHONPATH to include the mounted project root (see docker-compose.yml
`x-airflow-common.environment.PYTHONPATH=/opt/airflow/project`) so
`from src.mlops... import ...` resolves inside the worker container.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator

from airflow import DAG

default_args = {
    "owner": "hotelmind-mlops",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}


def _run_daily_training() -> None:
    """Check drift/accuracy-drop signals per domain and retrain only those that
    breach a threshold. Reuses check_and_retrain (M12) verbatim.
    """
    from src.mlops.pipelines.churn_mlops_pipeline import ChurnMLOpsPipeline
    from src.mlops.pipelines.occupancy_mlops_pipeline import OccupancyMLOpsPipeline
    from src.mlops.pipelines.pricing_mlops_pipeline import PricingMLOpsPipeline
    from src.mlops.pipelines.restaurant_mlops_pipeline import RestaurantMLOpsPipeline
    from src.mlops.pipelines.retrain_trigger import check_and_retrain
    from src.mlops.pipelines.staffing_mlops_pipeline import StaffingMLOpsPipeline
    from src.utils.logging import get_logger

    logger = get_logger(__name__)

    # check_and_retrain expects {model_name: pipeline_class}; primary metric
    # models per domain (mirrors _PRIMARY_METRIC_BY_MODEL in retrain_trigger.py).
    model_name_to_pipeline: dict[str, type] = {
        "occupancy_prophet": OccupancyMLOpsPipeline,
        "occupancy_xgboost": OccupancyMLOpsPipeline,
        "pricing_xgboost": PricingMLOpsPipeline,
        "restaurant_breakfast": RestaurantMLOpsPipeline,
        "restaurant_lunch": RestaurantMLOpsPipeline,
        "restaurant_dinner": RestaurantMLOpsPipeline,
        "staffing_regression": StaffingMLOpsPipeline,
        "churn_xgboost": ChurnMLOpsPipeline,
        "churn_random_forest": ChurnMLOpsPipeline,
    }

    results: dict[str, bool] = {}
    for model_name, pipeline_cls in model_name_to_pipeline.items():
        try:
            triggered = check_and_retrain({model_name: pipeline_cls})
            results.update(triggered)
        except Exception as exc:
            # Defensive: one domain's failure must not kill the whole DAG run.
            # We log a warning and continue with the remaining domains.
            logger.warning("daily_training: model=%s failed: %s", model_name, exc)
            results[model_name] = False

    logger.info("daily_training complete: %s", results)


with DAG(
    dag_id="daily_training",
    description="Per-domain conditional retrain based on drift/accuracy-drop signals",
    default_args=default_args,
    schedule="@daily",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["hotelmind", "mlops", "training"],
) as dag:
    run_daily_training = PythonOperator(
        task_id="check_and_retrain_all_domains",
        python_callable=_run_daily_training,
    )
