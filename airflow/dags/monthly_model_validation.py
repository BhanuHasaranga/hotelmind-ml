"""Monthly model validation DAG (M6).

Re-evaluates each domain's current PRODUCTION model against the latest
available feature data and writes an M11 evaluation dashboard entry. This DAG
never retrains — it is a read-only health check of what's currently serving.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator

from airflow import DAG

default_args = {
    "owner": "hotelmind-mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def _evaluate_domain(domain: str, pipeline_cls: type, registry_model_names: list[str]) -> list:
    """Load the pipeline's latest feature data, re-evaluate the current
    Production model(s) for this domain, and return ModelEvalRecord entries.
    Reuses the pipeline's own load/clean/feature/split/evaluate steps (same
    steps `run_with_mlops` composes) rather than reimplementing evaluation.
    """
    from src.mlops.metrics.evaluation_dashboard import ModelEvalRecord
    from src.mlops.registry.model_registry import ModelRegistry
    from src.utils.logging import get_logger

    logger = get_logger(__name__)
    registry = ModelRegistry()
    records: list[ModelEvalRecord] = []

    pipeline = pipeline_cls()
    df = pipeline.load_data()
    df = pipeline.clean(df)
    df = pipeline.engineer_features(df)
    _, X_test, _, y_test = pipeline.split(df)

    for model_name in registry_model_names:
        try:
            record = registry.get_record(model_name)
        except FileNotFoundError:
            logger.warning(
                "monthly_model_validation: no registry record for %s, skipping", model_name
            )
            continue

        try:
            model = registry.load_production(model_name)
            model_key = model_name.split("_", 1)[1] if "_" in model_name else model_name
            metrics = pipeline.evaluate({model_key: model}, X_test, y_test).get(model_key, {})
            records.append(
                ModelEvalRecord(
                    model_name=model_name,
                    version=record.version,
                    mae=metrics.get("mae"),
                    rmse=metrics.get("rmse"),
                    mape=metrics.get("mape"),
                    accuracy=metrics.get("accuracy"),
                    precision=metrics.get("precision"),
                    recall=metrics.get("recall"),
                    f1=metrics.get("f1"),
                    roc_auc=metrics.get("roc_auc"),
                )
            )
        except Exception as exc:
            logger.warning(
                "monthly_model_validation: model=%s evaluation failed: %s", model_name, exc
            )

    return records


def _run_monthly_validation() -> None:
    from src.mlops.metrics.evaluation_dashboard import write_dashboard
    from src.mlops.pipelines.churn_mlops_pipeline import ChurnMLOpsPipeline
    from src.mlops.pipelines.occupancy_mlops_pipeline import OccupancyMLOpsPipeline
    from src.mlops.pipelines.pricing_mlops_pipeline import PricingMLOpsPipeline
    from src.mlops.pipelines.restaurant_mlops_pipeline import RestaurantMLOpsPipeline
    from src.mlops.pipelines.staffing_mlops_pipeline import StaffingMLOpsPipeline
    from src.utils.logging import get_logger

    logger = get_logger(__name__)

    domains = [
        ("occupancy", OccupancyMLOpsPipeline, OccupancyMLOpsPipeline.registry_model_names),
        ("pricing", PricingMLOpsPipeline, PricingMLOpsPipeline.registry_model_names),
        ("restaurant", RestaurantMLOpsPipeline, RestaurantMLOpsPipeline.registry_model_names),
        ("staffing", StaffingMLOpsPipeline, StaffingMLOpsPipeline.registry_model_names),
        ("churn", ChurnMLOpsPipeline, ChurnMLOpsPipeline.registry_model_names),
    ]

    all_records = []
    for domain, pipeline_cls, model_names in domains:
        try:
            all_records.extend(_evaluate_domain(domain, pipeline_cls, model_names))
        except Exception as exc:
            # A failure evaluating one domain must not block the other domains
            # or the dashboard write for whatever did succeed.
            logger.warning("monthly_model_validation: domain=%s failed entirely: %s", domain, exc)

    if all_records:
        write_dashboard(all_records)
    logger.info("monthly_model_validation complete: %d records written", len(all_records))


with DAG(
    dag_id="monthly_model_validation",
    description="Re-evaluate current Production models against latest data; no retraining",
    default_args=default_args,
    schedule="@monthly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["hotelmind", "mlops", "validation"],
) as dag:
    validate = PythonOperator(
        task_id="validate_production_models",
        python_callable=_run_monthly_validation,
    )
