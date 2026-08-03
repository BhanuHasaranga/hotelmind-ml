"""Automatic retraining trigger: checks data-drift and accuracy-drop signals per
model, retrains via the model's MLOpsPipeline class when either threshold is
breached, and lets MLflowTracker.auto_promote_if_better decide Production promotion
(never regresses Production, per decision #2).

NOTE: transitively imports mlflow-dependent pipeline classes when instantiated and
run end-to-end; on hosts where mlflow/evidently cannot be installed (e.g. Python
3.14 without cmake), this module may fail to fully execute -- same caveat as
drift_detector.py. Verification of the end-to-end retrain flow happens in Docker.
"""

from typing import Any

from src.mlops.config.mlops_settings import mlops_settings
from src.mlops.monitoring.drift_detector import get_latest_drift_score
from src.mlops.registry.model_registry import ModelRegistry, ModelStage
from src.mlops.tracking.mlflow_tracker import MLflowTracker
from src.utils.logging import get_logger

logger = get_logger(__name__)

# Primary metric per model_name, mirroring promote.py's MODEL_METRIC_DIRECTION.
_PRIMARY_METRIC_BY_MODEL: dict[str, tuple[str, bool]] = {
    "occupancy_prophet": ("mae", True),
    "occupancy_xgboost": ("mae", True),
    "pricing_xgboost": ("mae", True),
    "restaurant_breakfast": ("mae", True),
    "restaurant_lunch": ("mae", True),
    "restaurant_dinner": ("mae", True),
    "staffing_regression": ("mae", True),
    "churn_xgboost": ("roc_auc", False),
    "churn_random_forest": ("roc_auc", False),
}


def _primary_metric_value(record_metrics: dict[str, float], primary_metric: str) -> float | None:
    return record_metrics.get(primary_metric)


def _accuracy_drop_breached(
    registry: ModelRegistry, model_name: str, primary_metric: str, lower_is_better: bool
) -> bool:
    """Compare the current PRODUCTION record's primary metric against the most
    recent (highest-version) evaluation for this model_name. Returns False (no
    signal, no retrain) if data is missing rather than raising.
    """
    try:
        versions = registry.list_versions(model_name)
    except FileNotFoundError:
        return False
    if not versions:
        return False

    production_versions = [r for r in versions if r.stage == ModelStage.PRODUCTION]
    if not production_versions:
        return False
    production_record = max(production_versions, key=lambda r: r.version)

    latest_record = max(versions, key=lambda r: r.version)
    if latest_record.version == production_record.version:
        return False  # nothing newer to compare against

    prod_value = _primary_metric_value(production_record.metrics, primary_metric)
    latest_value = _primary_metric_value(latest_record.metrics, primary_metric)
    if prod_value is None or latest_value is None:
        return False

    threshold = mlops_settings.RETRAIN_ACCURACY_DROP_THRESHOLD
    if prod_value == 0:
        return False

    if lower_is_better:
        # Higher error (latest) than production is a drop in quality.
        relative_drop = (latest_value - prod_value) / abs(prod_value)
    else:
        # Lower score (latest) than production is a drop in quality.
        relative_drop = (prod_value - latest_value) / abs(prod_value)

    return relative_drop > threshold


def check_and_retrain(model_name_to_pipeline: dict[str, type]) -> dict[str, bool]:
    """For each (model_name, MLOpsPipeline class) pair, check drift + accuracy-drop
    signals; retrain and (conditionally) promote to Production if breached.

    Returns {model_name: bool} indicating whether a retrain was triggered.
    """
    registry = ModelRegistry()
    triggered: dict[str, bool] = {}

    for model_name, pipeline_cls in model_name_to_pipeline.items():
        primary_metric, lower_is_better = _PRIMARY_METRIC_BY_MODEL.get(model_name, ("mae", True))

        drift_score = get_latest_drift_score(model_name)
        drift_breached = (
            drift_score is not None and drift_score > mlops_settings.RETRAIN_DRIFT_SCORE_THRESHOLD
        )

        accuracy_breached = False
        try:
            accuracy_breached = _accuracy_drop_breached(
                registry, model_name, primary_metric, lower_is_better
            )
        except Exception as exc:
            logger.warning("Accuracy-drop check failed for model=%s: %s", model_name, exc)

        should_retrain = drift_breached or accuracy_breached
        triggered[model_name] = should_retrain

        if not should_retrain:
            logger.info(
                "No retrain signal for model=%s (drift_score=%s, accuracy_breached=%s)",
                model_name,
                drift_score,
                accuracy_breached,
            )
            continue

        logger.info(
            "Retrain triggered for model=%s (drift_score=%s > %s: %s; accuracy_breached: %s)",
            model_name,
            drift_score,
            mlops_settings.RETRAIN_DRIFT_SCORE_THRESHOLD,
            drift_breached,
            accuracy_breached,
        )

        try:
            pipeline = pipeline_cls()
            summary: dict[str, Any] = pipeline.run_with_mlops()
            logger.info("Retrain complete for model=%s: %s", model_name, summary)

            registered = summary.get("registered", {})
            info = registered.get(model_name)
            if info is None:
                continue

            tracker = MLflowTracker(getattr(pipeline, "mlflow_experiment_name", model_name))
            beats_production = tracker.auto_promote_if_better(
                model_name=model_name,
                new_version=info["version"],
                new_metrics=info["metrics"],
                registry=registry,
                primary_metric=primary_metric,
                lower_is_better=lower_is_better,
            )
            if beats_production:
                registry.promote(model_name, info["version"], ModelStage.PRODUCTION)
                logger.info(
                    "Promoted model=%s version=%s to Production after retrain",
                    model_name,
                    info["version"],
                )
            else:
                logger.info(
                    "Retrained model=%s version=%s did not beat Production; staying in Staging",
                    model_name,
                    info["version"],
                )
        except Exception as exc:
            logger.exception("Retrain failed for model=%s: %s", model_name, exc)
            triggered[model_name] = False

    return triggered
