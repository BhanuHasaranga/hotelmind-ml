import importlib.metadata
import platform
import subprocess
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import mlflow

from src.mlops.config.mlops_settings import PROJECT_ROOT, mlops_settings
from src.utils.logging import get_logger

if TYPE_CHECKING:
    from src.mlops.registry.model_registry import ModelRegistry
    from src.mlops.validation.dataset_version import DatasetVersion

logger = get_logger(__name__)

_TRACKED_LIBRARIES = ["pandas", "numpy", "scikit-learn", "xgboost", "prophet", "mlflow"]


def _library_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except Exception:
        return "unknown"


def _git_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return "unknown"
    except Exception:
        return "unknown"


class MLflowTracker:
    """Thin wrapper around the mlflow client for a single experiment."""

    def __init__(self, experiment_name: str) -> None:
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(mlops_settings.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_name)

    @contextmanager
    def start_run(self, run_name: str | None = None):
        """Context manager yielding the active mlflow run, tagged with
        environment/lineage metadata (python version, git commit, library versions).
        """
        with mlflow.start_run(run_name=run_name) as run:
            tags = {
                "python_version": platform.python_version(),
                "git_commit": _git_commit(),
            }
            for lib in _TRACKED_LIBRARIES:
                tags[f"lib_{lib}"] = _library_version(lib)
            mlflow.set_tags(tags)
            yield run

    def log_training_run(
        self,
        metrics: dict[str, float],
        params: dict[str, Any],
        feature_importance: dict[str, float] | None,
        training_seconds: float,
        dataset_version: "DatasetVersion",
    ) -> str:
        """Log params/metrics/feature-importance/training time for the active run.

        NOTE: we do not attempt mlflow.sklearn.log_model with full signature inference
        here -- Prophet/joblib-payload models aren't plain sklearn estimators, so a
        simple mlflow.log_artifact of the joblib file is used as the safe substitute.
        The joblib file via ModelRegistry remains the actual serving artifact; MLflow's
        model logging (if added later) would be lineage/UI only.
        """
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)
        if feature_importance:
            mlflow.log_dict(feature_importance, "feature_importance.json")
        mlflow.log_metric("training_seconds", training_seconds)
        mlflow.set_tags(
            {
                "dataset_version": dataset_version.content_hash,
                "dataset_source": dataset_version.source,
            }
        )
        return mlflow.active_run().info.run_id

    def auto_promote_if_better(
        self,
        model_name: str,
        new_version: int,
        new_metrics: dict[str, float],
        registry: "ModelRegistry",
        primary_metric: str,
        lower_is_better: bool,
    ) -> bool:
        """Always promotes new_version to STAGING (decision #2: every successful run
        auto-promotes to Staging). Returns whether new_metrics also beat the current
        PRODUCTION record on primary_metric -- informational only, does NOT promote
        to Production (that is gated via promote.py / M12's retrain trigger).
        """
        from src.mlops.registry.model_registry import ModelStage

        registry.promote(model_name, new_version, ModelStage.STAGING)

        try:
            production_records = [
                r for r in registry.list_versions(model_name) if r.stage == ModelStage.PRODUCTION
            ]
        except FileNotFoundError:
            production_records = []

        if not production_records:
            return True

        current_production = max(production_records, key=lambda r: r.version)
        current_value = current_production.metrics.get(primary_metric)
        new_value = new_metrics.get(primary_metric)
        if current_value is None or new_value is None:
            return True

        if lower_is_better:
            return new_value < current_value
        return new_value > current_value
