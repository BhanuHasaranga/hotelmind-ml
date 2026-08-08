import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.evaluation.report_writer import write_report
from src.mlops.metrics.evaluation_dashboard import ModelEvalRecord, write_dashboard
from src.mlops.registry.model_registry import ModelRegistry
from src.mlops.tracking.mlflow_tracker import MLflowTracker
from src.mlops.validation.dataset_version import (
    compute_dataset_version,
    has_dataset_changed,
    save_dataset_version,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)

# model_name -> (primary_metric, lower_is_better), used for auto_promote_if_better comparisons
_METRIC_DIRECTION_BY_KEY: dict[str, tuple[str, bool]] = {
    "prophet": ("mae", True),
    "xgboost": ("mae", True),
    "regression": ("mae", True),
    "random_forest": ("roc_auc", False),
    "breakfast": ("mae", True),
    "lunch": ("mae", True),
    "dinner": ("mae", True),
}


class MLOpsPipelineMixin:
    """Mix in with an existing BasePipeline subclass via multiple inheritance to add
    MLflow tracking, dataset versioning, and registry-based model promotion around the
    unchanged base pipeline steps: class Foo(MLOpsPipelineMixin, FooPipeline): ...
    """

    mlflow_experiment_name: str  # subclasses set this
    registry_model_names: list[str]  # subclasses set this, e.g. ["occupancy_prophet", "occupancy_xgboost"]

    module_name: str
    logger: Any

    # These are declarations for type checkers only. The mixin precedes the concrete
    # pipeline in the MRO, so defining them at runtime would shadow the real
    # BasePipeline implementations and silently return None from every step.
    if TYPE_CHECKING:
        def load_data(self, **kwargs) -> Any: ...
        def clean(self, df): ...
        def engineer_features(self, df): ...
        def split(self, df): ...
        def build_models(self) -> dict: ...
        def train(self, models, X_train, y_train) -> dict: ...
        def evaluate(self, models, X_test, y_test) -> dict: ...
        def model_save_path(self, model_key: str) -> Path: ...
        def save_models(self, models: dict) -> None: ...

    def _registry_name_for_key(self, model_key: str) -> str:
        """Map a pipeline's internal model_key (e.g. 'xgboost', 'prophet', 'breakfast')
        to its registry model_name (e.g. 'occupancy_xgboost', 'restaurant_breakfast').
        """
        for name in self.registry_model_names:
            if name.endswith(f"_{model_key}") or name == model_key:
                return name
        # Fallback: single-model pipelines (e.g. pricing_xgboost, staffing_regression)
        # where registry_model_names has exactly one entry.
        if len(self.registry_model_names) == 1:
            return self.registry_model_names[0]
        raise ValueError(
            f"Could not map model_key={model_key!r} to a registry name in {self.registry_model_names}"
        )

    def run_with_mlops(self, **load_kwargs) -> dict[str, Any]:
        """Parallel to BasePipeline.run(): reuses the existing load/clean/feature/
        split/train/evaluate/save steps unchanged, adding dataset versioning, MLflow
        tracking, and registry registration/auto-promotion around them.
        """
        df = self.load_data(**load_kwargs)

        dataset_version = compute_dataset_version(df, source=self.mlflow_experiment_name)
        changed = has_dataset_changed(self.mlflow_experiment_name, dataset_version)
        save_dataset_version(dataset_version, self.mlflow_experiment_name)
        self.logger.info(
            "Dataset version for %s: hash=%s changed=%s",
            self.mlflow_experiment_name,
            dataset_version.content_hash,
            changed,
        )

        df = self.clean(df)
        df = self.engineer_features(df)
        X_train, X_test, y_train, y_test = self.split(df)

        models = self.build_models()

        start = time.perf_counter()
        models = self.train(models, X_train, y_train)
        training_seconds = time.perf_counter() - start

        metrics_by_model = self.evaluate(models, X_test, y_test)
        self.save_models(models)

        registry = ModelRegistry()
        tracker = MLflowTracker(self.mlflow_experiment_name)
        summary: dict[str, Any] = {"dataset_changed": changed, "registered": {}}
        eval_records: list[ModelEvalRecord] = []

        for model_key, model in models.items():
            model_name = self._registry_name_for_key(model_key)
            metrics = metrics_by_model[model_key]

            with tracker.start_run(run_name=f"{model_name}"):
                run_id = tracker.log_training_run(
                    metrics=metrics,
                    params={"model_key": model_key, "module": self.module_name},
                    feature_importance=None,
                    training_seconds=training_seconds,
                    dataset_version=dataset_version,
                )

            record = registry.register_model(
                model_name=model_name,
                source_path=self.model_save_path(model_key),
                metrics=metrics,
                dataset_version=dataset_version.content_hash,
                mlflow_run_id=run_id,
            )

            primary_metric, lower_is_better = _METRIC_DIRECTION_BY_KEY.get(model_key, ("mae", True))
            beats_production = tracker.auto_promote_if_better(
                model_name=model_name,
                new_version=record.version,
                new_metrics=metrics,
                registry=registry,
                primary_metric=primary_metric,
                lower_is_better=lower_is_better,
            )
            summary["registered"][model_name] = {
                "version": record.version,
                "metrics": metrics,
                "beats_production": beats_production,
            }

            eval_records.append(
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
                    training_duration_s=training_seconds,
                )
            )

        write_report(self.module_name, metrics_by_model)
        write_dashboard(eval_records)

        return summary
