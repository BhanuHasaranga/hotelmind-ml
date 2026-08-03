from typing import Any

from src.mlops.pipelines.mlops_pipeline_mixin import MLOpsPipelineMixin
from src.mlops.registry.model_registry import ModelRegistry
from src.mlops.tracking.mlflow_tracker import MLflowTracker
from src.mlops.validation.dataset_version import (
    compute_dataset_version,
    has_dataset_changed,
    save_dataset_version,
)
from src.pipelines.base_pipeline import default_time_series_split
from src.pipelines.restaurant_pipeline import FEATURE_COLS, RestaurantPipeline


class RestaurantMLOpsPipeline(MLOpsPipelineMixin, RestaurantPipeline):
    """RestaurantPipeline overrides run() with a per-meal loop (breakfast/lunch/
    dinner each has its own target column), so run_with_mlops() mirrors that loop
    directly rather than using the mixin's generic single-target flow.
    """

    mlflow_experiment_name = "restaurant"
    registry_model_names = ["restaurant_breakfast", "restaurant_lunch", "restaurant_dinner"]

    def run_with_mlops(self, **load_kwargs) -> dict[str, Any]:
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

        models = self.build_models()
        metrics_by_model: dict[str, dict[str, float]] = {}

        registry = ModelRegistry()
        tracker = MLflowTracker(self.mlflow_experiment_name)
        summary: dict[str, Any] = {"dataset_changed": changed, "registered": {}}

        for meal, model in models.items():
            target_col = f"{meal}_revenue"
            X_train, X_test, y_train, y_test = default_time_series_split(
                df, target_col, FEATURE_COLS, test_size=0.2
            )
            model.feature_names = FEATURE_COLS

            import time

            start = time.perf_counter()
            model.train(X_train, y_train)
            training_seconds = time.perf_counter() - start

            metrics = model.evaluate(X_test, y_test)
            metrics_by_model[meal] = metrics
            model.save(self.model_save_path(meal))
            self.logger.info("Trained+saved meal=%s metrics=%s", meal, metrics)

            model_name = f"restaurant_{meal}"
            with tracker.start_run(run_name=model_name):
                run_id = tracker.log_training_run(
                    metrics=metrics,
                    params={"model_key": meal, "module": self.module_name},
                    feature_importance=None,
                    training_seconds=training_seconds,
                    dataset_version=dataset_version,
                )

            record = registry.register_model(
                model_name=model_name,
                source_path=self.model_save_path(meal),
                metrics=metrics,
                dataset_version=dataset_version.content_hash,
                mlflow_run_id=run_id,
            )
            beats_production = tracker.auto_promote_if_better(
                model_name=model_name,
                new_version=record.version,
                new_metrics=metrics,
                registry=registry,
                primary_metric="mae",
                lower_is_better=True,
            )
            summary["registered"][model_name] = {
                "version": record.version,
                "metrics": metrics,
                "beats_production": beats_production,
            }

        from src.evaluation.report_writer import write_report

        write_report(self.module_name, metrics_by_model)
        # TODO(M11): write evaluation dashboard record here

        return summary
