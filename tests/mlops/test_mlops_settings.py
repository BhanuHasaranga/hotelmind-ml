"""Exercises every *_path property on MLOpsSettings (each just does
PROJECT_ROOT / X with mkdir(exist_ok=True)) -- pure smoke/coverage test, no
mocking needed since these all resolve to real repo-relative directories that
already exist."""

from src.mlops.config.mlops_settings import mlops_settings


def test_all_dir_path_properties_return_existing_paths():
    for attr in (
        "mlflow_artifact_root_path",
        "model_registry_dir_path",
        "model_staging_dir_path",
        "model_production_dir_path",
        "model_archived_dir_path",
        "drift_reports_dir_path",
        "dataset_versions_dir_path",
        "logs_dir_path",
    ):
        path = getattr(mlops_settings, attr)
        assert path.exists()
        assert path.is_dir()


def test_default_settings_values():
    assert mlops_settings.MLOPS_JSON_LOGGING is False
    assert mlops_settings.RETRAIN_ACCURACY_DROP_THRESHOLD == 0.10
    assert mlops_settings.RETRAIN_DRIFT_SCORE_THRESHOLD == 0.30
