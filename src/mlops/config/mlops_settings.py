from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class MLOpsSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    MLFLOW_TRACKING_URI: str = "http://localhost:5500"
    MLFLOW_BACKEND_STORE_URI: str = "postgresql://mlflow:mlflow@localhost:5544/mlflow"
    MLFLOW_ARTIFACT_ROOT: str = "artifacts"

    MODEL_REGISTRY_DIR: str = "models/registry"
    MODEL_STAGING_DIR: str = "models/staging"
    MODEL_PRODUCTION_DIR: str = "models/production"
    MODEL_ARCHIVED_DIR: str = "models/archived"

    DRIFT_REPORTS_DIR: str = "reports/drift"
    DATASET_VERSIONS_DIR: str = "reports/dataset_versions"
    LOGS_DIR: str = "logs"

    RETRAIN_ACCURACY_DROP_THRESHOLD: float = 0.10
    RETRAIN_DRIFT_SCORE_THRESHOLD: float = 0.30

    MLOPS_JSON_LOGGING: bool = False

    @property
    def mlflow_artifact_root_path(self) -> Path:
        path = PROJECT_ROOT / self.MLFLOW_ARTIFACT_ROOT
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def model_registry_dir_path(self) -> Path:
        path = PROJECT_ROOT / self.MODEL_REGISTRY_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def model_staging_dir_path(self) -> Path:
        path = PROJECT_ROOT / self.MODEL_STAGING_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def model_production_dir_path(self) -> Path:
        path = PROJECT_ROOT / self.MODEL_PRODUCTION_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def model_archived_dir_path(self) -> Path:
        path = PROJECT_ROOT / self.MODEL_ARCHIVED_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def drift_reports_dir_path(self) -> Path:
        path = PROJECT_ROOT / self.DRIFT_REPORTS_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def dataset_versions_dir_path(self) -> Path:
        path = PROJECT_ROOT / self.DATASET_VERSIONS_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def logs_dir_path(self) -> Path:
        path = PROJECT_ROOT / self.LOGS_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path


mlops_settings = MLOpsSettings()
