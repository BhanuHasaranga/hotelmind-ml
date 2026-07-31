from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    WAREHOUSE_DB_HOST: str = "localhost"
    WAREHOUSE_DB_PORT: int = 5433
    WAREHOUSE_DB_NAME: str = "hotelmind_warehouse"
    WAREHOUSE_DB_USER: str = "hotelmind_dw"
    WAREHOUSE_DB_PASSWORD: str = "hotelmind_dw_secret"

    CHURN_WINDOW_DAYS: int = 180
    FORECAST_HORIZON_DAYS: int = 30

    MODEL_DIR: str = "models"
    REPORTS_DIR: str = "reports"
    DATA_RAW_DIR: str = "data/raw"
    DATA_PROCESSED_DIR: str = "data/processed"
    DATA_WAREHOUSE_DIR: str = "data/warehouse"
    DATA_FEATURES_DIR: str = "data/features"
    SQL_DIR: str = "sql"

    LOG_LEVEL: str = "INFO"

    @property
    def WAREHOUSE_DB_URL(self) -> str:
        return (
            f"postgresql://{self.WAREHOUSE_DB_USER}:{self.WAREHOUSE_DB_PASSWORD}"
            f"@{self.WAREHOUSE_DB_HOST}:{self.WAREHOUSE_DB_PORT}/{self.WAREHOUSE_DB_NAME}"
        )

    @property
    def model_dir_path(self) -> Path:
        path = PROJECT_ROOT / self.MODEL_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def reports_dir_path(self) -> Path:
        path = PROJECT_ROOT / self.REPORTS_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def data_raw_dir_path(self) -> Path:
        return PROJECT_ROOT / self.DATA_RAW_DIR

    @property
    def data_processed_dir_path(self) -> Path:
        path = PROJECT_ROOT / self.DATA_PROCESSED_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def data_warehouse_dir_path(self) -> Path:
        path = PROJECT_ROOT / self.DATA_WAREHOUSE_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def data_features_dir_path(self) -> Path:
        path = PROJECT_ROOT / self.DATA_FEATURES_DIR
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def warehouse_loading_reports_dir_path(self) -> Path:
        path = self.reports_dir_path / "warehouse_loading"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def features_reports_dir_path(self) -> Path:
        path = self.reports_dir_path / "features"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def models_reports_dir_path(self) -> Path:
        path = self.reports_dir_path / "models"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def final_phase4_reports_dir_path(self) -> Path:
        path = self.reports_dir_path / "final_phase4"
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def sql_dir_path(self) -> Path:
        return PROJECT_ROOT / self.SQL_DIR


settings = Settings()
