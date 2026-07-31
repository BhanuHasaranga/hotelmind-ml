"""Generates reports/models/occupancy_metrics.json and
reports/models/occupancy_forecast.csv (30-day forecast with confidence
interval), using the already-trained Prophet model via
src.prediction.predict_occupancy.forecast_occupancy (which anchors the
forecast start date to the model's own training range -- see that module's
docstring).

CLI: python -m src.pipelines.generate_occupancy_report
"""

import json

from src.config.settings import settings
from src.prediction.predict_occupancy import forecast_occupancy
from src.utils.logging import get_logger

logger = get_logger(__name__)


def run(branch_id: int = 1) -> None:
    latest_path = settings.reports_dir_path / "latest_occupancy.json"
    payload = json.loads(latest_path.read_text(encoding="utf-8"))

    metrics_path = settings.models_reports_dir_path / "occupancy_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Wrote %s", metrics_path)

    forecast = forecast_occupancy(branch_id=branch_id, horizon_days=settings.FORECAST_HORIZON_DAYS)
    forecast_path = settings.models_reports_dir_path / "occupancy_forecast.csv"
    forecast.to_csv(forecast_path, index=False)
    logger.info("Wrote %s (%d rows)", forecast_path, len(forecast))


if __name__ == "__main__":
    run()
