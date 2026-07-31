import datetime as dt

import pandas as pd

from src.config.settings import settings
from src.models.occupancy.prophet_model import OccupancyProphetModel
from src.models.occupancy.xgboost_model import OccupancyXGBoostModel


def forecast_occupancy(
    branch_id: int, horizon_days: int | None = None
) -> pd.DataFrame:
    """Forecast the next `horizon_days` of occupancy_pct for a branch.

    Prophet is the primary model (native confidence interval); XGBoost's
    point forecast is included as `model_used` comparison metadata only
    when Prophet's forecast is unavailable.
    """
    horizon_days = horizon_days or settings.FORECAST_HORIZON_DAYS

    prophet_model = OccupancyProphetModel()
    prophet_path = settings.model_dir_path / "occupancy_prophet.pkl"
    prophet_model.load(prophet_path)

    future_dates = pd.DataFrame(
        {
            "occupancy_date": pd.date_range(
                start=dt.date.today(), periods=horizon_days, freq="D"
            )
        }
    )
    forecast = prophet_model.predict_with_interval(future_dates)
    forecast["branch_id"] = branch_id
    forecast["model_used"] = "prophet"
    return forecast[
        ["branch_id", "occupancy_date", "predicted_occupancy_pct", "ci_lower", "ci_upper", "model_used"]
    ]
