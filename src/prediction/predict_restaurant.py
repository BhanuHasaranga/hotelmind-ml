import pandas as pd

from src.config.constants import MEAL_PERIODS
from src.config.settings import settings
from src.features.calendar_features import add_calendar_features
from src.models.restaurant.xgboost_model import RestaurantDemandModel
from src.pipelines.restaurant_pipeline import FEATURE_COLS


def forecast_restaurant_demand(
    branch_id: int,
    date: str,
    recent_total_orders_lag_1: float,
    recent_total_orders_lag_7: float,
    recent_total_orders_rolling_mean_7: float,
    avg_item_value: float,
) -> dict:
    row = pd.DataFrame(
        [
            {
                "date": pd.to_datetime(date),
                "total_orders_lag_1": recent_total_orders_lag_1,
                "total_orders_lag_7": recent_total_orders_lag_7,
                "total_orders_rolling_mean_7": recent_total_orders_rolling_mean_7,
            }
        ]
    )
    row = add_calendar_features(row, "date")

    result: dict = {"branch_id": branch_id, "date": date}
    for meal in MEAL_PERIODS:
        model = RestaurantDemandModel()
        model.load(settings.model_dir_path / f"restaurant_{meal}.pkl")
        predicted_revenue = float(model.predict(row[FEATURE_COLS])[0])
        predicted_qty = predicted_revenue / avg_item_value if avg_item_value else 0.0
        result[meal] = {
            "expected_quantity": round(predicted_qty, 1),
            "expected_revenue": round(predicted_revenue, 2),
        }
    return result
