import datetime as dt

import pandas as pd

from src.config.settings import settings
from src.features.calendar_features import add_calendar_features
from src.features.preprocessing import apply_categorical_encoder, apply_scaler
from src.mlops.registry.model_registry import ModelRegistry
from src.pipelines.pricing_pipeline import (
    CATEGORICAL_COLS,
    FEATURE_COLS,
    NUMERIC_FEATURE_COLS,
)

_ROOM_TYPE_DIM_PATH = settings.data_raw_dir_path / "room_type_dim.csv"


def recommend_price(
    branch_id: int,
    room_type_id: int,
    date: str,
    current_occupancy_pct: float,
    current_revenue: float,
    revenue_7day_avg: float,
    total_rooms: int,
) -> dict:
    model = ModelRegistry().load_production("pricing_xgboost")

    room_types = pd.read_csv(_ROOM_TYPE_DIM_PATH)
    room_type = room_types[room_types["room_type_id"] == room_type_id].iloc[0]

    row = pd.DataFrame(
        [
            {
                "date": pd.to_datetime(date),
                "occupancy_pct": current_occupancy_pct,
                "occupancy_7day_avg": current_occupancy_pct,
                "total_revenue": current_revenue,
                "revenue_7day_avg": revenue_7day_avg,
                "room_type_name": room_type["room_type_name"],
                "base_price_multiplier": room_type["base_price_multiplier"],
            }
        ]
    )
    row = add_calendar_features(row, "date")
    revenue_ratio = row["total_revenue"] / row["revenue_7day_avg"]
    row["demand_index"] = row["occupancy_pct"] * revenue_ratio.fillna(1.0)

    row = apply_categorical_encoder(row, CATEGORICAL_COLS, model.encoders["categorical"])
    row = apply_scaler(row, NUMERIC_FEATURE_COLS, model.scaler)

    recommended_price = float(model.predict(row[FEATURE_COLS])[0])
    expected_revenue = recommended_price * current_occupancy_pct / 100.0 * total_rooms

    return {
        "branch_id": branch_id,
        "room_type_id": room_type_id,
        "date": date,
        "recommended_price": round(recommended_price, 2),
        "expected_revenue": round(expected_revenue, 2),
    }
