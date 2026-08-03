"""Calls the existing `src/prediction/predict_*.py` functions directly,
in-process, and renders their outputs into indexable text documents so the
RAG assistant can answer "why" questions about ML predictions without an
HTTP round-trip.
"""

import pandas as pd

from src.config.settings import settings
from src.prediction.predict_occupancy import forecast_occupancy
from src.prediction.predict_pricing import recommend_price
from src.prediction.predict_restaurant import forecast_restaurant_demand
from src.prediction.predict_staffing import recommend_staffing
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _default_branch_ids(limit: int = 3) -> list[int]:
    try:
        dim_hotel = pd.read_parquet(settings.data_warehouse_dir_path / "dim_hotel.parquet")
        return sorted(dim_hotel["hotel_key"].unique())[:limit] or [1]
    except Exception:
        return [1]


def load_occupancy_predictions(branch_ids: list[int] | None = None) -> list[dict]:
    docs = []
    for branch_id in branch_ids or _default_branch_ids():
        try:
            forecast_df = forecast_occupancy(branch_id)
        except Exception as exc:
            logger.warning("Occupancy prediction unavailable for branch %s: %s", branch_id, exc)
            continue
        for _, row in forecast_df.iterrows():
            text = (
                f"Occupancy forecast for branch {branch_id} on {row['occupancy_date']}: "
                f"{row['predicted_occupancy_pct']}% (range {row['ci_lower']}-{row['ci_upper']}%, "
                f"model {row['model_used']})."
            )
            docs.append(
                {
                    "text": text,
                    "metadata": {"source": "predict_occupancy", "doc_type": "prediction", "branch_id": branch_id},
                }
            )
    return docs


def load_all_prediction_documents(branch_ids: list[int] | None = None) -> list[dict]:
    """Aggregates occupancy prediction snapshots for RAG indexing. Pricing,
    restaurant, and staffing predictions require additional per-request
    inputs (room type, current occupancy, department, etc.) that aren't
    available ambiently, so only occupancy (which forecasts autonomously
    per branch) is auto-indexed here; the others remain queryable live via
    their own `/predict/*` endpoints, and richer indexing can be added by
    passing explicit inputs through `prediction_loader` directly."""
    return load_occupancy_predictions(branch_ids)
