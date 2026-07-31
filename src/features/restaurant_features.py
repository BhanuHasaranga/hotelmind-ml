import pandas as pd

from src.config.constants import MEAL_PERIODS
from src.utils.logging import get_logger

logger = get_logger(__name__)


def derive_meal_quantities(df: pd.DataFrame) -> pd.DataFrame:
    """Approximate per-meal quantity sold.

    ASSUMPTION: mart_restaurant_daily has day-level `items_sold` and
    per-meal revenue (breakfast/lunch/dinner_revenue) but no ground-truth
    per-meal item count. We approximate:

        qty_meal = items_sold * (meal_revenue / total_revenue)

    with a cross-check qty_meal_check = meal_revenue / avg_item_value,
    logged when the two estimates diverge by more than 20%. This is a
    revenue-share approximation, not a measured quantity — see README
    "Assumptions & Data Gaps".
    """
    out = df.copy()
    for meal in MEAL_PERIODS:
        revenue_col = f"{meal}_revenue"
        qty_col = f"{meal}_qty"
        check_col = f"{meal}_qty_check"

        revenue_share = (out[revenue_col] / out["total_revenue"]).fillna(0.0)
        out[qty_col] = out["items_sold"] * revenue_share
        out[check_col] = (out[revenue_col] / out["avg_item_value"]).fillna(0.0)

        divergence = (
            (out[qty_col] - out[check_col]).abs() / out[qty_col].replace(0, pd.NA)
        ).fillna(0.0)
        n_divergent = int((divergence > 0.20).sum())
        if n_divergent:
            logger.info(
                "%s: %d/%d rows diverge >20%% between revenue-share and "
                "avg-item-value quantity estimates",
                meal, n_divergent, len(out),
            )
    return out
