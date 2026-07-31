import pandas as pd

from src.config.settings import settings

_ROOM_TYPE_DIM_PATH = settings.data_warehouse_dir_path / "dim_room_type.parquet"


def add_room_types(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-join each branch/date row with the warehouse room-type dimension.

    Sources room_type_name/base_price_multiplier from
    data/warehouse/dim_room_type.parquet (the single source of truth built
    in the Phase 3 warehouse-loading milestone) rather than the raw
    data/raw/room_type_dim.csv seed it was originally derived from, since
    the warehouse dim additionally carries a documented letter-code mapping
    (source_code_examples) and an explicit "Unmapped" bucket.
    """
    room_types = pd.read_parquet(_ROOM_TYPE_DIM_PATH)[["room_type_name", "base_price_multiplier"]]
    df = df.copy()
    df["_key"] = 1
    room_types = room_types.copy()
    room_types["_key"] = 1
    out = df.merge(room_types, on="_key").drop(columns=["_key"])
    return out


def add_demand_index(df: pd.DataFrame) -> pd.DataFrame:
    """Derive a demand_index proxy from occupancy_pct and revenue_7day_avg trend.

    ASSUMPTION: the warehouse has no direct "demand" measure. demand_index is
    defined as occupancy_pct weighted by the ratio of current revenue to its
    7-day trailing average, i.e. demand_index = occupancy_pct * (revenue /
    revenue_7day_avg) when the trailing average is available, else
    occupancy_pct alone. This approximates "demand is high when both
    occupancy and revenue momentum are elevated." See README "Assumptions &
    Data Gaps".
    """
    out = df.copy()
    revenue_ratio = (out["total_revenue"] / out["revenue_7day_avg"]).replace(
        [float("inf"), -float("inf")], 1.0
    )
    revenue_ratio = revenue_ratio.fillna(1.0)
    out["demand_index"] = out["occupancy_pct"] * revenue_ratio
    return out
