"""Forces `local_reader._ensure_mart_snapshots` to actually regenerate (by
pointing at an isolated, empty warehouse dir seeded only with the upstream
dim_hotel/fact_booking tables it needs) so the synthesis code paths get
exercised, rather than short-circuiting on already-generated mart files."""

from __future__ import annotations

import pandas as pd
import pytest

from genai.data_access import local_reader
from src.config.settings import settings


@pytest.fixture
def empty_warehouse_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "DATA_WAREHOUSE_DIR", str(tmp_path))
    dim_hotel = pd.DataFrame({"hotel_key": [10, 20, 30]})
    fact_booking = pd.DataFrame({"guest_key": [1, 2], "nights": [2, 3]})
    dim_hotel.to_parquet(tmp_path / "dim_hotel.parquet", index=False)
    fact_booking.to_parquet(tmp_path / "fact_booking.parquet", index=False)
    return tmp_path


def test_ensure_mart_snapshots_generates_all_files(empty_warehouse_dir):
    local_reader._ensure_mart_snapshots()
    for filename in local_reader.MART_FILES.values():
        assert (empty_warehouse_dir / filename).exists()


def test_synthesize_mart_occupancy_daily_shape(empty_warehouse_dir):
    df = local_reader._synthesize_mart_occupancy_daily()
    assert {"branch_id", "date", "occupancy_pct", "total_rooms", "occupied_rooms"}.issubset(df.columns)
    assert df["occupancy_pct"].between(0, 100).all()


def test_synthesize_mart_revenue_daily_derived_from_occupancy(empty_warehouse_dir):
    occ = local_reader._synthesize_mart_occupancy_daily()
    revenue = local_reader._synthesize_mart_revenue_daily(occ)
    assert (revenue["total_revenue"] >= 0).all()
    assert "revpar" in revenue.columns


def test_synthesize_mart_restaurant_daily_derived_from_occupancy(empty_warehouse_dir):
    occ = local_reader._synthesize_mart_occupancy_daily()
    restaurant = local_reader._synthesize_mart_restaurant_daily(occ)
    assert (restaurant["total_orders"] >= 0).all()


def test_synthesize_mart_staff_daily_derived_from_occupancy(empty_warehouse_dir):
    occ = local_reader._synthesize_mart_occupancy_daily()
    staff = local_reader._synthesize_mart_staff_daily(occ)
    assert set(staff["department_name"].unique()) == {"Reception", "Kitchen", "Housekeeping"}
    assert (staff["scheduled_employees"] >= 0).all()


def test_read_mart_triggers_generation_once(empty_warehouse_dir):
    df1 = local_reader.read_mart("mart_occupancy_daily")
    assert not df1.empty
    # Second read should hit the already-generated file, not regenerate.
    df2 = local_reader.read_mart("mart_occupancy_daily")
    pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))
