import pandas as pd
import pytest

from src.pipelines import feature_engineering as fe


@pytest.fixture
def warehouse(tmp_path, monkeypatch):
    """Builds a tiny, self-consistent warehouse parquet set under tmp_path
    and points settings.data_warehouse_dir_path / data_raw_dir_path at it,
    so feature_engineering's builders can run fully offline.
    """
    from src.config.settings import settings

    warehouse_dir = tmp_path / "warehouse"
    warehouse_dir.mkdir()
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()

    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    dim_date = pd.DataFrame({
        "date_key": [int(d.strftime("%Y%m%d")) for d in dates],
        "full_date": dates,
    })
    dim_hotel = pd.DataFrame({
        "hotel_key": [1, 2],
        "hotel_id": ["RESORT", "CITY"],
        "hotel_name": ["Resort Hotel", "City Hotel"],
    })
    dim_room_type = pd.DataFrame({
        "room_type_key": [1, 2, 3],
        "room_type_id": [1, 2, 3],
        "room_type_name": ["Standard", "Deluxe", "Suite"],
        "base_price_multiplier": [1.0, 1.4, 2.0],
    })
    dim_guest = pd.DataFrame({
        "guest_key": [1001, 1002],
        "guest_id": [1001, 1002],
        "full_name": [None, None],
        "nationality": ["PRT", "USA"],
        "lifetime_bookings": [3, 1],
        "lifetime_spend": [900.0, 100.0],
        "first_stay_date": pd.to_datetime(["2024-01-01", "2024-01-05"]),
        "last_stay_date": pd.to_datetime(["2024-01-08", "2024-01-05"]),
    })
    fact_booking = pd.DataFrame({
        "surrogate_key": [1, 2, 3],
        "reservation_id": [1, 2, 3],
        "reservation_status": ["Check-Out", "Check-Out", "Canceled"],
        "room_key": [1, 2, 1],
        "branch_key": [1, 1, 2],
        "guest_key": [1001, 1001, 1002],
        "check_in_date_key": [20240101, 20240103, 20240105],
        "check_out_date_key": [20240103, 20240105, 20240107],
        "nights": [2, 2, 2],
        "adults": [2, 1, 2],
        "children": [0, 0, 0],
        "total_amount": [200.0, 300.0, 0.0],
        "avg_daily_rate": [100.0, 150.0, 0.0],
        "is_terminal": [True, True, True],
        "is_completed": [True, True, False],
        "is_adults_outlier": [False, False, False],
    })

    dim_date.to_parquet(warehouse_dir / "dim_date.parquet", index=False)
    dim_hotel.to_parquet(warehouse_dir / "dim_hotel.parquet", index=False)
    dim_room_type.to_parquet(warehouse_dir / "dim_room_type.parquet", index=False)
    dim_guest.to_parquet(warehouse_dir / "dim_guest.parquet", index=False)
    fact_booking.to_parquet(warehouse_dir / "fact_booking.parquet", index=False)

    monkeypatch.setattr(type(settings), "data_warehouse_dir_path", property(lambda self: warehouse_dir))
    monkeypatch.setattr(type(settings), "data_raw_dir_path", property(lambda self: raw_dir))
    monkeypatch.setattr(type(settings), "data_features_dir_path", property(lambda self: tmp_path / "features"))
    (tmp_path / "features").mkdir(exist_ok=True)

    return settings


def test_build_occupancy_features_nonempty_and_has_expected_cols(warehouse):
    df = fe.build_occupancy_features()
    assert not df.empty
    for col in ["occupancy_pct", "occupancy_pct_lag_1", "occupancy_pct_rolling_mean_7", "occupancy_trend"]:
        assert col in df.columns


def test_build_pricing_features_nonempty_and_has_room_type(warehouse):
    df = fe.build_pricing_features()
    assert not df.empty
    assert "demand_index" in df.columns
    assert "room_type_name" in df.columns


def test_build_restaurant_features_nonempty(warehouse):
    df = fe.build_restaurant_features()
    assert not df.empty
    assert "breakfast_qty" in df.columns


def test_build_staff_features_nonempty(warehouse):
    df = fe.build_staff_features()
    assert not df.empty
    assert "present_employees_lag_1" in df.columns


def test_build_churn_features_nonempty_and_has_label(warehouse):
    df = fe.build_churn_features()
    assert not df.empty
    assert "churn" in df.columns
    assert df["churn"].isin([0, 1]).all()
