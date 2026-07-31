import pandas as pd

from src.features.occupancy_aggregation import build_daily_occupancy, expand_booking_nights


def _dim_date(n_days: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    return pd.DataFrame({
        "date_key": [int(d.strftime("%Y%m%d")) for d in dates],
        "full_date": dates,
    })


def _dim_hotel() -> pd.DataFrame:
    return pd.DataFrame({
        "hotel_key": [1, 2],
        "hotel_id": ["RESORT", "CITY"],
        "hotel_name": ["Resort Hotel", "City Hotel"],
    })


def _fact_booking(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)


def test_expand_booking_nights_excludes_cancelled():
    fb = _fact_booking([
        {"branch_key": 1, "check_in_date_key": 20240101, "nights": 2, "avg_daily_rate": 100.0, "reservation_status": "Check-Out"},
        {"branch_key": 1, "check_in_date_key": 20240101, "nights": 2, "avg_daily_rate": 100.0, "reservation_status": "Canceled"},
        {"branch_key": 1, "check_in_date_key": 20240101, "nights": 2, "avg_daily_rate": 100.0, "reservation_status": "No-Show"},
    ])
    expanded = expand_booking_nights(fb, _dim_date())
    assert len(expanded) == 2  # only the Check-Out booking contributes 2 nights


def test_expand_booking_nights_zero_nights_contributes_nothing():
    fb = _fact_booking([
        {"branch_key": 1, "check_in_date_key": 20240101, "nights": 0, "avg_daily_rate": 100.0, "reservation_status": "Check-Out"},
    ])
    expanded = expand_booking_nights(fb, _dim_date())
    assert expanded.empty


def test_expand_booking_nights_date_range_correct():
    fb = _fact_booking([
        {"branch_key": 1, "check_in_date_key": 20240101, "nights": 3, "avg_daily_rate": 100.0, "reservation_status": "Check-Out"},
    ])
    expanded = expand_booking_nights(fb, _dim_date())
    expected_dates = pd.date_range("2024-01-01", periods=3, freq="D")
    assert sorted(expanded["full_date"].tolist()) == sorted(expected_dates.tolist())


def test_build_daily_occupancy_pct_bounds():
    fb = _fact_booking([
        {"branch_key": 1, "check_in_date_key": 20240101, "nights": 1, "avg_daily_rate": 100.0, "reservation_status": "Check-Out"},
    ])
    daily = build_daily_occupancy(fb, _dim_date(), _dim_hotel())
    assert (daily["occupancy_pct"] >= 0).all()
    assert (daily["occupancy_pct"] <= 100).all()
    assert daily.loc[0, "occupied_rooms"] == 1
    assert daily.loc[0, "branch_name"] == "Resort Hotel"


def test_build_daily_occupancy_aggregates_multiple_bookings_same_day():
    fb = _fact_booking([
        {"branch_key": 1, "check_in_date_key": 20240101, "nights": 1, "avg_daily_rate": 100.0, "reservation_status": "Check-Out"},
        {"branch_key": 1, "check_in_date_key": 20240101, "nights": 1, "avg_daily_rate": 150.0, "reservation_status": "Check-Out"},
    ])
    daily = build_daily_occupancy(fb, _dim_date(), _dim_hotel())
    assert daily.loc[0, "occupied_rooms"] == 2
    assert daily.loc[0, "total_revenue"] == 250.0
