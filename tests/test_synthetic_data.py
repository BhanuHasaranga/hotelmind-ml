import pandas as pd

from src.pipelines.synthetic_data import generate_restaurant_daily, generate_staffing_daily


def _daily_occupancy() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=14, freq="D")
    return pd.DataFrame({
        "branch_id": 1,
        "occupancy_date": dates,
        "occupied_rooms": [10 + i for i in range(14)],
    })


def test_generate_restaurant_daily_deterministic():
    daily = _daily_occupancy()
    r1 = generate_restaurant_daily(daily, seed=1)
    r2 = generate_restaurant_daily(daily, seed=1)
    pd.testing.assert_frame_equal(r1, r2)


def test_generate_restaurant_daily_different_seed_differs():
    daily = _daily_occupancy()
    r1 = generate_restaurant_daily(daily, seed=1)
    r2 = generate_restaurant_daily(daily, seed=2)
    assert not r1["total_revenue"].equals(r2["total_revenue"])


def test_generate_restaurant_daily_non_negative():
    daily = _daily_occupancy()
    r = generate_restaurant_daily(daily)
    assert (r["total_revenue"] >= 0).all()
    assert (r["breakfast_revenue"] >= 0).all()
    assert (r["items_sold"] >= 0).all()


def test_generate_restaurant_daily_columns_match_derive_meal_quantities():
    from src.features.restaurant_features import derive_meal_quantities

    daily = _daily_occupancy()
    r = generate_restaurant_daily(daily)
    out = derive_meal_quantities(r)
    assert "breakfast_qty" in out.columns
    assert "lunch_qty" in out.columns
    assert "dinner_qty" in out.columns


def test_generate_staffing_daily_deterministic():
    daily = _daily_occupancy()
    s1 = generate_staffing_daily(daily, seed=1)
    s2 = generate_staffing_daily(daily, seed=1)
    pd.testing.assert_frame_equal(s1, s2)


def test_generate_staffing_daily_non_negative_and_all_departments():
    daily = _daily_occupancy()
    s = generate_staffing_daily(daily)
    assert (s["present_employees"] >= 0).all()
    assert (s["scheduled_employees"] >= 0).all()
    assert set(s["department_name"].unique()) == {"Reception", "Kitchen", "Housekeeping"}
