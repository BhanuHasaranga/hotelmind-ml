import pandas as pd

from src.pipelines import keygen, warehouse_loader as wl


def _clean_df(n: int = 4) -> pd.DataFrame:
    base = {
        "hotel": "Resort Hotel", "is_canceled": 0, "lead_time": 10,
        "arrival_date_year": 2016, "arrival_date_month": "July",
        "arrival_date_week_number": 27, "arrival_date_day_of_month": 1,
        "stays_in_weekend_nights": 1, "stays_in_week_nights": 2,
        "adults": 2, "children": 0, "babies": 0,
        "meal": "BB", "country": "PRT", "market_segment": "Direct",
        "distribution_channel": "Direct", "is_repeated_guest": 0,
        "previous_cancellations": 0, "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A", "assigned_room_type": "A",
        "booking_changes": 0, "deposit_type": "No Deposit", "agent": None,
        "company": None, "days_in_waiting_list": 0, "customer_type": "Transient",
        "adr": 100.0, "required_car_parking_spaces": 0,
        "total_of_special_requests": 0, "reservation_status": "Check-Out",
        "reservation_status_date": pd.Timestamp("2016-07-03"),
        "is_adults_outlier": False,
    }
    rows = []
    for i in range(n):
        row = dict(base)
        row["lead_time"] = 10 + i
        row["arrival_date_day_of_month"] = 1 + i
        row["assigned_room_type"] = ["A", "D", "H", "Z"][i % 4]
        rows.append(row)
    df = pd.DataFrame(rows)
    df["arrival_date"] = pd.to_datetime(dict(
        year=df["arrival_date_year"], month=7, day=df["arrival_date_day_of_month"]
    ))
    return df


def test_build_dim_date_distinct_dates_and_date_key_format():
    df = _clean_df()
    dim_date = wl.build_dim_date(df)
    assert dim_date["date_key"].iloc[0] == int(dim_date["full_date"].iloc[0].strftime("%Y%m%d"))
    assert dim_date["date_key"].is_unique


def test_build_dim_hotel_exactly_two_rows():
    df = _clean_df()
    df.loc[0, "hotel"] = "City Hotel"
    dim_hotel = wl.build_dim_hotel(df)
    assert len(dim_hotel) == 2


def test_build_dim_room_type_includes_unmapped_bucket_when_needed():
    df = _clean_df()
    dim_room_type = wl.build_dim_room_type(df)
    assert keygen.UNMAPPED_ROOM_TYPE_ID in dim_room_type["room_type_key"].tolist()


def test_build_dim_guest_clusters_by_profile_and_aggregates_lifetime_bookings():
    df = _clean_df(n=3)
    dim_guest = wl.build_dim_guest(df)
    assert len(dim_guest) == 1  # identical profile fields across rows
    assert dim_guest.loc[0, "lifetime_bookings"] == 3


def test_build_fact_booking_computes_nights_and_total_amount():
    df = _clean_df(n=2)
    fact, stats = wl.build_fact_booking(df)
    assert (fact["nights"] == 3).all()
    assert (fact["total_amount"] == fact["nights"] * fact["avg_daily_rate"]).all()


def test_build_fact_booking_excludes_rows_with_unresolvable_date():
    df = _clean_df(n=2)
    df.loc[0, "arrival_date"] = pd.NaT
    fact, stats = wl.build_fact_booking(df)
    assert stats["excluded_unresolvable_date_fk"] == 1
    assert len(fact) == 1


def test_build_fact_booking_is_terminal_and_is_completed_flags():
    df = _clean_df(n=1)
    df.loc[0, "reservation_status"] = "Canceled"
    fact, _ = wl.build_fact_booking(df)
    assert fact.loc[0, "is_terminal"] == True
    assert fact.loc[0, "is_completed"] == False


def _consistent_tables():
    df = _clean_df(n=3)
    tables, _ = wl.build_all(df)
    return tables


def test_validate_warehouse_passes_on_consistent_tables():
    tables = _consistent_tables()
    results = wl.validate_warehouse(tables)
    assert all(t["pk_unique"] for t in results["tables"].values())
    assert all(fk["orphan_count"] == 0 for fk in results["fk_checks"].values())


def test_validate_warehouse_detects_orphan_fk():
    tables = _consistent_tables()
    tables["fact_booking"].loc[0, "guest_key"] = 999999999
    results = wl.validate_warehouse(tables)
    orphan = results["fk_checks"]["fact_booking.guest_key -> dim_guest.guest_key"]
    assert orphan["orphan_count"] == 1
    assert 999999999 in orphan["sample_orphans"]


def test_validate_warehouse_detects_duplicate_pk():
    tables = _consistent_tables()
    tables["dim_hotel"] = pd.concat([tables["dim_hotel"], tables["dim_hotel"].iloc[[0]]], ignore_index=True)
    results = wl.validate_warehouse(tables)
    assert results["tables"]["dim_hotel"]["pk_unique"] is False
