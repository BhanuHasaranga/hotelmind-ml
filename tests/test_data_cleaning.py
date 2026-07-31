import pandas as pd
import pytest

from src.pipelines import data_cleaning as dc


def _sample_row(**overrides) -> dict:
    row = {
        "hotel": "Resort Hotel", "is_canceled": 0, "lead_time": 10,
        "arrival_date_year": 2016, "arrival_date_month": "July",
        "arrival_date_week_number": 27, "arrival_date_day_of_month": 1,
        "stays_in_weekend_nights": 1, "stays_in_week_nights": 2,
        "adults": 2, "children": 0.0, "babies": 0,
        "meal": "BB", "country": "PRT", "market_segment": "Direct",
        "distribution_channel": "Direct", "is_repeated_guest": 0,
        "previous_cancellations": 0, "previous_bookings_not_canceled": 0,
        "reserved_room_type": "A", "assigned_room_type": "A",
        "booking_changes": 0, "deposit_type": "No Deposit", "agent": None,
        "company": None, "days_in_waiting_list": 0, "customer_type": "Transient",
        "adr": 100.0, "required_car_parking_spaces": 0,
        "total_of_special_requests": 0, "reservation_status": "Check-Out",
        "reservation_status_date": "2016-07-03",
    }
    row.update(overrides)
    return row


def _sample_df(rows: list[dict]) -> pd.DataFrame:
    return pd.DataFrame(rows)[dc.EXPECTED_COLUMNS]


def test_dedupe_full_row_drops_exact_duplicates():
    df = _sample_df([_sample_row(), _sample_row(), _sample_row(lead_time=99)])
    out, stats = dc.dedupe_full_row(df)
    assert len(out) == 2
    assert stats["values_changed"] == 1


def test_replace_undefined_with_null():
    df = _sample_df([_sample_row(meal="Undefined"), _sample_row(meal="BB")])
    out, stats = dc.replace_undefined_with_null(df)
    assert pd.isna(out.loc[0, "meal"])
    assert out.loc[1, "meal"] == "BB"
    assert stats["per_column"]["meal"] == 1


def test_impute_missing_children_fills_zero():
    df = _sample_df([_sample_row(children=None)])
    out, stats = dc.impute_missing_children(df)
    assert out.loc[0, "children"] == 0
    assert stats["values_changed"] == 1


def test_impute_missing_country_fills_unk():
    df = _sample_df([_sample_row(country=None)])
    out, stats = dc.impute_missing_country(df)
    assert out.loc[0, "country"] == "UNK"
    assert stats["values_changed"] == 1


def test_fix_country_code_cn_maps_to_chn():
    df = _sample_df([_sample_row(country="CN")])
    out, stats = dc.fix_country_code_cn(df)
    assert out.loc[0, "country"] == "CHN"
    assert stats["values_changed"] == 1


def test_clean_adr_drops_negative_rows():
    df = _sample_df([_sample_row(adr=-5.0), _sample_row(adr=100.0)])
    out, stats = dc.clean_adr(df)
    assert len(out) == 1
    assert stats["rows_dropped_negative"] == 1


def test_clean_adr_winsorizes_above_995th_percentile():
    rows = [_sample_row(adr=float(i)) for i in range(1, 200)]
    rows.append(_sample_row(adr=100000.0))
    df = _sample_df(rows)
    out, stats = dc.clean_adr(df, winsor_pct=0.995)
    assert len(out) == len(df)  # no rows dropped by winsorizing
    cap = stats["winsor_cap"]
    assert out["adr"].max() == pytest.approx(cap)
    assert stats["values_changed"] >= 1


def test_flag_adults_outliers_true_above_threshold_false_at_or_below():
    df = _sample_df([_sample_row(adults=11), _sample_row(adults=10), _sample_row(adults=2)])
    out, stats = dc.flag_adults_outliers(df, threshold=10)
    assert out["is_adults_outlier"].tolist() == [True, False, False]
    assert stats["values_changed"] == 1


def test_flag_adults_outliers_preserves_raw_adults_values():
    df = _sample_df([_sample_row(adults=55)])
    out, _ = dc.flag_adults_outliers(df)
    assert out.loc[0, "adults"] == 55


def test_check_cancellation_consistency_detects_mismatch_without_dropping():
    df = _sample_df([_sample_row(reservation_status="Canceled", is_canceled=0)])
    out, stats = dc.check_cancellation_consistency(df)
    assert len(out) == 1
    assert stats["mismatch_count"] == 1


def test_build_arrival_date_maps_month_name_to_date():
    df = _sample_df([_sample_row(arrival_date_year=2016, arrival_date_month="July", arrival_date_day_of_month=15)])
    out, stats = dc.build_arrival_date(df)
    assert out.loc[0, "arrival_date"] == pd.Timestamp("2016-07-15")
    assert stats["unparseable_dates"] == 0


def test_build_arrival_date_coerces_invalid_to_nat():
    df = _sample_df([_sample_row(arrival_date_year=2016, arrival_date_month="April", arrival_date_day_of_month=31)])
    out, stats = dc.build_arrival_date(df)
    assert pd.isna(out.loc[0, "arrival_date"])
    assert stats["unparseable_dates"] == 1


def test_convert_dtypes_produces_expected_dtypes():
    df = _sample_df([_sample_row()])
    out, _ = dc.convert_dtypes(df)
    assert str(out["adults"].dtype) == "Int64"
    assert str(out["adr"].dtype) == "float64"
    assert str(out["hotel"].dtype) == "category"


def test_clean_end_to_end_row_count():
    rows = [
        _sample_row(),
        _sample_row(),  # exact duplicate -> dropped
        _sample_row(adr=-1.0, lead_time=5),  # negative adr -> dropped
        _sample_row(meal="Undefined", lead_time=6),
    ]
    df = _sample_df(rows)
    out, stats = dc.clean(df)
    assert len(out) == 2
    assert "is_adults_outlier" in out.columns
    assert "arrival_date" in out.columns
