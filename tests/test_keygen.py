import pandas as pd
import pytest

from src.pipelines import keygen


def test_stable_hash_int_deterministic():
    assert keygen.stable_hash_int("a", "b", 1) == keygen.stable_hash_int("a", "b", 1)


def test_stable_hash_int_differs_for_different_inputs():
    assert keygen.stable_hash_int("a", "b") != keygen.stable_hash_int("a", "c")


def test_date_id_format():
    assert keygen.date_id(pd.Timestamp("2024-07-05")) == 20240705


def test_date_id_nat_returns_none():
    assert keygen.date_id(pd.NaT) is None


def test_hotel_id_fixed_lookup():
    assert keygen.hotel_id("Resort Hotel") == 1
    assert keygen.hotel_id("City Hotel") == 2
    with pytest.raises(KeyError):
        keygen.hotel_id("Unknown Hotel")


def test_room_type_id_known_and_unmapped_codes():
    assert keygen.room_type_id("A") == 1
    assert keygen.room_type_id("D") == 2
    assert keygen.room_type_id("H") == 3
    assert keygen.room_type_id("Z") == keygen.UNMAPPED_ROOM_TYPE_ID


def _profile_row(**overrides) -> pd.Series:
    row = {
        "country": "PRT", "market_segment": "Direct", "distribution_channel": "Direct",
        "customer_type": "Transient", "is_repeated_guest": 0, "agent": None, "company": None,
        "adults": 2, "children": 0, "adr": 100.0,
    }
    row.update(overrides)
    return pd.Series(row)


def test_guest_id_deterministic_same_profile():
    row1 = _profile_row(adr=50.0)
    row2 = _profile_row(adr=999.0)
    assert keygen.guest_id(row1) == keygen.guest_id(row2)


def test_guest_id_differs_for_different_profile():
    row1 = _profile_row(country="PRT")
    row2 = _profile_row(country="USA")
    assert keygen.guest_id(row1) != keygen.guest_id(row2)


def _full_row(**overrides) -> pd.Series:
    row = {c: f"val_{c}" for c in ["hotel", "lead_time", "adr"]}
    row.update(overrides)
    return pd.Series(row)


def test_reservation_id_deterministic_across_repeated_calls():
    row = _full_row()
    columns = list(row.index)
    id1 = keygen.reservation_id(row, 5, columns)
    id2 = keygen.reservation_id(row, 5, columns)
    assert id1 == id2


def test_reservation_id_differs_by_row_index_for_identical_content():
    row = _full_row()
    columns = list(row.index)
    id1 = keygen.reservation_id(row, 5, columns)
    id2 = keygen.reservation_id(row, 6, columns)
    assert id1 != id2
