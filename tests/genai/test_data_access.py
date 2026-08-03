"""Tests for genai/data_access/* -- local mart synthesis, postgres timeout
handling, and the DATA_SOURCE abstraction switch. No live Postgres
connection is ever made (an unreachable host + short connect_timeout is
used to exercise the ConnectionError path quickly and deterministically)."""

from __future__ import annotations

import pytest

from genai.data_access import local_reader, postgres_reader, warehouse_source
from genai.config.genai_settings import genai_settings


def test_local_reader_read_mart_occupancy():
    df = local_reader.read_mart("mart_occupancy_daily")
    assert not df.empty
    assert {"branch_id", "date", "occupancy_pct"}.issubset(df.columns)


def test_local_reader_read_mart_revenue():
    df = local_reader.read_mart("mart_revenue_daily")
    assert not df.empty
    assert "total_revenue" in df.columns


def test_local_reader_read_mart_restaurant():
    df = local_reader.read_mart("mart_restaurant_daily")
    assert not df.empty
    assert "total_orders" in df.columns


def test_local_reader_read_mart_staff():
    df = local_reader.read_mart("mart_staff_daily")
    assert not df.empty
    assert "attendance_rate_pct" in df.columns


def test_local_reader_unknown_mart_raises():
    with pytest.raises(ValueError):
        local_reader.read_mart("mart_does_not_exist")


def test_postgres_reader_unknown_mart_raises():
    with pytest.raises(ValueError):
        postgres_reader.read_mart("mart_does_not_exist")


def test_postgres_reader_unreachable_raises_connection_error():
    from src.config.settings import settings

    original_host = settings.WAREHOUSE_DB_HOST
    original_port = settings.WAREHOUSE_DB_PORT
    settings.WAREHOUSE_DB_HOST = "192.0.2.1"  # TEST-NET-1, guaranteed unreachable
    settings.WAREHOUSE_DB_PORT = 5999
    try:
        with pytest.raises(ConnectionError):
            postgres_reader.read_mart("mart_revenue_daily")
    finally:
        settings.WAREHOUSE_DB_HOST = original_host
        settings.WAREHOUSE_DB_PORT = original_port


def test_warehouse_source_local(monkeypatch):
    monkeypatch.setattr(genai_settings, "DATA_SOURCE", "local")
    df = warehouse_source.get_mart("mart_occupancy_daily")
    assert not df.empty


def test_warehouse_source_postgres_routes_to_postgres_reader(monkeypatch):
    monkeypatch.setattr(genai_settings, "DATA_SOURCE", "postgres")
    called = {}

    def _fake_read_mart(name, limit=None):
        called["name"] = name
        import pandas as pd

        return pd.DataFrame({"branch_id": [1]})

    monkeypatch.setattr(postgres_reader, "read_mart", _fake_read_mart)
    df = warehouse_source.get_mart("mart_revenue_daily")
    assert called["name"] == "mart_revenue_daily"
    assert not df.empty


def test_warehouse_source_unknown_data_source_raises(monkeypatch):
    monkeypatch.setattr(genai_settings, "DATA_SOURCE", "carrier_pigeon")
    with pytest.raises(ValueError):
        warehouse_source.get_mart("mart_revenue_daily")


def test_list_marts():
    marts = warehouse_source.list_marts()
    assert set(marts) == {
        "mart_revenue_daily",
        "mart_occupancy_daily",
        "mart_restaurant_daily",
        "mart_staff_daily",
    }
