"""Tests for src/mlops/monitoring/metrics_middleware.py. prometheus_client and
prometheus_fastapi_instrumentator ARE installed on this host, so these run
against the real library (no mocking needed).
"""

from fastapi import FastAPI
from fastapi.testclient import TestClient
from prometheus_client import generate_latest

from src.mlops.monitoring.metrics_middleware import (
    PREDICTION_COUNTER,
    PREDICTION_ERRORS,
    instrument_app,
    record_prediction,
    update_resource_gauges,
)


def test_record_prediction_increments_counter_and_histogram():
    before = PREDICTION_COUNTER.labels(domain="test_mw", model_version="1")._value.get()
    record_prediction("test_mw", "1", 0.05, error=False)
    after = PREDICTION_COUNTER.labels(domain="test_mw", model_version="1")._value.get()
    assert after == before + 1


def test_record_prediction_error_increments_error_counter_not_success_counter():
    before_errors = PREDICTION_ERRORS.labels(domain="test_mw_err")._value.get()
    before_success = PREDICTION_COUNTER.labels(domain="test_mw_err", model_version="1")._value.get()

    record_prediction("test_mw_err", "1", 0.01, error=True)

    after_errors = PREDICTION_ERRORS.labels(domain="test_mw_err")._value.get()
    after_success = PREDICTION_COUNTER.labels(domain="test_mw_err", model_version="1")._value.get()

    assert after_errors == before_errors + 1
    assert after_success == before_success  # unchanged on error path


def test_record_prediction_never_raises_on_bad_input():
    # Passing a non-numeric latency should be swallowed, not raised.
    record_prediction("test_mw_bad", "1", "not-a-float", error=False)


def test_prediction_latency_is_observed():
    record_prediction("test_mw_latency", "1", 0.123, error=False)
    metrics_text = generate_latest().decode("utf-8")
    assert "hotelmind_prediction_latency_seconds" in metrics_text


def test_update_resource_gauges_runs_without_error():
    update_resource_gauges()  # should not raise
    metrics_text = generate_latest().decode("utf-8")
    assert "hotelmind_process_cpu_percent" in metrics_text
    assert "hotelmind_process_ram_mb" in metrics_text


def test_metric_names_present_in_generate_latest():
    record_prediction("test_mw_names", "1", 0.01)
    metrics_text = generate_latest().decode("utf-8")
    assert "hotelmind_predictions_total" in metrics_text
    assert "hotelmind_prediction_errors_total" in metrics_text


def test_instrument_app_exposes_metrics_endpoint():
    app = FastAPI()

    @app.get("/ping")
    def ping():
        return {"ok": True}

    instrument_app(app)
    client = TestClient(app)

    resp = client.get("/ping")
    assert resp.status_code == 200

    metrics_resp = client.get("/metrics")
    assert metrics_resp.status_code == 200
    assert "text/plain" in metrics_resp.headers["content-type"]
