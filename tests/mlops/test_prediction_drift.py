"""Tests for src/mlops/monitoring/prediction_drift.py (pure Python/pandas-free,
no evidently dependency, runs fully on any host)."""

import json
from datetime import UTC, datetime, timedelta

import pytest

from src.mlops.monitoring.prediction_drift import (
    generate_prediction_drift_report,
    log_prediction_for_drift,
)


@pytest.fixture(autouse=True)
def isolated_drift_dir(tmp_path, monkeypatch):
    from src.mlops.config import mlops_settings as settings_module

    drift_dir = tmp_path / "drift"
    drift_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        type(settings_module.mlops_settings),
        "drift_reports_dir_path",
        property(lambda self: drift_dir),
    )
    return drift_dir


def test_log_prediction_for_drift_writes_valid_jsonl(isolated_drift_dir):
    log_prediction_for_drift("occupancy", {"branch_id": 1, "predicted_occupancy_pct": 55.0})
    log_path = isolated_drift_dir / "predictions" / "occupancy.jsonl"
    assert log_path.exists()

    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["branch_id"] == 1
    assert "_logged_at" in record


def test_log_prediction_for_drift_appends(isolated_drift_dir):
    log_prediction_for_drift("occupancy", {"a": 1})
    log_prediction_for_drift("occupancy", {"a": 2})
    log_path = isolated_drift_dir / "predictions" / "occupancy.jsonl"
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 2


def test_log_prediction_for_drift_never_raises_on_bad_input(monkeypatch):
    import src.mlops.monitoring.prediction_drift as pd_module

    def _boom():
        raise OSError("disk full")

    monkeypatch.setattr(pd_module, "_predictions_dir", _boom)
    # Must not raise despite _predictions_dir failing.
    log_prediction_for_drift("occupancy", {"a": 1})


def test_generate_prediction_drift_report_no_log_file_returns_none():
    assert generate_prediction_drift_report("no_such_domain") is None


def _write_jsonl(path, records):
    with path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")


def test_generate_prediction_drift_report_insufficient_data_returns_none(isolated_drift_dir):
    predictions_dir = isolated_drift_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    log_path = predictions_dir / "occupancy.jsonl"

    now = datetime.now(UTC)
    _write_jsonl(
        log_path,
        [{"value": 1.0, "_logged_at": now.isoformat()}],  # only 1 record total
    )
    assert generate_prediction_drift_report("occupancy", window_days=7) is None


def test_generate_prediction_drift_report_with_sufficient_history(isolated_drift_dir):
    predictions_dir = isolated_drift_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    log_path = predictions_dir / "occupancy.jsonl"

    now = datetime.now(UTC)
    recent = [
        {"value": 50.0 + i, "_logged_at": (now - timedelta(days=1)).isoformat()}
        for i in range(3)
    ]
    prior = [
        {"value": 50.0 + i, "_logged_at": (now - timedelta(days=10)).isoformat()}
        for i in range(3)
    ]
    _write_jsonl(log_path, recent + prior)

    out_path = generate_prediction_drift_report("occupancy", window_days=7)
    assert out_path is not None
    assert out_path.exists()

    payload = json.loads(out_path.read_text())
    assert payload["domain"] == "occupancy"
    assert payload["recent_count"] == 3
    assert payload["prior_count"] == 3
    assert "value" in payload["fields"]
    assert payload["fields"]["value"]["stable"] is True


def test_generate_prediction_drift_report_detects_unstable_shift(isolated_drift_dir):
    predictions_dir = isolated_drift_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    log_path = predictions_dir / "occupancy.jsonl"

    now = datetime.now(UTC)
    recent = [
        {"value": 500.0 + i, "_logged_at": (now - timedelta(days=1)).isoformat()}
        for i in range(3)
    ]
    prior = [
        {"value": 1.0 + i * 0.01, "_logged_at": (now - timedelta(days=10)).isoformat()}
        for i in range(3)
    ]
    _write_jsonl(log_path, recent + prior)

    out_path = generate_prediction_drift_report("occupancy", window_days=7)
    payload = json.loads(out_path.read_text())
    assert payload["stable"] is False
    assert payload["fields"]["value"]["stable"] is False


def test_generate_prediction_drift_report_skips_malformed_lines(isolated_drift_dir):
    predictions_dir = isolated_drift_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    log_path = predictions_dir / "occupancy.jsonl"

    now = datetime.now(UTC)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("not valid json\n")
        f.write("\n")
        f.write(
            json.dumps({"value": 1.0, "_logged_at": (now - timedelta(days=1)).isoformat()}) + "\n"
        )
        f.write(json.dumps({"value": 2.0}) + "\n")  # missing _logged_at, skipped
        f.write(
            json.dumps({"value": 3.0, "_logged_at": (now - timedelta(days=1)).isoformat()}) + "\n"
        )
        f.write(
            json.dumps({"value": 1.0, "_logged_at": (now - timedelta(days=10)).isoformat()}) + "\n"
        )
        f.write(
            json.dumps({"value": 2.0, "_logged_at": (now - timedelta(days=10)).isoformat()}) + "\n"
        )

    # Should not raise despite malformed/missing-field lines.
    result = generate_prediction_drift_report("occupancy", window_days=7)
    assert result is not None
