"""Tests for src/mlops/metrics/evaluation_dashboard.py."""

import json

import pytest

from src.mlops.metrics.evaluation_dashboard import (
    ModelEvalRecord,
    build_dashboard,
    write_dashboard,
)


@pytest.fixture(autouse=True)
def isolated_dashboards_dir(tmp_path, monkeypatch):
    import src.mlops.metrics.evaluation_dashboard as dashboard_module

    dashboards_dir = tmp_path / "dashboards"
    dashboards_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(dashboard_module, "_dashboards_dir", lambda: dashboards_dir)
    return dashboards_dir


def test_model_eval_record_defaults():
    record = ModelEvalRecord(model_name="occupancy_prophet", version=1, mae=1.2)
    assert record.rmse is None
    assert record.evaluated_at  # auto-populated
    assert record.model_name == "occupancy_prophet"


def test_build_dashboard_shape():
    records = [
        ModelEvalRecord(model_name="occupancy_prophet", version=1, mae=1.2),
        ModelEvalRecord(model_name="churn_xgboost", version=2, roc_auc=0.91),
    ]
    payload = build_dashboard(records)
    assert "generated_at" in payload
    assert len(payload["models"]) == 2
    assert payload["models"][0]["model_name"] == "occupancy_prophet"
    assert payload["models"][1]["roc_auc"] == 0.91


def test_write_dashboard_creates_latest_and_timestamped(isolated_dashboards_dir):
    records = [ModelEvalRecord(model_name="occupancy_prophet", version=1, mae=1.2)]
    latest_path = write_dashboard(records)

    assert latest_path.name == "model_evaluation_dashboard.json"
    assert latest_path.exists()

    payload = json.loads(latest_path.read_text())
    assert payload["models"][0]["model_name"] == "occupancy_prophet"

    timestamped = [
        p
        for p in isolated_dashboards_dir.glob("model_evaluation_dashboard_*.json")
    ]
    assert len(timestamped) == 1


def test_write_dashboard_overwrites_latest_on_repeated_calls(isolated_dashboards_dir):
    write_dashboard([ModelEvalRecord(model_name="a", version=1, mae=1.0)])
    latest_path = write_dashboard([ModelEvalRecord(model_name="b", version=1, mae=2.0)])

    payload = json.loads(latest_path.read_text())
    assert payload["models"][0]["model_name"] == "b"

    # At least one timestamped snapshot should exist (history preserved);
    # both calls may share the same second-resolution timestamp on a fast host.
    timestamped = list(isolated_dashboards_dir.glob("model_evaluation_dashboard_*.json"))
    assert len(timestamped) >= 1
