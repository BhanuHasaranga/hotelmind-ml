"""Tests for src/mlops/pipelines/retrain_trigger.py::check_and_retrain.

retrain_trigger.py imports MLflowTracker at module level, which imports the
real `mlflow` package -- unavailable on this host. We stub `mlflow` in
sys.modules before importing, then mock get_latest_drift_score / registry
records to simulate threshold-breach vs no-breach scenarios, and mock out the
pipeline class's run_with_mlops() so no real training happens.
"""

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def retrain_module(monkeypatch):
    fake_mlflow = MagicMock()
    fake_mlflow.active_run.return_value.info.run_id = "fake-run-id"
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    for mod_name in (
        "src.mlops.tracking.mlflow_tracker",
        "src.mlops.pipelines.retrain_trigger",
    ):
        sys.modules.pop(mod_name, None)

    import src.mlops.pipelines.retrain_trigger as rt_module

    yield rt_module

    sys.modules.pop("src.mlops.pipelines.retrain_trigger", None)
    sys.modules.pop("src.mlops.tracking.mlflow_tracker", None)


class _FakeRecord:
    def __init__(self, version, metrics, stage):
        self.version = version
        self.metrics = metrics
        self.stage = stage


def test_no_retrain_when_no_signal(retrain_module, monkeypatch):
    from src.mlops.registry.model_registry import ModelStage

    monkeypatch.setattr(retrain_module, "get_latest_drift_score", lambda name: 0.05)

    fake_registry = MagicMock()
    fake_registry.list_versions.return_value = [
        _FakeRecord(1, {"mae": 1.0}, ModelStage.PRODUCTION),
    ]
    monkeypatch.setattr(retrain_module, "ModelRegistry", lambda: fake_registry)

    fake_pipeline_cls = MagicMock()

    result = retrain_module.check_and_retrain({"occupancy_prophet": fake_pipeline_cls})

    assert result == {"occupancy_prophet": False}
    fake_pipeline_cls.assert_not_called()


def test_retrain_triggered_by_drift_breach(retrain_module, monkeypatch):
    from src.mlops.registry.model_registry import ModelStage

    monkeypatch.setattr(retrain_module, "get_latest_drift_score", lambda name: 0.99)

    fake_registry = MagicMock()
    fake_registry.list_versions.return_value = [
        _FakeRecord(1, {"mae": 1.0}, ModelStage.PRODUCTION),
    ]
    monkeypatch.setattr(retrain_module, "ModelRegistry", lambda: fake_registry)

    fake_pipeline_instance = MagicMock()
    fake_pipeline_instance.run_with_mlops.return_value = {
        "registered": {"occupancy_prophet": {"version": 2, "metrics": {"mae": 0.5}}}
    }
    fake_pipeline_instance.mlflow_experiment_name = "occupancy"
    fake_pipeline_cls = MagicMock(return_value=fake_pipeline_instance)

    fake_tracker = MagicMock()
    fake_tracker.auto_promote_if_better.return_value = True
    monkeypatch.setattr(retrain_module, "MLflowTracker", lambda name: fake_tracker)

    result = retrain_module.check_and_retrain({"occupancy_prophet": fake_pipeline_cls})

    assert result == {"occupancy_prophet": True}
    fake_pipeline_instance.run_with_mlops.assert_called_once()
    fake_registry.promote.assert_called_once_with(
        "occupancy_prophet", 2, ModelStage.PRODUCTION
    )


def test_retrain_triggered_but_does_not_promote_when_not_better(retrain_module, monkeypatch):
    from src.mlops.registry.model_registry import ModelStage

    monkeypatch.setattr(retrain_module, "get_latest_drift_score", lambda name: 0.99)

    fake_registry = MagicMock()
    fake_registry.list_versions.return_value = [
        _FakeRecord(1, {"mae": 1.0}, ModelStage.PRODUCTION),
    ]
    monkeypatch.setattr(retrain_module, "ModelRegistry", lambda: fake_registry)

    fake_pipeline_instance = MagicMock()
    fake_pipeline_instance.run_with_mlops.return_value = {
        "registered": {"occupancy_prophet": {"version": 2, "metrics": {"mae": 5.0}}}
    }
    fake_pipeline_instance.mlflow_experiment_name = "occupancy"
    fake_pipeline_cls = MagicMock(return_value=fake_pipeline_instance)

    fake_tracker = MagicMock()
    fake_tracker.auto_promote_if_better.return_value = False
    monkeypatch.setattr(retrain_module, "MLflowTracker", lambda name: fake_tracker)

    result = retrain_module.check_and_retrain({"occupancy_prophet": fake_pipeline_cls})

    assert result == {"occupancy_prophet": True}
    fake_registry.promote.assert_not_called()


def test_retrain_triggered_by_accuracy_drop(retrain_module, monkeypatch):
    from src.mlops.registry.model_registry import ModelStage

    monkeypatch.setattr(retrain_module, "get_latest_drift_score", lambda name: None)

    fake_registry = MagicMock()
    fake_registry.list_versions.return_value = [
        _FakeRecord(1, {"mae": 1.0}, ModelStage.PRODUCTION),
        _FakeRecord(2, {"mae": 5.0}, ModelStage.STAGING),  # big accuracy drop vs production
    ]
    monkeypatch.setattr(retrain_module, "ModelRegistry", lambda: fake_registry)

    fake_pipeline_instance = MagicMock()
    fake_pipeline_instance.run_with_mlops.return_value = {
        "registered": {"occupancy_prophet": {"version": 3, "metrics": {"mae": 0.9}}}
    }
    fake_pipeline_instance.mlflow_experiment_name = "occupancy"
    fake_pipeline_cls = MagicMock(return_value=fake_pipeline_instance)

    fake_tracker = MagicMock()
    fake_tracker.auto_promote_if_better.return_value = True
    monkeypatch.setattr(retrain_module, "MLflowTracker", lambda name: fake_tracker)

    result = retrain_module.check_and_retrain({"occupancy_prophet": fake_pipeline_cls})

    assert result == {"occupancy_prophet": True}
    fake_pipeline_instance.run_with_mlops.assert_called_once()


def test_check_and_retrain_handles_pipeline_exception(retrain_module, monkeypatch):
    from src.mlops.registry.model_registry import ModelStage

    monkeypatch.setattr(retrain_module, "get_latest_drift_score", lambda name: 0.99)

    fake_registry = MagicMock()
    fake_registry.list_versions.return_value = [
        _FakeRecord(1, {"mae": 1.0}, ModelStage.PRODUCTION),
    ]
    monkeypatch.setattr(retrain_module, "ModelRegistry", lambda: fake_registry)

    fake_pipeline_cls = MagicMock(side_effect=RuntimeError("boom"))

    result = retrain_module.check_and_retrain({"occupancy_prophet": fake_pipeline_cls})

    # triggered flag flips back to False when the retrain attempt itself fails
    assert result == {"occupancy_prophet": False}


def test_accuracy_drop_no_production_record_returns_false(retrain_module, monkeypatch):
    monkeypatch.setattr(retrain_module, "get_latest_drift_score", lambda name: None)

    fake_registry = MagicMock()
    fake_registry.list_versions.return_value = []
    monkeypatch.setattr(retrain_module, "ModelRegistry", lambda: fake_registry)

    fake_pipeline_cls = MagicMock()
    result = retrain_module.check_and_retrain({"occupancy_prophet": fake_pipeline_cls})
    assert result == {"occupancy_prophet": False}
    fake_pipeline_cls.assert_not_called()


def test_accuracy_drop_helper_directly_higher_is_better(retrain_module):
    from src.mlops.registry.model_registry import ModelStage

    fake_registry = MagicMock()
    fake_registry.list_versions.return_value = [
        _FakeRecord(1, {"roc_auc": 0.9}, ModelStage.PRODUCTION),
        _FakeRecord(2, {"roc_auc": 0.5}, ModelStage.STAGING),
    ]
    breached = retrain_module._accuracy_drop_breached(
        fake_registry, "churn_xgboost", "roc_auc", lower_is_better=False
    )
    assert breached is True
