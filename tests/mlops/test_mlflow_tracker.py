"""Tests for src/mlops/tracking/mlflow_tracker.py.

`mlflow` cannot be installed on this host (Python 3.14, no cmake for
evidently/mlflow's transitive deps) -- see repo-level notes in
mlflow_tracker.py. We stub `mlflow` in sys.modules with a MagicMock *before*
importing mlflow_tracker, so the module's top-level `import mlflow` succeeds
and every mlflow.* call becomes a mock we can assert against, per the M16
spec ("mock mlflow, assert log_params/log_metrics called").
"""

import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def mlflow_mock(monkeypatch):
    """Install a MagicMock as the `mlflow` module and reload mlflow_tracker
    against it, returning the mock for assertions. Restores the real module
    state afterwards (module either wasn't importable, or we swap back)."""
    fake_mlflow = MagicMock()
    # start_run must behave as a context manager yielding a fake "run" object.
    fake_run = MagicMock()
    fake_run.info.run_id = "fake-run-id-123"
    fake_mlflow.start_run.return_value.__enter__.return_value = fake_run
    fake_mlflow.start_run.return_value.__exit__.return_value = False
    fake_mlflow.active_run.return_value.info.run_id = "fake-run-id-123"

    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    # Force a fresh import of mlflow_tracker bound to our fake mlflow module.
    sys.modules.pop("src.mlops.tracking.mlflow_tracker", None)
    import src.mlops.tracking.mlflow_tracker as tracker_module

    yield fake_mlflow, tracker_module

    sys.modules.pop("src.mlops.tracking.mlflow_tracker", None)


def test_mlflow_tracker_init_sets_uri_and_experiment(mlflow_mock):
    fake_mlflow, tracker_module = mlflow_mock
    tracker = tracker_module.MLflowTracker("occupancy")
    fake_mlflow.set_tracking_uri.assert_called_once()
    fake_mlflow.set_experiment.assert_called_once_with("occupancy")
    assert tracker.experiment_name == "occupancy"


def test_start_run_tags_environment_metadata(mlflow_mock):
    fake_mlflow, tracker_module = mlflow_mock
    tracker = tracker_module.MLflowTracker("occupancy")
    with tracker.start_run(run_name="run1") as run:
        assert run is not None
    fake_mlflow.start_run.assert_called_once_with(run_name="run1")
    fake_mlflow.set_tags.assert_called()
    tags = fake_mlflow.set_tags.call_args[0][0]
    assert "python_version" in tags
    assert "git_commit" in tags


def test_log_training_run_calls_expected_mlflow_functions(mlflow_mock):
    fake_mlflow, tracker_module = mlflow_mock
    from src.mlops.validation.dataset_version import DatasetVersion

    tracker = tracker_module.MLflowTracker("occupancy")
    dataset_version = DatasetVersion(
        source="occupancy", row_count=10, feature_count=3, content_hash="abc123",
        generated_at="2026-01-01T00:00:00+00:00",
    )

    run_id = tracker.log_training_run(
        metrics={"mae": 1.23},
        params={"model_key": "prophet"},
        feature_importance={"feat1": 0.5},
        training_seconds=2.5,
        dataset_version=dataset_version,
    )

    fake_mlflow.log_params.assert_called_once_with({"model_key": "prophet"})
    fake_mlflow.log_metrics.assert_called_once_with({"mae": 1.23})
    fake_mlflow.log_dict.assert_called_once_with({"feat1": 0.5}, "feature_importance.json")
    fake_mlflow.log_metric.assert_called_once_with("training_seconds", 2.5)
    assert run_id == "fake-run-id-123"

    tag_call = fake_mlflow.set_tags.call_args[0][0]
    assert tag_call["dataset_version"] == "abc123"
    assert tag_call["dataset_source"] == "occupancy"


def test_log_training_run_skips_feature_importance_when_none(mlflow_mock):
    fake_mlflow, tracker_module = mlflow_mock
    from src.mlops.validation.dataset_version import DatasetVersion

    tracker = tracker_module.MLflowTracker("occupancy")
    dataset_version = DatasetVersion(
        source="occupancy", row_count=1, feature_count=1, content_hash="h",
        generated_at="2026-01-01T00:00:00+00:00",
    )
    tracker.log_training_run(
        metrics={"mae": 1.0},
        params={},
        feature_importance=None,
        training_seconds=1.0,
        dataset_version=dataset_version,
    )
    fake_mlflow.log_dict.assert_not_called()


class _FakeRegistry:
    """Minimal stand-in for ModelRegistry sufficient for auto_promote_if_better."""

    def __init__(self, production_records=None):
        self._production_records = production_records or []
        self.promoted_to_staging = []

    def promote(self, model_name, version, to):
        self.promoted_to_staging.append((model_name, version, to))

    def list_versions(self, model_name):
        return self._production_records


class _FakeRecord:
    def __init__(self, version, metrics):
        from src.mlops.registry.model_registry import ModelStage

        self.version = version
        self.metrics = metrics
        self.stage = ModelStage.PRODUCTION


def test_auto_promote_if_better_always_promotes_to_staging(mlflow_mock):
    _, tracker_module = mlflow_mock
    tracker = tracker_module.MLflowTracker("occupancy")
    registry = _FakeRegistry(production_records=[])

    result = tracker.auto_promote_if_better(
        model_name="occupancy_prophet",
        new_version=2,
        new_metrics={"mae": 1.0},
        registry=registry,
        primary_metric="mae",
        lower_is_better=True,
    )
    from src.mlops.registry.model_registry import ModelStage

    assert registry.promoted_to_staging == [("occupancy_prophet", 2, ModelStage.STAGING)]
    # No production record exists yet -> considered an improvement.
    assert result is True


def test_auto_promote_if_better_lower_is_better_true_when_improved(mlflow_mock):
    _, tracker_module = mlflow_mock
    tracker = tracker_module.MLflowTracker("occupancy")
    registry = _FakeRegistry(production_records=[_FakeRecord(version=1, metrics={"mae": 5.0})])

    result = tracker.auto_promote_if_better(
        model_name="occupancy_prophet",
        new_version=2,
        new_metrics={"mae": 3.0},  # lower error -> better
        registry=registry,
        primary_metric="mae",
        lower_is_better=True,
    )
    assert result is True


def test_auto_promote_if_better_lower_is_better_false_when_worse(mlflow_mock):
    _, tracker_module = mlflow_mock
    tracker = tracker_module.MLflowTracker("occupancy")
    registry = _FakeRegistry(production_records=[_FakeRecord(version=1, metrics={"mae": 2.0})])

    result = tracker.auto_promote_if_better(
        model_name="occupancy_prophet",
        new_version=2,
        new_metrics={"mae": 5.0},  # higher error -> worse
        registry=registry,
        primary_metric="mae",
        lower_is_better=True,
    )
    assert result is False


def test_auto_promote_if_better_higher_is_better_true_when_improved(mlflow_mock):
    _, tracker_module = mlflow_mock
    tracker = tracker_module.MLflowTracker("churn")
    registry = _FakeRegistry(production_records=[_FakeRecord(version=1, metrics={"roc_auc": 0.80})])

    result = tracker.auto_promote_if_better(
        model_name="churn_random_forest",
        new_version=2,
        new_metrics={"roc_auc": 0.90},  # higher AUC -> better
        registry=registry,
        primary_metric="roc_auc",
        lower_is_better=False,
    )
    assert result is True


def test_auto_promote_if_better_higher_is_better_false_when_worse(mlflow_mock):
    _, tracker_module = mlflow_mock
    tracker = tracker_module.MLflowTracker("churn")
    registry = _FakeRegistry(production_records=[_FakeRecord(version=1, metrics={"roc_auc": 0.95})])

    result = tracker.auto_promote_if_better(
        model_name="churn_random_forest",
        new_version=2,
        new_metrics={"roc_auc": 0.70},
        registry=registry,
        primary_metric="roc_auc",
        lower_is_better=False,
    )
    assert result is False


def test_auto_promote_if_better_missing_metric_defaults_true(mlflow_mock):
    _, tracker_module = mlflow_mock
    tracker = tracker_module.MLflowTracker("occupancy")
    registry = _FakeRegistry(production_records=[_FakeRecord(version=1, metrics={})])

    result = tracker.auto_promote_if_better(
        model_name="occupancy_prophet",
        new_version=2,
        new_metrics={"mae": 1.0},
        registry=registry,
        primary_metric="mae",
        lower_is_better=True,
    )
    assert result is True


def test_auto_promote_if_better_list_versions_raises_file_not_found(mlflow_mock):
    _, tracker_module = mlflow_mock
    tracker = tracker_module.MLflowTracker("occupancy")

    class _RaisingRegistry(_FakeRegistry):
        def list_versions(self, model_name):
            raise FileNotFoundError

    registry = _RaisingRegistry()
    result = tracker.auto_promote_if_better(
        model_name="occupancy_prophet",
        new_version=1,
        new_metrics={"mae": 1.0},
        registry=registry,
        primary_metric="mae",
        lower_is_better=True,
    )
    assert result is True
