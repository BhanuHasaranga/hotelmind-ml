"""Tests for src/mlops/pipelines/mlops_pipeline_mixin.py.

MLOpsPipelineMixin.run_with_mlops() transitively depends on MLflowTracker
(which imports the real `mlflow` package at module level -- unavailable on
this Python 3.14 host). We stub `mlflow` in sys.modules before importing the
mixin module, then exercise run_with_mlops()'s orchestration logic against a
fully mocked pipeline (load/clean/engineer/split/train/evaluate/save) rather
than a real Prophet/XGBoost training run, per the M16 spec's guidance to
focus this test on orchestration rather than real training.
"""

import sys
from unittest.mock import MagicMock

import pandas as pd
import pytest


@pytest.fixture
def mixin_module(monkeypatch):
    fake_mlflow = MagicMock()
    fake_run = MagicMock()
    fake_run.info.run_id = "fake-run-id"
    fake_mlflow.start_run.return_value.__enter__.return_value = fake_run
    fake_mlflow.start_run.return_value.__exit__.return_value = False
    fake_mlflow.active_run.return_value.info.run_id = "fake-run-id"
    monkeypatch.setitem(sys.modules, "mlflow", fake_mlflow)

    for mod_name in (
        "src.mlops.tracking.mlflow_tracker",
        "src.mlops.pipelines.mlops_pipeline_mixin",
    ):
        sys.modules.pop(mod_name, None)

    import src.mlops.pipelines.mlops_pipeline_mixin as mixin_mod

    yield mixin_mod

    sys.modules.pop("src.mlops.pipelines.mlops_pipeline_mixin", None)
    sys.modules.pop("src.mlops.tracking.mlflow_tracker", None)


class _FakePipeline:
    """A minimal stand-in implementing exactly the hooks MLOpsPipelineMixin
    expects from a mixed-in BasePipeline subclass, all instrumented so the
    test can assert call order and arguments.
    """

    def __init__(self):
        self.mlflow_experiment_name = "occupancy"
        self.registry_model_names = ["occupancy_prophet"]
        self.module_name = "occupancy"
        self.logger = MagicMock()
        self.calls: list[str] = []

    def load_data(self, **kwargs):
        self.calls.append("load_data")
        return pd.DataFrame({"a": [1, 2, 3]})

    def clean(self, df):
        self.calls.append("clean")
        return df

    def engineer_features(self, df):
        self.calls.append("engineer_features")
        return df

    def split(self, df):
        self.calls.append("split")
        return df, df, df["a"], df["a"]

    def build_models(self):
        self.calls.append("build_models")
        return {"prophet": MagicMock()}

    def train(self, models, X_train, y_train):
        self.calls.append("train")
        return models

    def evaluate(self, models, X_test, y_test):
        self.calls.append("evaluate")
        return {"prophet": {"mae": 1.5}}

    def model_save_path(self, model_key):
        return f"/fake/path/{model_key}.pkl"

    def save_models(self, models):
        self.calls.append("save_models")


def _make_pipeline_instance(mixin_mod):
    # NOTE: _FakePipeline must come first in the MRO -- MLOpsPipelineMixin
    # declares stub methods (load_data, clean, ...) purely for type-hinting
    # purposes, and putting the mixin first would let those stubs shadow
    # _FakePipeline's real implementations.
    class _CombinedPipeline(_FakePipeline, mixin_mod.MLOpsPipelineMixin):
        pass

    return _CombinedPipeline()


def test_run_with_mlops_calls_steps_in_order(mixin_module, monkeypatch, tmp_path):
    pipeline = _make_pipeline_instance(mixin_module)

    fake_registry = MagicMock()
    fake_record = MagicMock(version=1)
    fake_registry.register_model.return_value = fake_record
    monkeypatch.setattr(mixin_module, "ModelRegistry", lambda: fake_registry)

    fake_tracker = MagicMock()
    fake_tracker.log_training_run.return_value = "run-1"
    fake_tracker.auto_promote_if_better.return_value = True
    monkeypatch.setattr(mixin_module, "MLflowTracker", lambda name: fake_tracker)

    monkeypatch.setattr(mixin_module, "write_report", MagicMock())
    write_dashboard_mock = MagicMock()
    monkeypatch.setattr(mixin_module, "write_dashboard", write_dashboard_mock)

    # Isolate dataset-version I/O to tmp_path.
    from src.mlops.config import mlops_settings as settings_module

    versions_dir = tmp_path / "dataset_versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        type(settings_module.mlops_settings),
        "dataset_versions_dir_path",
        property(lambda self: versions_dir),
    )

    summary = pipeline.run_with_mlops()

    assert pipeline.calls == [
        "load_data",
        "clean",
        "engineer_features",
        "split",
        "build_models",
        "train",
        "evaluate",
        "save_models",
    ]

    fake_registry.register_model.assert_called_once()
    register_kwargs = fake_registry.register_model.call_args.kwargs
    assert register_kwargs["model_name"] == "occupancy_prophet"
    assert register_kwargs["metrics"] == {"mae": 1.5}

    fake_tracker.auto_promote_if_better.assert_called_once()
    write_dashboard_mock.assert_called_once()

    assert summary["dataset_changed"] is True
    assert "occupancy_prophet" in summary["registered"]
    assert summary["registered"]["occupancy_prophet"]["version"] == 1
    assert summary["registered"]["occupancy_prophet"]["beats_production"] is True


def test_registry_name_for_key_maps_suffix(mixin_module):
    pipeline = _make_pipeline_instance(mixin_module)
    pipeline.registry_model_names = ["occupancy_prophet", "occupancy_xgboost"]
    assert pipeline._registry_name_for_key("prophet") == "occupancy_prophet"
    assert pipeline._registry_name_for_key("xgboost") == "occupancy_xgboost"


def test_registry_name_for_key_single_model_fallback(mixin_module):
    pipeline = _make_pipeline_instance(mixin_module)
    pipeline.registry_model_names = ["pricing_xgboost"]
    assert pipeline._registry_name_for_key("anything") == "pricing_xgboost"


def test_registry_name_for_key_unmappable_raises(mixin_module):
    pipeline = _make_pipeline_instance(mixin_module)
    pipeline.registry_model_names = ["occupancy_prophet", "occupancy_xgboost"]
    with pytest.raises(ValueError):
        pipeline._registry_name_for_key("unknown_key")


def test_concrete_pipeline_wrappers_importable():
    """The 5 concrete *_mlops_pipeline.py wrappers must be valid compositions
    (mixin + existing BasePipeline subclass) with correctly set class attrs.
    Import is deferred inside the test since these transitively import mlflow
    via the mixin module."""
    import sys as _sys
    from unittest.mock import MagicMock as _MM

    fake_mlflow = _MM()
    _sys.modules["mlflow"] = fake_mlflow
    for mod in list(_sys.modules):
        if mod.startswith("src.mlops.pipelines") or mod == "src.mlops.tracking.mlflow_tracker":
            _sys.modules.pop(mod, None)

    from src.mlops.pipelines.churn_mlops_pipeline import ChurnMLOpsPipeline
    from src.mlops.pipelines.occupancy_mlops_pipeline import OccupancyMLOpsPipeline
    from src.mlops.pipelines.pricing_mlops_pipeline import PricingMLOpsPipeline
    from src.mlops.pipelines.restaurant_mlops_pipeline import RestaurantMLOpsPipeline
    from src.mlops.pipelines.staffing_mlops_pipeline import StaffingMLOpsPipeline

    assert OccupancyMLOpsPipeline.mlflow_experiment_name == "occupancy"
    assert set(OccupancyMLOpsPipeline.registry_model_names) == {
        "occupancy_prophet",
        "occupancy_xgboost",
    }
    assert PricingMLOpsPipeline.registry_model_names == ["pricing_xgboost"]
    assert StaffingMLOpsPipeline.registry_model_names == ["staffing_regression"]
    assert set(ChurnMLOpsPipeline.registry_model_names) == {
        "churn_random_forest",
        "churn_xgboost",
    }
    assert set(RestaurantMLOpsPipeline.registry_model_names) == {
        "restaurant_breakfast",
        "restaurant_lunch",
        "restaurant_dinner",
    }

    _sys.modules.pop("mlflow", None)


def test_restaurant_pipeline_run_with_mlops_per_meal_loop(monkeypatch, tmp_path):
    """RestaurantMLOpsPipeline overrides run_with_mlops with its own per-meal
    (breakfast/lunch/dinner) loop instead of using the mixin's generic flow --
    covered separately here since it is not exercised by the generic mixin test.
    """
    import sys as _sys
    from unittest.mock import MagicMock as _MM

    fake_mlflow = _MM()
    fake_run = _MM()
    fake_run.info.run_id = "fake-run-id"
    fake_mlflow.start_run.return_value.__enter__.return_value = fake_run
    fake_mlflow.start_run.return_value.__exit__.return_value = False
    fake_mlflow.active_run.return_value.info.run_id = "fake-run-id"
    monkeypatch.setitem(_sys.modules, "mlflow", fake_mlflow)

    for mod in (
        "src.mlops.tracking.mlflow_tracker",
        "src.mlops.pipelines.restaurant_mlops_pipeline",
    ):
        _sys.modules.pop(mod, None)

    import src.mlops.pipelines.restaurant_mlops_pipeline as restaurant_mod

    pipeline = restaurant_mod.RestaurantMLOpsPipeline.__new__(
        restaurant_mod.RestaurantMLOpsPipeline
    )
    pipeline.module_name = "restaurant"
    pipeline.logger = _MM()

    df = pd.DataFrame({"a": [1, 2, 3]})
    monkeypatch.setattr(pipeline, "load_data", lambda **kw: df)
    monkeypatch.setattr(pipeline, "clean", lambda d: d)
    monkeypatch.setattr(pipeline, "engineer_features", lambda d: d)

    fake_model = _MM()
    fake_model.evaluate.return_value = {"mae": 1.0}
    monkeypatch.setattr(pipeline, "build_models", lambda: {"breakfast": fake_model})
    monkeypatch.setattr(pipeline, "model_save_path", lambda meal: tmp_path / f"{meal}.pkl")

    monkeypatch.setattr(
        restaurant_mod,
        "default_time_series_split",
        lambda df, target_col, feature_cols, test_size=0.2: (df, df, df["a"], df["a"]),
    )

    versions_dir = tmp_path / "dataset_versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    from src.mlops.config import mlops_settings as settings_module

    monkeypatch.setattr(
        type(settings_module.mlops_settings),
        "dataset_versions_dir_path",
        property(lambda self: versions_dir),
    )

    fake_registry = _MM()
    fake_record = _MM(version=1)
    fake_registry.register_model.return_value = fake_record
    monkeypatch.setattr(restaurant_mod, "ModelRegistry", lambda: fake_registry)

    fake_tracker = _MM()
    fake_tracker.auto_promote_if_better.return_value = True
    monkeypatch.setattr(restaurant_mod, "MLflowTracker", lambda name: fake_tracker)

    summary = pipeline.run_with_mlops()

    fake_model.train.assert_called_once()
    fake_model.evaluate.assert_called_once()
    fake_model.save.assert_called_once()
    assert "restaurant_breakfast" in summary["registered"]
    assert summary["registered"]["restaurant_breakfast"]["version"] == 1

    _sys.modules.pop("mlflow", None)
