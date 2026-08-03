"""Tests for promote.py (repo root CLI).

promote.py's main() reads sys.argv itself via argparse, so we monkeypatch
sys.argv and call main() directly. The registry is redirected to tmp_path so
no test writes into the real models/registry/.
"""

import sys

import joblib
import pytest

import promote
from src.mlops.registry.model_registry import ModelRegistry, ModelStage


def _make_source_pkl(tmp_path, name="src_model.pkl"):
    path = tmp_path / name
    joblib.dump(
        {"model": "fake-estimator", "encoders": {}, "scaler": None, "feature_names": []}, path
    )
    return path


@pytest.fixture
def registry(tmp_path, monkeypatch):
    from src.mlops.config import mlops_settings as settings_module

    staging_dir = tmp_path / "staging"
    production_dir = tmp_path / "production"
    archived_dir = tmp_path / "archived"
    for d in (staging_dir, production_dir, archived_dir):
        d.mkdir(parents=True, exist_ok=True)

    monkeypatch.setattr(
        type(settings_module.mlops_settings),
        "model_staging_dir_path",
        property(lambda self: staging_dir),
    )
    monkeypatch.setattr(
        type(settings_module.mlops_settings),
        "model_production_dir_path",
        property(lambda self: production_dir),
    )
    monkeypatch.setattr(
        type(settings_module.mlops_settings),
        "model_archived_dir_path",
        property(lambda self: archived_dir),
    )

    reg_dir = tmp_path / "registry"
    reg_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        type(settings_module.mlops_settings),
        "model_registry_dir_path",
        property(lambda self: reg_dir),
    )
    return ModelRegistry(registry_dir=reg_dir)


def test_resolve_model_name_domain_default():
    assert promote.resolve_model_name("occupancy", None) == "occupancy_prophet"
    assert promote.resolve_model_name("pricing", None) == "pricing_xgboost"
    assert promote.resolve_model_name("staffing", None) == "staffing_regression"
    assert promote.resolve_model_name("churn", None) == "churn_xgboost"


def test_resolve_model_name_variant():
    assert promote.resolve_model_name("occupancy", "xgboost") == "occupancy_xgboost"
    assert promote.resolve_model_name("churn", "random_forest") == "churn_random_forest"


def test_resolve_model_name_restaurant_requires_variant():
    with pytest.raises(SystemExit):
        promote.resolve_model_name("restaurant", None)


def test_resolve_model_name_restaurant_with_variant():
    assert promote.resolve_model_name("restaurant", "lunch") == "restaurant_lunch"


def test_resolve_model_name_unknown_domain():
    with pytest.raises(SystemExit):
        promote.resolve_model_name("unknown_domain", None)


def test_resolve_model_name_unknown_variant():
    with pytest.raises(SystemExit):
        promote.resolve_model_name("occupancy", "not_a_variant")


def test_main_promotes_staging_when_no_production(tmp_path, registry, monkeypatch):
    source = _make_source_pkl(tmp_path)
    registry.register_model("occupancy_prophet", source, metrics={"mae": 1.0}, dataset_version="v1")

    monkeypatch.setattr(sys, "argv", ["promote.py", "--model", "occupancy"])
    exit_code = promote.main()

    assert exit_code == 0
    record = registry.get_record("occupancy_prophet", version=1)
    assert record.stage == ModelStage.PRODUCTION


def test_main_no_staging_version_fails(registry, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["promote.py", "--model", "occupancy"])
    exit_code = promote.main()
    assert exit_code == 1


def test_main_does_not_promote_when_worse(tmp_path, registry, monkeypatch):
    source = _make_source_pkl(tmp_path)
    registry.register_model("occupancy_prophet", source, metrics={"mae": 1.0}, dataset_version="v1")
    registry.promote("occupancy_prophet", 1, ModelStage.PRODUCTION)
    registry.register_model("occupancy_prophet", source, metrics={"mae": 5.0}, dataset_version="v2")

    monkeypatch.setattr(sys, "argv", ["promote.py", "--model", "occupancy"])
    exit_code = promote.main()
    assert exit_code == 1

    record = registry.get_record("occupancy_prophet", version=2)
    assert record.stage == ModelStage.STAGING


def test_main_force_promotes_even_when_worse(tmp_path, registry, monkeypatch):
    source = _make_source_pkl(tmp_path)
    registry.register_model("occupancy_prophet", source, metrics={"mae": 1.0}, dataset_version="v1")
    registry.promote("occupancy_prophet", 1, ModelStage.PRODUCTION)
    registry.register_model("occupancy_prophet", source, metrics={"mae": 5.0}, dataset_version="v2")

    monkeypatch.setattr(sys, "argv", ["promote.py", "--model", "occupancy", "--force"])
    exit_code = promote.main()
    assert exit_code == 0

    record = registry.get_record("occupancy_prophet", version=2)
    assert record.stage == ModelStage.PRODUCTION
