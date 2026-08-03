"""Tests for rollback.py (repo root CLI).

rollback.py imports promote.resolve_model_name and reuses it, so we mostly
verify main()'s registry-state-transition behavior; resolve_model_name's own
logic is already covered by test_promote_cli.py.
"""

import sys

import joblib
import pytest

import rollback
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


def test_main_rolls_back_to_archived_version(tmp_path, registry, monkeypatch):
    source = _make_source_pkl(tmp_path)
    registry.register_model("occupancy_prophet", source, metrics={}, dataset_version="v1")
    registry.register_model("occupancy_prophet", source, metrics={}, dataset_version="v2")
    registry.promote("occupancy_prophet", 1, ModelStage.PRODUCTION)
    registry.promote("occupancy_prophet", 2, ModelStage.PRODUCTION)  # archives v1

    monkeypatch.setattr(sys, "argv", ["rollback.py", "--model", "occupancy"])
    exit_code = rollback.main()

    assert exit_code == 0
    record = registry.get_record("occupancy_prophet", version=1)
    assert record.stage == ModelStage.PRODUCTION


def test_main_rollback_no_archived_fails(tmp_path, registry, monkeypatch):
    source = _make_source_pkl(tmp_path)
    registry.register_model("occupancy_prophet", source, metrics={}, dataset_version="v1")
    registry.promote("occupancy_prophet", 1, ModelStage.PRODUCTION)

    monkeypatch.setattr(sys, "argv", ["rollback.py", "--model", "occupancy"])
    exit_code = rollback.main()
    assert exit_code == 1


def test_main_rollback_explicit_version(tmp_path, registry, monkeypatch):
    source = _make_source_pkl(tmp_path)
    registry.register_model("occupancy_prophet", source, metrics={}, dataset_version="v1")
    registry.register_model("occupancy_prophet", source, metrics={}, dataset_version="v2")
    registry.promote("occupancy_prophet", 2, ModelStage.PRODUCTION)

    monkeypatch.setattr(
        sys, "argv", ["rollback.py", "--model", "occupancy", "--to-version", "1"]
    )
    exit_code = rollback.main()
    assert exit_code == 0

    record = registry.get_record("occupancy_prophet", version=1)
    assert record.stage == ModelStage.PRODUCTION


def test_main_rollback_restaurant_requires_variant(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["rollback.py", "--model", "restaurant"])
    with pytest.raises(SystemExit):
        rollback.main()
