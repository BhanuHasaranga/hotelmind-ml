"""Tests for src/mlops/registry/model_registry.py, including the legacy-fallback
load_production() path. Uses tmp_path for an isolated registry dir per test so
tests never write into the real models/registry/.
"""

import joblib
import pytest

from src.mlops.registry.model_registry import ModelRecord, ModelRegistry, ModelStage


def _make_source_pkl(tmp_path, name="src_model.pkl"):
    path = tmp_path / name
    payload = {
        "model": "fake-estimator",
        "encoders": {},
        "scaler": None,
        "feature_names": ["a", "b"],
    }
    joblib.dump(payload, path)
    return path


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """A ModelRegistry backed entirely by tmp_path -- registry dir AND the
    flat staging/production/archived dirs are all redirected via monkeypatched
    mlops_settings properties so nothing touches the real models/ tree.
    """
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
    return ModelRegistry(registry_dir=reg_dir)


def test_register_model_creates_version_1(tmp_path, registry):
    source = _make_source_pkl(tmp_path)
    record = registry.register_model(
        "occupancy_prophet", source, metrics={"mae": 1.0}, dataset_version="hash1"
    )
    assert record.version == 1
    assert record.stage == ModelStage.STAGING
    assert record.model_name == "occupancy_prophet"
    assert record.metrics == {"mae": 1.0}


def test_register_model_increments_version(tmp_path, registry):
    source = _make_source_pkl(tmp_path)
    registry.register_model("occupancy_prophet", source, metrics={}, dataset_version="v1")
    record2 = registry.register_model("occupancy_prophet", source, metrics={}, dataset_version="v2")
    assert record2.version == 2


def test_list_versions(tmp_path, registry):
    source = _make_source_pkl(tmp_path)
    registry.register_model("occupancy_prophet", source, metrics={}, dataset_version="v1")
    registry.register_model("occupancy_prophet", source, metrics={}, dataset_version="v2")
    versions = registry.list_versions("occupancy_prophet")
    assert len(versions) == 2
    assert all(isinstance(v, ModelRecord) for v in versions)


def test_get_record_latest_and_by_version(tmp_path, registry):
    source = _make_source_pkl(tmp_path)
    registry.register_model("occupancy_prophet", source, metrics={"mae": 2.0}, dataset_version="v1")
    registry.register_model("occupancy_prophet", source, metrics={"mae": 1.0}, dataset_version="v2")

    latest = registry.get_record("occupancy_prophet")
    assert latest.version == 2

    v1 = registry.get_record("occupancy_prophet", version=1)
    assert v1.metrics == {"mae": 2.0}


def test_get_record_missing_raises(registry):
    with pytest.raises(FileNotFoundError):
        registry.get_record("no_such_model")


def test_get_record_missing_version_raises(tmp_path, registry):
    source = _make_source_pkl(tmp_path)
    registry.register_model("occupancy_prophet", source, metrics={}, dataset_version="v1")
    with pytest.raises(FileNotFoundError):
        registry.get_record("occupancy_prophet", version=99)


def test_promote_to_production_and_archives_previous(tmp_path, registry):
    source = _make_source_pkl(tmp_path)
    registry.register_model("occupancy_prophet", source, metrics={}, dataset_version="v1")
    registry.register_model("occupancy_prophet", source, metrics={}, dataset_version="v2")

    registry.promote("occupancy_prophet", 1, ModelStage.PRODUCTION)
    rec1 = registry.get_record("occupancy_prophet", version=1)
    assert rec1.stage == ModelStage.PRODUCTION

    # Promoting version 2 to production should archive version 1.
    registry.promote("occupancy_prophet", 2, ModelStage.PRODUCTION)
    rec1_after = registry.get_record("occupancy_prophet", version=1)
    rec2_after = registry.get_record("occupancy_prophet", version=2)
    assert rec1_after.stage == ModelStage.ARCHIVED
    assert rec2_after.stage == ModelStage.PRODUCTION


def test_promote_missing_version_raises(registry):
    with pytest.raises(FileNotFoundError):
        registry.promote("occupancy_prophet", 1, ModelStage.PRODUCTION)


def test_rollback_to_archived_version(tmp_path, registry):
    source = _make_source_pkl(tmp_path)
    registry.register_model("occupancy_prophet", source, metrics={}, dataset_version="v1")
    registry.register_model("occupancy_prophet", source, metrics={}, dataset_version="v2")
    registry.promote("occupancy_prophet", 1, ModelStage.PRODUCTION)
    registry.promote("occupancy_prophet", 2, ModelStage.PRODUCTION)  # archives v1

    rolled_back = registry.rollback("occupancy_prophet")
    assert rolled_back.version == 1
    assert rolled_back.stage == ModelStage.PRODUCTION

    # v2 should now be archived (it was the previous production).
    v2 = registry.get_record("occupancy_prophet", version=2)
    assert v2.stage == ModelStage.ARCHIVED


def test_rollback_explicit_version(tmp_path, registry):
    source = _make_source_pkl(tmp_path)
    registry.register_model("occupancy_prophet", source, metrics={}, dataset_version="v1")
    registry.register_model("occupancy_prophet", source, metrics={}, dataset_version="v2")
    registry.promote("occupancy_prophet", 2, ModelStage.PRODUCTION)

    rolled_back = registry.rollback("occupancy_prophet", to_version=1)
    assert rolled_back.version == 1
    assert rolled_back.stage == ModelStage.PRODUCTION


def test_rollback_no_archived_raises(tmp_path, registry):
    source = _make_source_pkl(tmp_path)
    registry.register_model("occupancy_prophet", source, metrics={}, dataset_version="v1")
    with pytest.raises(ValueError):
        registry.rollback("occupancy_prophet")


def test_archive_model(tmp_path, registry):
    source = _make_source_pkl(tmp_path)
    registry.register_model("occupancy_prophet", source, metrics={}, dataset_version="v1")
    registry.promote("occupancy_prophet", 1, ModelStage.PRODUCTION)

    archived = registry.archive_model("occupancy_prophet", 1)
    assert archived.stage == ModelStage.ARCHIVED


def test_load_latest_and_load_staging(tmp_path, registry, monkeypatch):
    source = _make_source_pkl(tmp_path)
    registry.register_model("occupancy_prophet", source, metrics={}, dataset_version="v1")
    registry.promote("occupancy_prophet", 1, ModelStage.STAGING)

    model = registry.load_latest("occupancy_prophet")
    assert model.model == "fake-estimator"

    staged = registry.load_staging("occupancy_prophet")
    assert staged.model == "fake-estimator"


def test_load_staging_missing_raises(registry):
    with pytest.raises(FileNotFoundError):
        registry.load_staging("occupancy_prophet")


def test_load_production_from_registry(tmp_path, registry):
    source = _make_source_pkl(tmp_path)
    registry.register_model("occupancy_prophet", source, metrics={}, dataset_version="v1")
    registry.promote("occupancy_prophet", 1, ModelStage.PRODUCTION)

    model = registry.load_production("occupancy_prophet")
    assert model.model == "fake-estimator"


def test_load_production_legacy_fallback(tmp_path, registry, monkeypatch):
    """When the registry has no entries at all for a model_name, load_production
    must fall back to settings.model_dir_path/<name>.pkl (Phase 1-5 compatibility).
    """
    from src.config.settings import settings as core_settings

    legacy_dir = tmp_path / "legacy_models"
    legacy_dir.mkdir(parents=True, exist_ok=True)
    legacy_path = legacy_dir / "occupancy_prophet.pkl"
    joblib.dump(
        {"model": "legacy-estimator", "encoders": {}, "scaler": None, "feature_names": []},
        legacy_path,
    )

    monkeypatch.setattr(
        type(core_settings), "model_dir_path", property(lambda self: legacy_dir)
    )

    # No registry entries registered at all for this model_name.
    model = registry.load_production("occupancy_prophet")
    assert model.model == "legacy-estimator"


def test_load_production_raises_when_nothing_found(tmp_path, registry, monkeypatch):
    from src.config.settings import settings as core_settings

    empty_dir = tmp_path / "no_legacy_here"
    empty_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        type(core_settings), "model_dir_path", property(lambda self: empty_dir)
    )

    with pytest.raises(FileNotFoundError):
        registry.load_production("occupancy_prophet")
