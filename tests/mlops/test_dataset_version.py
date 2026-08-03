"""Tests for src/mlops/validation/dataset_version.py."""

import pandas as pd
import pytest

from src.mlops.validation.dataset_version import (
    compute_dataset_version,
    has_dataset_changed,
    load_latest_dataset_version,
    save_dataset_version,
)


@pytest.fixture(autouse=True)
def isolated_versions_dir(tmp_path, monkeypatch):
    from src.mlops.config import mlops_settings as settings_module

    versions_dir = tmp_path / "dataset_versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        type(settings_module.mlops_settings),
        "dataset_versions_dir_path",
        property(lambda self: versions_dir),
    )
    return versions_dir


def _df(n=3):
    return pd.DataFrame({"a": range(n), "b": [f"x{i}" for i in range(n)]})


def test_compute_dataset_version_stable_hash():
    df = _df()
    v1 = compute_dataset_version(df, source="occupancy")
    v2 = compute_dataset_version(df.copy(), source="occupancy")
    assert v1.content_hash == v2.content_hash
    assert v1.row_count == 3
    assert v1.feature_count == 2


def test_compute_dataset_version_changes_with_data():
    df1 = _df(3)
    df2 = _df(4)
    v1 = compute_dataset_version(df1, source="occupancy")
    v2 = compute_dataset_version(df2, source="occupancy")
    assert v1.content_hash != v2.content_hash


def test_save_and_load_latest_dataset_version():
    df = _df()
    version = compute_dataset_version(df, source="occupancy")
    save_dataset_version(version, "occupancy")

    loaded = load_latest_dataset_version("occupancy")
    assert loaded is not None
    assert loaded.content_hash == version.content_hash
    assert loaded.source == "occupancy"


def test_load_latest_dataset_version_missing_returns_none():
    assert load_latest_dataset_version("does_not_exist_module") is None


def test_has_dataset_changed_true_when_no_previous():
    df = _df()
    version = compute_dataset_version(df, source="pricing")
    assert has_dataset_changed("pricing", version) is True


def test_has_dataset_changed_false_when_same():
    df = _df()
    version = compute_dataset_version(df, source="pricing")
    save_dataset_version(version, "pricing")
    version_again = compute_dataset_version(df.copy(), source="pricing")
    assert has_dataset_changed("pricing", version_again) is False


def test_has_dataset_changed_true_when_different():
    df1 = _df(3)
    version1 = compute_dataset_version(df1, source="pricing")
    save_dataset_version(version1, "pricing")

    df2 = _df(10)
    version2 = compute_dataset_version(df2, source="pricing")
    assert has_dataset_changed("pricing", version2) is True


def test_save_dataset_version_writes_latest_and_timestamped(isolated_versions_dir):
    df = _df()
    version = compute_dataset_version(df, source="staffing")
    latest_path = save_dataset_version(version, "staffing")

    assert latest_path.name == "staffing_latest.json"
    assert latest_path.exists()

    timestamped = list(isolated_versions_dir.glob("staffing_*.json"))
    # includes both staffing_latest.json and the timestamped copy
    assert len(timestamped) >= 2
