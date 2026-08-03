"""Tests for src/mlops/monitoring/drift_detector.py.

`evidently` cannot be installed on this host (Python 3.14, no cmake). We stub
`evidently.report` / `evidently.metric_preset` in sys.modules to verify
generate_data_drift_report()'s call sequence (Report(...).run(...).save_html(...))
without needing evidently installed, per the M16 spec. get_latest_drift_score
is pure JSON file I/O and is tested for real, no mocking needed.
"""

import json
import sys
import types
from unittest.mock import MagicMock

import pandas as pd
import pytest

from src.mlops.monitoring.drift_detector import get_latest_drift_score


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


def test_get_latest_drift_score_no_reports_returns_none():
    assert get_latest_drift_score("occupancy") is None


def test_get_latest_drift_score_reads_latest_json(isolated_drift_dir):
    old = isolated_drift_dir / "occupancy_data_drift_20260101T000000Z.json"
    new = isolated_drift_dir / "occupancy_data_drift_20260102T000000Z.json"
    old.write_text(json.dumps({"drift_score": 0.1}))
    new.write_text(json.dumps({"drift_score": 0.42}))

    assert get_latest_drift_score("occupancy") == 0.42


def test_get_latest_drift_score_handles_missing_score_key(isolated_drift_dir):
    path = isolated_drift_dir / "occupancy_data_drift_20260101T000000Z.json"
    path.write_text(json.dumps({"some_other_key": 1}))
    assert get_latest_drift_score("occupancy") is None


def test_get_latest_drift_score_handles_corrupt_json(isolated_drift_dir):
    path = isolated_drift_dir / "occupancy_data_drift_20260101T000000Z.json"
    path.write_text("{not valid json")
    assert get_latest_drift_score("occupancy") is None


@pytest.fixture
def fake_evidently(monkeypatch):
    """Install fake evidently.report / evidently.metric_preset modules so
    generate_data_drift_report's `from evidently... import ...` (a function-local
    import) succeeds, and reload drift_detector against a clean state."""
    fake_report_instance = MagicMock()
    fake_report_instance.as_dict.return_value = {
        "metrics": [
            {
                "metric": "DataDriftTable",
                "result": {
                    "share_of_drifted_columns": 0.25,
                    "dataset_drift": True,
                    "number_of_drifted_columns": 2,
                },
            }
        ]
    }
    fake_report_cls = MagicMock(return_value=fake_report_instance)
    fake_report_instance.run.return_value = None

    report_module = types.ModuleType("evidently.report")
    report_module.Report = fake_report_cls
    metric_preset_module = types.ModuleType("evidently.metric_preset")
    metric_preset_module.DataDriftPreset = MagicMock()
    evidently_module = types.ModuleType("evidently")

    monkeypatch.setitem(sys.modules, "evidently", evidently_module)
    monkeypatch.setitem(sys.modules, "evidently.report", report_module)
    monkeypatch.setitem(sys.modules, "evidently.metric_preset", metric_preset_module)

    yield fake_report_cls, fake_report_instance


def test_generate_data_drift_report_calls_report_run_save_html(fake_evidently, isolated_drift_dir):
    fake_report_cls, fake_report_instance = fake_evidently
    from src.mlops.monitoring.drift_detector import generate_data_drift_report

    reference_df = pd.DataFrame({"a": [1, 2, 3]})
    current_df = pd.DataFrame({"a": [4, 5, 6]})

    html_path = generate_data_drift_report(reference_df, current_df, "occupancy")

    fake_report_cls.assert_called_once()
    fake_report_instance.run.assert_called_once_with(
        reference_data=reference_df, current_data=current_df
    )
    fake_report_instance.save_html.assert_called_once()
    assert html_path.exists() or str(html_path).endswith(".html")

    json_candidates = list(isolated_drift_dir.glob("occupancy_data_drift_*.json"))
    assert len(json_candidates) == 1
    summary = json.loads(json_candidates[0].read_text())
    assert summary["drift_score"] == 0.25
    assert summary["dataset_drift"] is True


def test_generate_data_drift_report_then_get_latest_drift_score(fake_evidently, isolated_drift_dir):
    from src.mlops.monitoring.drift_detector import generate_data_drift_report

    reference_df = pd.DataFrame({"a": [1, 2, 3]})
    current_df = pd.DataFrame({"a": [4, 5, 6]})
    generate_data_drift_report(reference_df, current_df, "pricing")

    assert get_latest_drift_score("pricing") == 0.25
