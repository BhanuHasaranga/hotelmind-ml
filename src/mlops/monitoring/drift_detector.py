"""Data drift detection via Evidently AI (evidently>=0.4,<0.5).

IMPORTANT: This module requires `evidently`, which has no prebuilt wheel for
Python 3.14 and fails to build from source without cmake on some hosts. It is
expected to fail to import on such hosts -- verification of this module happens
in Docker against the pinned python:3.12-slim base image, not on those hosts.
Do not "fix" an ImportError here by removing the evidently dependency.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.mlops.config.mlops_settings import mlops_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def generate_data_drift_report(
    reference_df: pd.DataFrame, current_df: pd.DataFrame, module_name: str
) -> Path:
    """Run Evidently's DataDriftPreset comparing `reference_df` (last successful
    training snapshot) against `current_df` (latest feature data), writing an HTML
    report plus a JSON drift-score summary to mlops_settings.drift_reports_dir_path.

    Returns the path to the HTML report.
    """
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    reports_dir = mlops_settings.drift_reports_dir_path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    html_path = reports_dir / f"{module_name}_data_drift_{timestamp}.html"
    json_path = reports_dir / f"{module_name}_data_drift_{timestamp}.json"

    report.save_html(str(html_path))

    report_dict = report.as_dict()
    drift_metric = next(
        (
            m
            for m in report_dict.get("metrics", [])
            if m.get("metric") == "DataDriftTable"
        ),
        None,
    )
    if drift_metric is None:
        # Fall back to searching for any metric result containing drift share info.
        drift_metric = next(
            (
                m
                for m in report_dict.get("metrics", [])
                if "share_of_drifted_columns" in m.get("result", {})
            ),
            {},
        )

    result = drift_metric.get("result", {}) if drift_metric else {}
    drift_share = result.get("share_of_drifted_columns")
    dataset_drift = result.get("dataset_drift")
    number_of_drifted_columns = result.get("number_of_drifted_columns")

    summary = {
        "module": module_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "drift_score": drift_share,
        "share_of_drifted_columns": drift_share,
        "dataset_drift": dataset_drift,
        "number_of_drifted_columns": number_of_drifted_columns,
        "html_report": str(html_path),
    }
    json_path.write_text(json.dumps(summary, indent=2))
    logger.info(
        "Wrote data drift report for module=%s drift_score=%s to %s",
        module_name,
        drift_share,
        html_path,
    )
    return html_path


def get_latest_drift_score(module_name: str) -> float | None:
    """Return the most recent {module_name}_data_drift_*.json summary's overall
    drift score/share, or None if no report exists yet."""
    reports_dir = mlops_settings.drift_reports_dir_path
    candidates = sorted(reports_dir.glob(f"{module_name}_data_drift_*.json"))
    if not candidates:
        return None
    latest = candidates[-1]
    try:
        payload = json.loads(latest.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Failed to read drift report %s: %s", latest, exc)
        return None
    score = payload.get("drift_score")
    return float(score) if score is not None else None
