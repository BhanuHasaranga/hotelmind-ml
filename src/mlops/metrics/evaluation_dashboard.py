"""Model evaluation dashboard: aggregates per-model evaluation metrics into a single
JSON dashboard file (latest, overwritten) plus a timestamped history copy, mirroring
src/evaluation/report_writer.py's json.dump style/indent convention.
"""

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from src.mlops.config.mlops_settings import PROJECT_ROOT
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ModelEvalRecord:
    model_name: str
    version: int
    mae: float | None = None
    rmse: float | None = None
    mape: float | None = None
    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    roc_auc: float | None = None
    training_duration_s: float | None = None
    evaluated_at: str = field(default_factory=_now_iso)


def build_dashboard(records: list[ModelEvalRecord]) -> dict:
    return {
        "generated_at": _now_iso(),
        "models": [asdict(r) for r in records],
    }


def _dashboards_dir() -> Path:
    path = PROJECT_ROOT / "dashboards"
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_dashboard(records: list[ModelEvalRecord]) -> Path:
    """Write dashboards/model_evaluation_dashboard.json (latest, overwritten) and a
    timestamped history copy. Returns the path to the latest file."""
    payload = build_dashboard(records)
    dashboards_dir = _dashboards_dir()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    latest_path = dashboards_dir / "model_evaluation_dashboard.json"
    timestamped_path = dashboards_dir / f"model_evaluation_dashboard_{timestamp}.json"

    latest_path.write_text(json.dumps(payload, indent=2))
    timestamped_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote model evaluation dashboard with %d records to %s", len(records), latest_path)
    return latest_path
