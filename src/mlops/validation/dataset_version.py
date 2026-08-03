import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.mlops.config.mlops_settings import mlops_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class DatasetVersion:
    source: str
    row_count: int
    feature_count: int
    content_hash: str
    generated_at: str  # ISO timestamp


def compute_dataset_version(df: pd.DataFrame, source: str) -> DatasetVersion:
    """Hash a dataframe's content (rows + index) to a stable sha256 hex digest."""
    hashed = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    content_hash = hashlib.sha256(hashed).hexdigest()
    return DatasetVersion(
        source=source,
        row_count=len(df),
        feature_count=df.shape[1],
        content_hash=content_hash,
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def save_dataset_version(version: DatasetVersion, module_name: str) -> Path:
    """Write reports/dataset_versions/{module}_latest.json and a timestamped copy."""
    payload = asdict(version)
    versions_dir = mlops_settings.dataset_versions_dir_path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    latest_path = versions_dir / f"{module_name}_latest.json"
    timestamped_path = versions_dir / f"{module_name}_{timestamp}.json"

    latest_path.write_text(json.dumps(payload, indent=2))
    timestamped_path.write_text(json.dumps(payload, indent=2))
    logger.info("Saved dataset version for module=%s hash=%s", module_name, version.content_hash)
    return latest_path


def load_latest_dataset_version(module_name: str) -> DatasetVersion | None:
    """Read {module}_latest.json if present, else None."""
    latest_path = mlops_settings.dataset_versions_dir_path / f"{module_name}_latest.json"
    if not latest_path.exists():
        return None
    payload = json.loads(latest_path.read_text())
    return DatasetVersion(**payload)


def has_dataset_changed(module_name: str, current: DatasetVersion) -> bool:
    """True if no previous version exists, or the content hash differs."""
    previous = load_latest_dataset_version(module_name)
    if previous is None:
        return True
    return previous.content_hash != current.content_hash
