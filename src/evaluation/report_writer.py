import json
from datetime import datetime, timezone

from src.config.settings import settings


def write_report(module_name: str, metrics: dict, extra: dict | None = None) -> None:
    payload = {
        "module": module_name,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "extra": extra or {},
    }
    reports_dir = settings.reports_dir_path
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    timestamped_path = reports_dir / f"{module_name}_{timestamp}.json"
    latest_path = reports_dir / f"latest_{module_name}.json"

    timestamped_path.write_text(json.dumps(payload, indent=2))
    latest_path.write_text(json.dumps(payload, indent=2))
