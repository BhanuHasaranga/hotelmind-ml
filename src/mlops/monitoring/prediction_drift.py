"""Pure-Python/pandas prediction drift logging and reporting (no evidently dependency,
runs fully on any host). Logs each domain's prediction payloads to JSONL, and compares
a recent window against a prior window to flag basic distributional shifts.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.mlops.config.mlops_settings import mlops_settings
from src.utils.logging import get_logger

logger = get_logger(__name__)


def _predictions_dir() -> Path:
    path = mlops_settings.drift_reports_dir_path / "predictions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def log_prediction_for_drift(domain: str, prediction_payload: dict) -> None:
    """Append a JSON line (with a timestamp) to reports/drift/predictions/{domain}.jsonl.
    Never raises -- any failure is caught and logged as a warning.
    """
    try:
        record = dict(prediction_payload)
        record["_logged_at"] = datetime.now(timezone.utc).isoformat()
        path = _predictions_dir() / f"{domain}.jsonl"
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")
    except Exception as exc:
        logger.warning("log_prediction_for_drift failed for domain=%s: %s", domain, exc)


def _numeric_fields(records: list[dict]) -> set[str]:
    fields: set[str] = set()
    for rec in records:
        for key, value in rec.items():
            if key == "_logged_at":
                continue
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                fields.add(key)
    return fields


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    return mean, variance**0.5


def generate_prediction_drift_report(domain: str, window_days: int = 7) -> Path | None:
    """Compare a recent window_days slice vs the prior window_days slice of logged
    predictions for `domain`, writing a JSON summary. Returns None (and logs a
    warning) if there isn't enough data to form two windows.
    """
    log_path = _predictions_dir() / f"{domain}.jsonl"
    if not log_path.exists():
        logger.warning("No prediction log found for domain=%s", domain)
        return None

    records = []
    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    parsed = []
    for rec in records:
        ts_raw = rec.get("_logged_at")
        if not ts_raw:
            continue
        try:
            ts = datetime.fromisoformat(ts_raw)
        except ValueError:
            continue
        parsed.append((ts, rec))

    if not parsed:
        logger.warning("No timestamped predictions found for domain=%s", domain)
        return None

    now = max(ts for ts, _ in parsed)
    recent_cutoff = now - timedelta(days=window_days)
    prior_cutoff = recent_cutoff - timedelta(days=window_days)

    recent = [rec for ts, rec in parsed if ts > recent_cutoff]
    prior = [rec for ts, rec in parsed if prior_cutoff < ts <= recent_cutoff]

    if len(recent) < 2 or len(prior) < 2:
        logger.warning(
            "Insufficient data for two windows for domain=%s (recent=%d prior=%d)",
            domain,
            len(recent),
            len(prior),
        )
        return None

    fields = _numeric_fields(recent) & _numeric_fields(prior)
    field_stats: dict[str, dict] = {}
    stable = True
    for field in sorted(fields):
        recent_vals = [rec[field] for rec in recent if field in rec]
        prior_vals = [rec[field] for rec in prior if field in rec]
        if len(recent_vals) < 2 or len(prior_vals) < 2:
            continue
        recent_mean, recent_std = _mean_std(recent_vals)
        prior_mean, prior_std = _mean_std(prior_vals)
        # Basic stability heuristic: mean shift > 50% of prior std (or > 20% relative) => unstable
        shift = abs(recent_mean - prior_mean)
        threshold = max(prior_std * 0.5, abs(prior_mean) * 0.2, 1e-9)
        field_stable = shift <= threshold
        stable = stable and field_stable
        field_stats[field] = {
            "recent_mean": recent_mean,
            "recent_std": recent_std,
            "prior_mean": prior_mean,
            "prior_std": prior_std,
            "stable": field_stable,
        }

    payload = {
        "domain": domain,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "window_days": window_days,
        "recent_count": len(recent),
        "prior_count": len(prior),
        "fields": field_stats,
        "stable": stable,
    }

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = mlops_settings.drift_reports_dir_path / f"{domain}_prediction_drift_{timestamp}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote prediction drift report for domain=%s to %s", domain, out_path)
    return out_path
