"""One-time seed script (M20): registers each of the 9 existing flat
models/*.pkl files into the ModelRegistry, then promotes each to Production.

Idempotent: if a model_name already has registry entries (list_versions
returns non-empty), it is skipped with a warning rather than re-registered,
so re-running this script after the registry has real training history does
not create duplicate/no-op versions.

Usage:
    python scripts/seed_registry_from_legacy.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.settings import settings  # noqa: E402
from src.mlops.registry.model_registry import ModelRegistry, ModelStage  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

# The 9 existing flat model files, per the Phase 4 release report.
LEGACY_MODEL_NAMES = [
    "occupancy_prophet",
    "occupancy_xgboost",
    "pricing_xgboost",
    "restaurant_breakfast",
    "restaurant_lunch",
    "restaurant_dinner",
    "staffing_regression",
    "churn_random_forest",
    "churn_xgboost",
]


def seed_registry() -> list[dict]:
    registry = ModelRegistry()
    results: list[dict] = []

    for model_name in LEGACY_MODEL_NAMES:
        legacy_path = settings.model_dir_path / f"{model_name}.pkl"

        if not legacy_path.exists():
            results.append(
                {"model_name": model_name, "status": "skipped (legacy file not found)"}
            )
            logger.warning("Legacy file not found for model_name=%s at %s", model_name, legacy_path)
            continue

        # Idempotency guard: skip if this model_name already has registry entries.
        existing_versions = registry.list_versions(model_name)
        if existing_versions:
            results.append(
                {
                    "model_name": model_name,
                    "status": f"skipped (already registered, {len(existing_versions)} version(s))",
                }
            )
            logger.warning(
                "model_name=%s already has %d registry version(s); skipping re-registration",
                model_name,
                len(existing_versions),
            )
            continue

        record = registry.register_model(
            model_name=model_name,
            source_path=legacy_path,
            metrics={},
            dataset_version="unknown-legacy",
        )
        registry.promote(model_name, record.version, ModelStage.PRODUCTION)

        results.append(
            {
                "model_name": model_name,
                "status": "registered + promoted to production",
                "version": record.version,
            }
        )
        logger.info(
            "Seeded model_name=%s version=%d from legacy path=%s",
            model_name,
            record.version,
            legacy_path,
        )

    return results


def _print_summary(results: list[dict]) -> None:
    name_width = max(len(r["model_name"]) for r in results) + 2
    print("\nSeed registry from legacy .pkl files -- summary")
    print("=" * 70)
    print(f"{'model_name':<{name_width}}{'status'}")
    print("-" * 70)
    for r in results:
        print(f"{r['model_name']:<{name_width}}{r['status']}")
    print("=" * 70)

    registered = sum(1 for r in results if r["status"].startswith("registered"))
    skipped = len(results) - registered
    print(f"Registered+promoted: {registered}   Skipped: {skipped}   Total: {len(results)}")


def main() -> int:
    results = seed_registry()
    _print_summary(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
