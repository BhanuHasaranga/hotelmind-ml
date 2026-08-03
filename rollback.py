"""CLI: python rollback.py --model occupancy [--variant xgboost] [--to-version N]

Rolls back a domain's Production model to a previous registry version (the most
recent Archived version by default, or an explicit --to-version).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from promote import resolve_model_name  # noqa: E402
from src.mlops.registry.model_registry import ModelRegistry  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)


def main() -> int:
    parser = argparse.ArgumentParser(description="Roll back a domain's Production model")
    parser.add_argument("--model", required=True, help="Domain name, e.g. occupancy/pricing/staffing/churn/restaurant")
    parser.add_argument("--variant", default=None, help="Optional variant, e.g. xgboost/prophet/random_forest/breakfast/lunch/dinner")
    parser.add_argument("--to-version", type=int, default=None, dest="to_version", help="Explicit version to roll back to")
    args = parser.parse_args()

    model_name = resolve_model_name(args.model, args.variant)
    registry = ModelRegistry()

    try:
        record = registry.rollback(model_name, to_version=args.to_version)
    except ValueError as exc:
        print(f"Rollback failed for model_name={model_name!r}: {exc}")
        return 1

    print(
        f"Rolled back model_name={model_name!r} to version={record.version} "
        f"(stage={record.stage.value}, metrics={record.metrics})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
