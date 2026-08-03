"""CLI: python promote.py --model occupancy [--variant prophet|xgboost] [--force]

Promotes a Staging model version to Production, but only if it's better than the
current Production model on its primary metric (unless --force is passed).
"""

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mlops.registry.model_registry import ModelRegistry, ModelStage  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

DOMAIN_TO_PRIMARY_MODEL = {
    "occupancy": "occupancy_prophet",
    "pricing": "pricing_xgboost",
    "staffing": "staffing_regression",
    "churn": "churn_xgboost",
}

# domain -> {variant: model_name}
DOMAIN_VARIANTS = {
    "occupancy": {"prophet": "occupancy_prophet", "xgboost": "occupancy_xgboost"},
    "churn": {"xgboost": "churn_xgboost", "random_forest": "churn_random_forest"},
    "restaurant": {"breakfast": "restaurant_breakfast", "lunch": "restaurant_lunch", "dinner": "restaurant_dinner"},
}

# primary_metric + direction per model_name
MODEL_METRIC_DIRECTION = {
    "occupancy_prophet": ("mae", True),
    "occupancy_xgboost": ("mae", True),
    "pricing_xgboost": ("mae", True),
    "restaurant_breakfast": ("mae", True),
    "restaurant_lunch": ("mae", True),
    "restaurant_dinner": ("mae", True),
    "staffing_regression": ("mae", True),
    "churn_xgboost": ("roc_auc", False),
    "churn_random_forest": ("roc_auc", False),
}


def resolve_model_name(domain: str, variant: str | None) -> str:
    if domain == "restaurant":
        if not variant or variant not in DOMAIN_VARIANTS["restaurant"]:
            raise SystemExit(
                "--variant is required for domain=restaurant, one of: breakfast, lunch, dinner"
            )
        return DOMAIN_VARIANTS["restaurant"][variant]

    if variant is not None:
        variants = DOMAIN_VARIANTS.get(domain, {})
        if variant not in variants:
            raise SystemExit(f"Unknown --variant={variant!r} for domain={domain!r}")
        return variants[variant]

    if domain not in DOMAIN_TO_PRIMARY_MODEL:
        raise SystemExit(f"Unknown domain={domain!r}. Known domains: {list(DOMAIN_TO_PRIMARY_MODEL)}")
    return DOMAIN_TO_PRIMARY_MODEL[domain]


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote a Staging model version to Production")
    parser.add_argument("--model", required=True, help="Domain name, e.g. occupancy/pricing/staffing/churn/restaurant")
    parser.add_argument("--variant", default=None, help="Optional variant, e.g. xgboost/prophet/random_forest/breakfast/lunch/dinner")
    parser.add_argument("--force", action="store_true", help="Promote even if not better than current production")
    args = parser.parse_args()

    model_name = resolve_model_name(args.model, args.variant)
    registry = ModelRegistry()

    versions = registry.list_versions(model_name)
    staging_versions = [r for r in versions if r.stage == ModelStage.STAGING]
    if not staging_versions:
        print(f"No Staging version found for model_name={model_name!r}. Nothing to promote.")
        return 1

    staging_record = max(staging_versions, key=lambda r: r.version)

    production_versions = [r for r in versions if r.stage == ModelStage.PRODUCTION]
    primary_metric, lower_is_better = MODEL_METRIC_DIRECTION.get(model_name, ("mae", True))

    should_promote = args.force
    if not should_promote:
        if not production_versions:
            should_promote = True
        else:
            current_production = max(production_versions, key=lambda r: r.version)
            current_value = current_production.metrics.get(primary_metric)
            new_value = staging_record.metrics.get(primary_metric)
            if current_value is None or new_value is None:
                should_promote = True
            elif lower_is_better:
                should_promote = new_value < current_value
            else:
                should_promote = new_value > current_value

    if not should_promote:
        print(
            f"Staging version={staging_record.version} of model_name={model_name!r} is not "
            f"better than current Production on {primary_metric!r}. Use --force to override. No-op."
        )
        return 1

    record = registry.promote(model_name, staging_record.version, ModelStage.PRODUCTION)
    print(
        f"Promoted model_name={model_name!r} version={record.version} to Production "
        f"(metrics={record.metrics})."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
