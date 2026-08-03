"""Retrains every Phase 4 model and regenerates every report.

Runs, in order:
  1. Feature engineering (data/features/*.parquet + reports/features/*.md)
  2. Occupancy, Pricing, Restaurant, Staffing, Churn training
     (models/*.pkl + reports/latest_<module>.json)
  3. Occupancy forecast report (reports/models/occupancy_forecast.csv)
  4. Cross-domain comparison/leaderboard (reports/models/{comparison,leaderboard}.md)

Uses the canonical dataset's real date range (2015-07-01 to 2017-09-13) —
see reports/final_phase4/known_limitations.md for why the training CLIs'
own --start-date/--end-date defaults don't work for this dataset.

Usage:
    python scripts/train_all.py
    python scripts/train_all.py --branch-id 2
"""

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mlops.observability.logging_config import configure_mlops_logging  # noqa: E402

configure_mlops_logging("training", "training.log")

from src.pipelines.churn_pipeline import ChurnPipeline  # noqa: E402
from src.pipelines.feature_engineering import run as run_feature_engineering  # noqa: E402
from src.pipelines.generate_occupancy_report import run as run_occupancy_report  # noqa: E402
from src.pipelines.ml_reports import write_comparison_and_leaderboard  # noqa: E402
from src.pipelines.occupancy_pipeline import OccupancyPipeline  # noqa: E402
from src.pipelines.pricing_pipeline import PricingPipeline  # noqa: E402
from src.pipelines.restaurant_pipeline import RestaurantPipeline  # noqa: E402
from src.pipelines.staffing_pipeline import StaffingPipeline  # noqa: E402
from src.utils.logging import get_logger  # noqa: E402

logger = get_logger(__name__)

CANONICAL_START_DATE = "2015-07-01"
CANONICAL_END_DATE = "2017-09-13"


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain every Phase 4 model and regenerate reports")
    parser.add_argument("--branch-id", type=int, default=1)
    parser.add_argument("--start-date", type=str, default=CANONICAL_START_DATE)
    parser.add_argument("--end-date", type=str, default=CANONICAL_END_DATE)
    parser.add_argument(
        "--mlops",
        action="store_true",
        default=False,
        help="Use the Phase 6 MLOps pipeline classes (MLflow tracking + registry "
        "registration/auto-promotion) instead of the plain pipelines.",
    )
    args = parser.parse_args()

    start = time.perf_counter()
    results: dict[str, object] = {}

    logger.info("=== 1/3: Feature engineering ===")
    run_feature_engineering()

    logger.info("=== 2/3: Training all 5 domains (branch_id=%d, %s..%s) ===", args.branch_id, args.start_date, args.end_date)

    if args.mlops:
        from src.mlops.pipelines.churn_mlops_pipeline import ChurnMLOpsPipeline
        from src.mlops.pipelines.occupancy_mlops_pipeline import OccupancyMLOpsPipeline
        from src.mlops.pipelines.pricing_mlops_pipeline import PricingMLOpsPipeline
        from src.mlops.pipelines.restaurant_mlops_pipeline import RestaurantMLOpsPipeline
        from src.mlops.pipelines.staffing_mlops_pipeline import StaffingMLOpsPipeline

        results["occupancy"] = OccupancyMLOpsPipeline(
            branch_id=args.branch_id, start_date=args.start_date, end_date=args.end_date
        ).run_with_mlops()
        logger.info("occupancy: %s", results["occupancy"])

        results["pricing"] = PricingMLOpsPipeline(
            branch_id=args.branch_id, start_date=args.start_date, end_date=args.end_date
        ).run_with_mlops()
        logger.info("pricing: %s", results["pricing"])

        results["restaurant"] = RestaurantMLOpsPipeline(
            branch_id=args.branch_id, start_date=args.start_date, end_date=args.end_date
        ).run_with_mlops()
        logger.info("restaurant: %s", results["restaurant"])

        results["staffing"] = StaffingMLOpsPipeline(
            branch_id=args.branch_id, start_date=args.start_date, end_date=args.end_date
        ).run_with_mlops()
        logger.info("staffing: %s", results["staffing"])

        results["churn"] = ChurnMLOpsPipeline().run_with_mlops()
        logger.info("churn: %s", results["churn"])
    else:
        results["occupancy"] = OccupancyPipeline(
            branch_id=args.branch_id, start_date=args.start_date, end_date=args.end_date
        ).run()
        logger.info("occupancy: %s", results["occupancy"])

        results["pricing"] = PricingPipeline(
            branch_id=args.branch_id, start_date=args.start_date, end_date=args.end_date
        ).run()
        logger.info("pricing: %s", results["pricing"])

        results["restaurant"] = RestaurantPipeline(
            branch_id=args.branch_id, start_date=args.start_date, end_date=args.end_date
        ).run()
        logger.info("restaurant: %s", results["restaurant"])

        results["staffing"] = StaffingPipeline(
            branch_id=args.branch_id, start_date=args.start_date, end_date=args.end_date
        ).run()
        logger.info("staffing: %s", results["staffing"])

        results["churn"] = ChurnPipeline().run()
        logger.info("churn: %s", results["churn"])

    logger.info("=== 3/3: Generating occupancy forecast + comparison/leaderboard reports ===")
    run_occupancy_report(branch_id=args.branch_id)
    write_comparison_and_leaderboard()

    elapsed = time.perf_counter() - start
    logger.info("train_all complete in %.1fs. Trained: %s", elapsed, list(results.keys()))


if __name__ == "__main__":
    main()
