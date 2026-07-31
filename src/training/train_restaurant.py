import argparse
import datetime as dt

from src.pipelines.restaurant_pipeline import RestaurantPipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train restaurant demand models")
    parser.add_argument("--branch-id", type=int, required=True)
    parser.add_argument("--start-date", type=str, default="2023-01-01")
    parser.add_argument("--end-date", type=str, default=dt.date.today().isoformat())
    args = parser.parse_args()

    pipeline = RestaurantPipeline(
        branch_id=args.branch_id, start_date=args.start_date, end_date=args.end_date
    )
    metrics = pipeline.run()
    logger.info("Restaurant training complete: %s", metrics)


if __name__ == "__main__":
    main()
