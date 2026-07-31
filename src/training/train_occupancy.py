import argparse
import datetime as dt

from src.pipelines.occupancy_pipeline import OccupancyPipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train occupancy forecasting models")
    parser.add_argument("--branch-id", type=int, required=True)
    parser.add_argument("--start-date", type=str, default="2023-01-01")
    parser.add_argument(
        "--end-date", type=str, default=dt.date.today().isoformat()
    )
    args = parser.parse_args()

    pipeline = OccupancyPipeline(
        branch_id=args.branch_id, start_date=args.start_date, end_date=args.end_date
    )
    metrics = pipeline.run()
    logger.info("Occupancy training complete: %s", metrics)


if __name__ == "__main__":
    main()
