from src.pipelines.churn_pipeline import ChurnPipeline
from src.utils.logging import get_logger

logger = get_logger(__name__)


def main() -> None:
    pipeline = ChurnPipeline()
    metrics = pipeline.run()
    logger.info("Churn training complete: %s", metrics)


if __name__ == "__main__":
    main()
