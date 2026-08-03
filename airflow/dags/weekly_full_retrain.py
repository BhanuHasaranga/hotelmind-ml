"""Weekly unconditional full retrain DAG (M6).

Unlike daily_training.py (which only retrains domains with a drift/accuracy
breach), this DAG forces a full retrain of all 5 domains every week regardless
of signal, plus registers a versioned snapshot of the sentiment lexicon
(decision #4: sentiment has no trainable model, so a lexicon/config snapshot
is registered as a lightweight pseudo-model so all 6 domains are registry
citizens).
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta

from airflow.operators.python import PythonOperator

from airflow import DAG

default_args = {
    "owner": "hotelmind-mlops",
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


def _run_domain(domain: str, pipeline_cls: type) -> bool:
    from src.utils.logging import get_logger

    logger = get_logger(__name__)
    try:
        pipeline = pipeline_cls()
        summary = pipeline.run_with_mlops()
        logger.info("weekly_full_retrain: domain=%s summary=%s", domain, summary)
        return True
    except Exception as exc:
        logger.warning("weekly_full_retrain: domain=%s failed: %s", domain, exc)
        return False


def _run_weekly_full_retrain() -> None:
    from src.mlops.pipelines.churn_mlops_pipeline import ChurnMLOpsPipeline
    from src.mlops.pipelines.occupancy_mlops_pipeline import OccupancyMLOpsPipeline
    from src.mlops.pipelines.pricing_mlops_pipeline import PricingMLOpsPipeline
    from src.mlops.pipelines.restaurant_mlops_pipeline import RestaurantMLOpsPipeline
    from src.mlops.pipelines.staffing_mlops_pipeline import StaffingMLOpsPipeline
    from src.utils.logging import get_logger

    logger = get_logger(__name__)

    domains: dict[str, type] = {
        "occupancy": OccupancyMLOpsPipeline,
        "pricing": PricingMLOpsPipeline,
        "restaurant": RestaurantMLOpsPipeline,
        "staffing": StaffingMLOpsPipeline,
        "churn": ChurnMLOpsPipeline,
    }

    results = {domain: _run_domain(domain, cls) for domain, cls in domains.items()}
    logger.info("weekly_full_retrain complete: %s", results)


def _register_sentiment_lexicon_snapshot() -> None:
    """Register a versioned JSON snapshot of the sentiment lexicon/config as a
    pseudo-model in the ModelRegistry (decision #4), so the sentiment domain has
    a real registry citizen even though it has no trainable model.
    """
    from src.mlops.registry.model_registry import ModelRegistry
    from src.utils.logging import get_logger

    logger = get_logger(__name__)

    try:
        from genai.reviews.pipeline import _NEGATIVE_WORDS, _POSITIVE_WORDS
        from genai.reviews.synthetic_reviews import COMPLAINT_CATEGORIES
        from src.mlops.registry.model_registry import ModelStage

        lexicon_snapshot = {
            "positive_words": sorted(_POSITIVE_WORDS),
            "negative_words": sorted(_NEGATIVE_WORDS),
            "complaint_categories": COMPLAINT_CATEGORIES,
        }
        payload = json.dumps(lexicon_snapshot, sort_keys=True).encode("utf-8")
        content_hash = hashlib.sha256(payload).hexdigest()

        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_path = Path(tmp_dir) / "sentiment_lexicon.json"
            snapshot_path.write_text(json.dumps(lexicon_snapshot, indent=2, sort_keys=True))

            registry = ModelRegistry()
            vocab_size = len(lexicon_snapshot["positive_words"]) + len(
                lexicon_snapshot["negative_words"]
            )
            record = registry.register_model(
                model_name="sentiment_lexicon",
                source_path=snapshot_path,
                metrics={"vocab_size": float(vocab_size)},
                dataset_version=content_hash,
                mlflow_run_id=None,
            )
            registry.promote(record.model_name, record.version, ModelStage.STAGING)
        logger.info(
            "Registered sentiment_lexicon snapshot version=%s hash=%s",
            record.version,
            content_hash,
        )
    except Exception as exc:
        logger.warning("weekly_full_retrain: sentiment lexicon snapshot failed: %s", exc)


with DAG(
    dag_id="weekly_full_retrain",
    description="Unconditional full retrain of all 5 domains + sentiment lexicon snapshot",
    default_args=default_args,
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["hotelmind", "mlops", "training"],
) as dag:
    full_retrain = PythonOperator(
        task_id="full_retrain_all_domains",
        python_callable=_run_weekly_full_retrain,
    )

    sentiment_snapshot = PythonOperator(
        task_id="register_sentiment_lexicon_snapshot",
        python_callable=_register_sentiment_lexicon_snapshot,
    )

    full_retrain >> sentiment_snapshot
