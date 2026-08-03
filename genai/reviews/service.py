"""Orchestrates the review pipeline with a precompute-then-query pattern.

`analyze_reviews()` (triggered by `POST /reviews/analyze`) runs the full
pipeline and persists aggregated outputs to parquet under
`genai_settings.reviews_output_dir_path`. `get_summary/get_topics/
get_complaints/get_trends` (backing the `GET /reviews/*` endpoints) only
read the persisted outputs — they never re-run the (potentially expensive,
LLM-calling) pipeline inline in a request.
"""

from __future__ import annotations

import json

import pandas as pd

from genai.config.genai_settings import genai_settings
from genai.reviews.pipeline import run_pipeline
from genai.reviews.synthetic_reviews import ensure_synthetic_reviews
from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

_SUMMARY_FILE = "summary.json"
_KEYWORDS_FILE = "keywords.json"
_TOPICS_FILE = "topics.json"
_COMPLAINTS_FILE = "complaint_counts.parquet"
_CSAT_DAILY_FILE = "csat_daily.parquet"
_CSAT_WEEKLY_FILE = "csat_weekly.parquet"
_CSAT_MONTHLY_FILE = "csat_monthly.parquet"
_CSAT_HOTEL_FILE = "csat_hotel.parquet"
_TRENDS_FILE = "trends.json"


class ReviewsNotAnalyzedError(RuntimeError):
    """Raised when a GET endpoint is called before POST /reviews/analyze."""


def analyze_reviews(llm_provider=None, limit: int | None = None) -> dict:
    """Run the full pipeline and persist outputs. Returns a small manifest."""
    csv_path = ensure_synthetic_reviews()
    df = pd.read_csv(csv_path)
    if limit:
        df = df.head(limit)

    result = run_pipeline(df, llm_provider=llm_provider)
    out_dir = genai_settings.reviews_output_dir_path

    (out_dir / _SUMMARY_FILE).write_text(json.dumps({"summary": result["summary"]}), encoding="utf-8")
    (out_dir / _KEYWORDS_FILE).write_text(json.dumps({"keywords": result["keywords"]}), encoding="utf-8")
    (out_dir / _TOPICS_FILE).write_text(json.dumps({"topics": result["topics"]}), encoding="utf-8")
    (out_dir / _TRENDS_FILE).write_text(json.dumps({"trend_by_hotel": result["trend_by_hotel"]}), encoding="utf-8")

    result["complaint_counts"].to_parquet(out_dir / _COMPLAINTS_FILE, index=False)
    result["csat_daily"].to_parquet(out_dir / _CSAT_DAILY_FILE, index=False)
    result["csat_weekly"].to_parquet(out_dir / _CSAT_WEEKLY_FILE, index=False)
    result["csat_monthly"].to_parquet(out_dir / _CSAT_MONTHLY_FILE, index=False)
    result["csat_hotel"].to_parquet(out_dir / _CSAT_HOTEL_FILE, index=False)

    logger.info("analyze_reviews: processed %d reviews, outputs persisted to %s", len(df), out_dir)
    return {
        "reviews_analyzed": len(df),
        "output_dir": str(out_dir),
        "keywords": result["keywords"],
        "topics": result["topics"],
    }


def _require(path) -> None:
    if not path.exists():
        raise ReviewsNotAnalyzedError(
            "Review analysis has not been run yet; call POST /reviews/analyze first."
        )


def get_summary() -> dict:
    out_dir = genai_settings.reviews_output_dir_path
    path = out_dir / _SUMMARY_FILE
    _require(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    keywords_path = out_dir / _KEYWORDS_FILE
    keywords = json.loads(keywords_path.read_text(encoding="utf-8"))["keywords"] if keywords_path.exists() else []
    hotel_csat_path = out_dir / _CSAT_HOTEL_FILE
    hotel_csat = (
        pd.read_parquet(hotel_csat_path).to_dict(orient="records") if hotel_csat_path.exists() else []
    )
    return {"summary": payload["summary"], "keywords": keywords, "csat_by_hotel": hotel_csat}


def get_topics() -> dict:
    out_dir = genai_settings.reviews_output_dir_path
    path = out_dir / _TOPICS_FILE
    _require(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {"topics": payload["topics"]}


def get_complaints() -> dict:
    out_dir = genai_settings.reviews_output_dir_path
    path = out_dir / _COMPLAINTS_FILE
    _require(path)
    df = pd.read_parquet(path)
    return {"complaints": df.to_dict(orient="records")}


def get_trends(grain: str = "weekly") -> dict:
    out_dir = genai_settings.reviews_output_dir_path
    file_map = {
        "daily": _CSAT_DAILY_FILE,
        "weekly": _CSAT_WEEKLY_FILE,
        "monthly": _CSAT_MONTHLY_FILE,
    }
    if grain not in file_map:
        raise ValueError(f"Unknown grain: {grain}; expected one of {sorted(file_map)}")
    path = out_dir / file_map[grain]
    _require(path)
    df = pd.read_parquet(path)

    trends_path = out_dir / _TRENDS_FILE
    trend_by_hotel = json.loads(trends_path.read_text(encoding="utf-8"))["trend_by_hotel"] if trends_path.exists() else {}

    return {"grain": grain, "series": df.to_dict(orient="records"), "trend_by_hotel": trend_by_hotel}
