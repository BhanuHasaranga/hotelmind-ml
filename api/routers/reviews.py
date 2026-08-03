from fastapi import APIRouter, HTTPException

from api.schemas_genai import (
    ReviewsAnalyzeRequest,
    ReviewsAnalyzeResponse,
    ReviewsComplaintsResponse,
    ReviewsSummaryResponse,
    ReviewsTopicsResponse,
    ReviewsTrendsResponse,
)
from genai.reviews.service import ReviewsNotAnalyzedError, analyze_reviews, get_complaints, get_summary, get_topics, get_trends
from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/reviews/analyze", response_model=ReviewsAnalyzeResponse)
def analyze(request: ReviewsAnalyzeRequest) -> ReviewsAnalyzeResponse:
    try:
        result = analyze_reviews(limit=request.limit)
    except Exception as exc:
        logger.exception("Review analysis failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ReviewsAnalyzeResponse(**result)


@router.get("/reviews/summary", response_model=ReviewsSummaryResponse)
def summary() -> ReviewsSummaryResponse:
    try:
        result = get_summary()
    except ReviewsNotAnalyzedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Fetching review summary failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ReviewsSummaryResponse(**result)


@router.get("/reviews/topics", response_model=ReviewsTopicsResponse)
def topics() -> ReviewsTopicsResponse:
    try:
        result = get_topics()
    except ReviewsNotAnalyzedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Fetching review topics failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ReviewsTopicsResponse(**result)


@router.get("/reviews/complaints", response_model=ReviewsComplaintsResponse)
def complaints() -> ReviewsComplaintsResponse:
    try:
        result = get_complaints()
    except ReviewsNotAnalyzedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Fetching review complaints failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ReviewsComplaintsResponse(**result)


@router.get("/reviews/trends", response_model=ReviewsTrendsResponse)
def trends(grain: str = "weekly") -> ReviewsTrendsResponse:
    try:
        result = get_trends(grain=grain)
    except ReviewsNotAnalyzedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Fetching review trends failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ReviewsTrendsResponse(**result)
