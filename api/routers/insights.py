from fastapi import APIRouter, HTTPException

from api.schemas_genai import AnomaliesResponse, ExecutiveInsightsResponse, InsightsResponse, RecommendationsResponse
from genai.llm.factory import get_llm_provider
from genai.insights.service import get_anomalies, get_executive_summary, get_insights, get_recommendations
from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


def _safe_llm_provider():
    try:
        return get_llm_provider()
    except Exception as exc:
        logger.warning("LLM provider unavailable for insights generation: %s", exc)
        return None


@router.get("/insights", response_model=InsightsResponse)
def insights(category: str | None = None, min_severity: str | None = None) -> InsightsResponse:
    try:
        result = get_insights(category=category, min_severity=min_severity, llm_provider=_safe_llm_provider())
    except Exception as exc:
        logger.exception("Insights generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return InsightsResponse(insights=result)


@router.get("/insights/executive", response_model=ExecutiveInsightsResponse)
def executive() -> ExecutiveInsightsResponse:
    try:
        result = get_executive_summary(llm_provider=_safe_llm_provider())
    except Exception as exc:
        logger.exception("Executive insights generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return ExecutiveInsightsResponse(**result)


@router.get("/insights/recommendations", response_model=RecommendationsResponse)
def recommendations() -> RecommendationsResponse:
    try:
        result = get_recommendations(llm_provider=_safe_llm_provider())
    except Exception as exc:
        logger.exception("Recommendations generation failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return RecommendationsResponse(recommendations=result)


@router.get("/insights/anomalies", response_model=AnomaliesResponse)
def anomalies() -> AnomaliesResponse:
    try:
        result = get_anomalies(llm_provider=_safe_llm_provider())
    except Exception as exc:
        logger.exception("Anomaly detection failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return AnomaliesResponse(anomalies=result)
