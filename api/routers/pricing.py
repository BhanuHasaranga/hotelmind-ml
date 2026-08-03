import time

from fastapi import APIRouter, HTTPException

from api.schemas import ModelMeta, PricingRequest, PricingResponse
from src.mlops.monitoring.metrics_middleware import record_prediction
from src.mlops.monitoring.prediction_drift import log_prediction_for_drift
from src.mlops.registry.model_registry import ModelRegistry
from src.prediction.predict_pricing import recommend_price
from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

_DOMAIN = "pricing"
_MODEL_NAME = "pricing_xgboost"


@router.post("/predict/pricing", response_model=PricingResponse)
def predict_pricing(request: PricingRequest) -> PricingResponse:
    start = time.perf_counter()
    try:
        result = recommend_price(
            branch_id=request.branch_id,
            room_type_id=request.room_type_id,
            date=request.date,
            current_occupancy_pct=request.current_occupancy_pct,
            current_revenue=request.current_revenue,
            revenue_7day_avg=request.revenue_7day_avg,
            total_rooms=request.total_rooms,
        )
    except FileNotFoundError as exc:
        record_prediction(_DOMAIN, "unknown", time.perf_counter() - start, error=True)
        raise HTTPException(status_code=503, detail="Pricing model not trained yet") from exc
    except Exception as exc:
        record_prediction(_DOMAIN, "unknown", time.perf_counter() - start, error=True)
        logger.exception("Pricing prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_s = time.perf_counter() - start

    model_version: int | str = "legacy"
    trained_at = "unknown"
    mlflow_experiment_id = None
    try:
        record = ModelRegistry().get_record(_MODEL_NAME)
        model_version = record.version
        trained_at = record.trained_at
        mlflow_experiment_id = record.mlflow_run_id
    except Exception:
        pass

    record_prediction(_DOMAIN, str(model_version), latency_s, error=False)

    response = PricingResponse(
        **result,
        meta=ModelMeta(
            model_version=model_version,
            trained_at=trained_at,
            mlflow_experiment_id=mlflow_experiment_id,
            latency_ms=latency_s * 1000,
            confidence=None,
        ),
    )

    try:
        log_prediction_for_drift(_DOMAIN, response.model_dump())
    except Exception:
        pass

    return response
