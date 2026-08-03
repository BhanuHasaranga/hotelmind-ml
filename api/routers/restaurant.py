import time

from fastapi import APIRouter, HTTPException

from api.schemas import ModelMeta, RestaurantRequest, RestaurantResponse
from src.mlops.monitoring.metrics_middleware import record_prediction
from src.mlops.monitoring.prediction_drift import log_prediction_for_drift
from src.mlops.registry.model_registry import ModelRegistry
from src.prediction.predict_restaurant import forecast_restaurant_demand
from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

_DOMAIN = "restaurant"
_MODEL_NAME = "restaurant_lunch"  # representative model for meta/registry lookups


@router.post("/predict/restaurant", response_model=RestaurantResponse)
def predict_restaurant(request: RestaurantRequest) -> RestaurantResponse:
    start = time.perf_counter()
    try:
        result = forecast_restaurant_demand(
            branch_id=request.branch_id,
            date=request.date,
            recent_total_orders_lag_1=request.recent_total_orders_lag_1,
            recent_total_orders_lag_7=request.recent_total_orders_lag_7,
            recent_total_orders_rolling_mean_7=request.recent_total_orders_rolling_mean_7,
            avg_item_value=request.avg_item_value,
        )
    except FileNotFoundError as exc:
        record_prediction(_DOMAIN, "unknown", time.perf_counter() - start, error=True)
        raise HTTPException(status_code=503, detail="Restaurant models not trained yet") from exc
    except Exception as exc:
        record_prediction(_DOMAIN, "unknown", time.perf_counter() - start, error=True)
        logger.exception("Restaurant prediction failed")
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

    response = RestaurantResponse(
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
