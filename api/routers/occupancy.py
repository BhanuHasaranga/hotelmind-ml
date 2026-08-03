import time

from fastapi import APIRouter, HTTPException

from api.schemas import ModelMeta, OccupancyRequest, OccupancyResponse
from src.mlops.monitoring.metrics_middleware import record_prediction
from src.mlops.monitoring.prediction_drift import log_prediction_for_drift
from src.mlops.registry.model_registry import ModelRegistry
from src.prediction.predict_occupancy import forecast_occupancy
from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

_DOMAIN = "occupancy"
_MODEL_NAME = "occupancy_prophet"


@router.post("/predict/occupancy", response_model=OccupancyResponse)
def predict_occupancy(request: OccupancyRequest) -> OccupancyResponse:
    start = time.perf_counter()
    try:
        forecast_df = forecast_occupancy(request.branch_id, request.horizon_days)
    except FileNotFoundError as exc:
        record_prediction(_DOMAIN, "unknown", time.perf_counter() - start, error=True)
        raise HTTPException(status_code=503, detail="Occupancy model not trained yet") from exc
    except Exception as exc:
        record_prediction(_DOMAIN, "unknown", time.perf_counter() - start, error=True)
        logger.exception("Occupancy prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_s = time.perf_counter() - start

    forecast_df = forecast_df.copy()
    forecast_df["occupancy_date"] = forecast_df["occupancy_date"].dt.strftime("%Y-%m-%d")
    forecast = forecast_df.rename(columns={"occupancy_date": "date"}).to_dict(orient="records")

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

    confidence = None
    try:
        if forecast:
            first = forecast[0]
            width = first.get("ci_upper", 0) - first.get("ci_lower", 0)
            confidence = max(0.0, 1.0 - (width / 100.0))
    except Exception:
        confidence = None

    response = OccupancyResponse(
        branch_id=request.branch_id,
        forecast=forecast,
        meta=ModelMeta(
            model_version=model_version,
            trained_at=trained_at,
            mlflow_experiment_id=mlflow_experiment_id,
            latency_ms=latency_s * 1000,
            confidence=confidence,
        ),
    )

    try:
        log_prediction_for_drift(_DOMAIN, response.model_dump())
    except Exception:
        pass

    return response
