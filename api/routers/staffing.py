import time

from fastapi import APIRouter, HTTPException

from api.schemas import ModelMeta, StaffingRequest, StaffingResponse
from src.mlops.monitoring.metrics_middleware import record_prediction
from src.mlops.monitoring.prediction_drift import log_prediction_for_drift
from src.mlops.registry.model_registry import ModelRegistry
from src.prediction.predict_staffing import recommend_staffing
from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

_DOMAIN = "staffing"
_MODEL_NAME = "staffing_regression"


@router.post("/predict/staff", response_model=StaffingResponse)
def predict_staffing(request: StaffingRequest) -> StaffingResponse:
    start = time.perf_counter()
    try:
        result = recommend_staffing(
            branch_id=request.branch_id,
            department=request.department,
            date=request.date,
            scheduled_employees=request.scheduled_employees,
            present_employees_lag_7=request.present_employees_lag_7,
            present_employees_rolling_mean_7=request.present_employees_rolling_mean_7,
        )
    except FileNotFoundError as exc:
        record_prediction(_DOMAIN, "unknown", time.perf_counter() - start, error=True)
        raise HTTPException(status_code=503, detail="Staffing model not trained yet") from exc
    except Exception as exc:
        record_prediction(_DOMAIN, "unknown", time.perf_counter() - start, error=True)
        logger.exception("Staffing prediction failed")
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

    response = StaffingResponse(
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
