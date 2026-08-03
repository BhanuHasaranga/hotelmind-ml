import time

from fastapi import APIRouter, HTTPException

from api.schemas import ChurnRequest, ChurnResponse, ModelMeta
from src.mlops.monitoring.metrics_middleware import record_prediction
from src.mlops.monitoring.prediction_drift import log_prediction_for_drift
from src.mlops.registry.model_registry import ModelRegistry
from src.prediction.predict_churn import predict_churn as predict_churn_fn
from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)

_DOMAIN = "churn"
_MODEL_NAME_BY_USED = {
    "xgboost": "churn_xgboost",
    "random_forest": "churn_random_forest",
}
_DEFAULT_MODEL_NAME = "churn_xgboost"


@router.post("/predict/churn", response_model=ChurnResponse)
def predict_churn(request: ChurnRequest) -> ChurnResponse:
    start = time.perf_counter()
    try:
        result = predict_churn_fn(request.guest_id)
    except FileNotFoundError as exc:
        record_prediction(_DOMAIN, "unknown", time.perf_counter() - start, error=True)
        raise HTTPException(status_code=503, detail="Churn models not trained yet") from exc
    except ValueError as exc:
        record_prediction(_DOMAIN, "unknown", time.perf_counter() - start, error=True)
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        record_prediction(_DOMAIN, "unknown", time.perf_counter() - start, error=True)
        logger.exception("Churn prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    latency_s = time.perf_counter() - start

    model_name = _MODEL_NAME_BY_USED.get(result.get("model_used"), _DEFAULT_MODEL_NAME)
    model_version: int | str = "legacy"
    trained_at = "unknown"
    mlflow_experiment_id = None
    try:
        record = ModelRegistry().get_record(model_name)
        model_version = record.version
        trained_at = record.trained_at
        mlflow_experiment_id = record.mlflow_run_id
    except Exception:
        pass

    record_prediction(_DOMAIN, str(model_version), latency_s, error=False)

    confidence = result.get("churn_probability")

    response = ChurnResponse(
        **result,
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
