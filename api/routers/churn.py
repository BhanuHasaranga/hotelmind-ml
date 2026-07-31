from fastapi import APIRouter, HTTPException

from api.schemas import ChurnRequest, ChurnResponse
from src.prediction.predict_churn import predict_churn as predict_churn_fn
from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/predict/churn", response_model=ChurnResponse)
def predict_churn(request: ChurnRequest) -> ChurnResponse:
    try:
        result = predict_churn_fn(request.guest_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Churn models not trained yet") from exc
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Churn prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChurnResponse(**result)
