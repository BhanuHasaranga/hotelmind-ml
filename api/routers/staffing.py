from fastapi import APIRouter, HTTPException

from api.schemas import StaffingRequest, StaffingResponse
from src.prediction.predict_staffing import recommend_staffing
from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/predict/staff", response_model=StaffingResponse)
def predict_staffing(request: StaffingRequest) -> StaffingResponse:
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
        raise HTTPException(status_code=503, detail="Staffing model not trained yet") from exc
    except Exception as exc:
        logger.exception("Staffing prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return StaffingResponse(**result)
