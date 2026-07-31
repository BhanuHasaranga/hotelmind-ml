from fastapi import APIRouter, HTTPException

from api.schemas import OccupancyRequest, OccupancyResponse
from src.prediction.predict_occupancy import forecast_occupancy
from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/predict/occupancy", response_model=OccupancyResponse)
def predict_occupancy(request: OccupancyRequest) -> OccupancyResponse:
    try:
        forecast_df = forecast_occupancy(request.branch_id, request.horizon_days)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Occupancy model not trained yet") from exc
    except Exception as exc:
        logger.exception("Occupancy prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    forecast_df = forecast_df.copy()
    forecast_df["occupancy_date"] = forecast_df["occupancy_date"].dt.strftime("%Y-%m-%d")
    forecast = forecast_df.rename(columns={"occupancy_date": "date"}).to_dict(orient="records")
    return OccupancyResponse(branch_id=request.branch_id, forecast=forecast)
