from fastapi import APIRouter, HTTPException

from api.schemas import PricingRequest, PricingResponse
from src.prediction.predict_pricing import recommend_price
from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/predict/pricing", response_model=PricingResponse)
def predict_pricing(request: PricingRequest) -> PricingResponse:
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
        raise HTTPException(status_code=503, detail="Pricing model not trained yet") from exc
    except Exception as exc:
        logger.exception("Pricing prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return PricingResponse(**result)
