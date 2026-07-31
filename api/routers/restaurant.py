from fastapi import APIRouter, HTTPException

from api.schemas import RestaurantRequest, RestaurantResponse
from src.prediction.predict_restaurant import forecast_restaurant_demand
from src.utils.logging import get_logger

router = APIRouter()
logger = get_logger(__name__)


@router.post("/predict/restaurant", response_model=RestaurantResponse)
def predict_restaurant(request: RestaurantRequest) -> RestaurantResponse:
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
        raise HTTPException(status_code=503, detail="Restaurant models not trained yet") from exc
    except Exception as exc:
        logger.exception("Restaurant prediction failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return RestaurantResponse(**result)
