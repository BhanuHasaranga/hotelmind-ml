from src.models.base import BaseMLModel
from src.models.churn.random_forest_model import ChurnRandomForestModel
from src.models.churn.xgboost_model import ChurnXGBoostModel
from src.models.occupancy.prophet_model import OccupancyProphetModel
from src.models.occupancy.xgboost_model import OccupancyXGBoostModel
from src.models.pricing.xgboost_model import PricingXGBoostModel
from src.models.restaurant.xgboost_model import RestaurantDemandModel
from src.models.staffing.regression_model import StaffingRegressionModel

_MODEL_CLASSES: dict[str, type[BaseMLModel]] = {
    "occupancy_prophet": OccupancyProphetModel,
    "occupancy_xgboost": OccupancyXGBoostModel,
    "pricing_xgboost": PricingXGBoostModel,
    "restaurant_breakfast": RestaurantDemandModel,
    "restaurant_lunch": RestaurantDemandModel,
    "restaurant_dinner": RestaurantDemandModel,
    "staffing_regression": StaffingRegressionModel,
    "churn_random_forest": ChurnRandomForestModel,
    "churn_xgboost": ChurnXGBoostModel,
}


def create_model(model_name: str) -> BaseMLModel:
    """Factory: instantiate the (untrained) BaseMLModel subclass for a registered model name."""
    model_cls = _MODEL_CLASSES.get(model_name)
    if model_cls is None:
        known = ", ".join(sorted(_MODEL_CLASSES))
        raise ValueError(f"Unknown model_name={model_name!r}. Known names: {known}")
    return model_cls()
