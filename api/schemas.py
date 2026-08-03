from pydantic import BaseModel


class ModelMeta(BaseModel):
    model_version: int | str
    trained_at: str
    mlflow_experiment_id: str | None = None
    latency_ms: float
    confidence: float | None = None


class OccupancyRequest(BaseModel):
    branch_id: int
    horizon_days: int | None = None


class OccupancyDayForecast(BaseModel):
    date: str
    predicted_occupancy_pct: float
    ci_lower: float
    ci_upper: float
    model_used: str


class OccupancyResponse(BaseModel):
    branch_id: int
    forecast: list[OccupancyDayForecast]
    meta: ModelMeta | None = None


class PricingRequest(BaseModel):
    branch_id: int
    room_type_id: int
    date: str
    current_occupancy_pct: float
    current_revenue: float
    revenue_7day_avg: float
    total_rooms: int


class PricingResponse(BaseModel):
    branch_id: int
    room_type_id: int
    date: str
    recommended_price: float
    expected_revenue: float
    meta: ModelMeta | None = None


class RestaurantRequest(BaseModel):
    branch_id: int
    date: str
    recent_total_orders_lag_1: float
    recent_total_orders_lag_7: float
    recent_total_orders_rolling_mean_7: float
    avg_item_value: float


class MealForecast(BaseModel):
    expected_quantity: float
    expected_revenue: float


class RestaurantResponse(BaseModel):
    branch_id: int
    date: str
    breakfast: MealForecast
    lunch: MealForecast
    dinner: MealForecast
    meta: ModelMeta | None = None


class StaffingRequest(BaseModel):
    branch_id: int
    department: str
    date: str
    scheduled_employees: int
    present_employees_lag_7: float
    present_employees_rolling_mean_7: float


class StaffingResponse(BaseModel):
    branch_id: int
    department: str
    date: str
    required_staff: int
    confidence_note: str
    meta: ModelMeta | None = None


class ChurnRequest(BaseModel):
    guest_id: str


class ChurnResponse(BaseModel):
    guest_id: str
    churn_probability: float | None
    risk_level: str
    model_used: str
    note: str | None = None
    meta: ModelMeta | None = None
