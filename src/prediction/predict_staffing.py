import pandas as pd

from src.config.settings import settings
from src.features.calendar_features import add_calendar_features
from src.features.preprocessing import apply_categorical_encoder
from src.models.staffing.regression_model import StaffingRegressionModel
from src.pipelines.staffing_pipeline import CATEGORICAL_COLS, FEATURE_COLS


def recommend_staffing(
    branch_id: int,
    department: str,
    date: str,
    scheduled_employees: int,
    present_employees_lag_7: float,
    present_employees_rolling_mean_7: float,
) -> dict:
    model = StaffingRegressionModel()
    model.load(settings.model_dir_path / "staffing_regression.pkl")

    row = pd.DataFrame(
        [
            {
                "date": pd.to_datetime(date),
                "scheduled_employees": scheduled_employees,
                "present_employees_lag_7": present_employees_lag_7,
                "present_employees_rolling_mean_7": present_employees_rolling_mean_7,
                "department_name": department,
            }
        ]
    )
    row = add_calendar_features(row, "date")
    row = apply_categorical_encoder(row, CATEGORICAL_COLS, model.encoders["categorical"])

    required_staff = float(model.predict(row[FEATURE_COLS])[0])
    return {
        "branch_id": branch_id,
        "department": department,
        "date": date,
        "required_staff": round(required_staff),
        "confidence_note": (
            "Point estimate from GradientBoostingRegressor; shift assignment "
            "still requires OR-Tools scheduling (see src/models/staffing/"
            "or_tools_scheduler.py TODO)."
        ),
    }
