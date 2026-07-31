import pandas as pd

from src.features.calendar_features import add_calendar_features
from src.features.preprocessing import (
    encode_categoricals,
    handle_missing_values,
    scale_features,
)
from src.features.time_series_features import add_lag_features, add_rolling_features


def test_add_calendar_features_basic_columns():
    df = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=10, freq="D")})
    out = add_calendar_features(df, "date")
    for col in ["month", "quarter", "day_of_week", "is_weekend", "season", "is_holiday", "is_event"]:
        assert col in out.columns
    assert out["is_weekend"].isin([0, 1]).all()


def test_add_calendar_features_flags_known_holiday():
    df = pd.DataFrame({"date": [pd.Timestamp("2024-07-04")]})
    out = add_calendar_features(df, "date")
    assert out.loc[0, "is_holiday"] == 1


def test_add_lag_features_no_group():
    df = pd.DataFrame({"value": [1, 2, 3, 4, 5]})
    out = add_lag_features(df, "value", lags=[1, 2])
    assert out["value_lag_1"].tolist() == [None, 1, 2, 3, 4] or out["value_lag_1"].isna().sum() == 1
    assert out["value_lag_2"].iloc[2] == 1


def test_add_lag_features_grouped():
    df = pd.DataFrame({"branch": ["a", "a", "b", "b"], "value": [10, 20, 100, 200]})
    out = add_lag_features(df, "value", lags=[1], group_col="branch")
    assert out["value_lag_1"].iloc[1] == 10
    assert out["value_lag_1"].iloc[3] == 100


def test_add_rolling_features_uses_prior_values_only():
    df = pd.DataFrame({"value": [10, 20, 30, 40]})
    out = add_rolling_features(df, "value", windows=[2])
    assert out["value_rolling_mean_2"].iloc[0] != 10 or out["value_rolling_mean_2"].isna().iloc[0]


def test_handle_missing_values_median():
    df = pd.DataFrame({"x": [1.0, None, 3.0]})
    out = handle_missing_values(df, strategy="median")
    assert out["x"].isna().sum() == 0
    assert out["x"].iloc[1] == 2.0


def test_encode_categoricals_returns_encoder():
    df = pd.DataFrame({"cat": ["a", "b", "a", "c"]})
    out, encoder = encode_categoricals(df, ["cat"])
    assert out["cat"].dtype.kind in "fi"
    assert encoder is not None


def test_scale_features_returns_scaler():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0]})
    out, scaler = scale_features(df, ["x"])
    assert abs(out["x"].mean()) < 1e-6
    assert scaler is not None
