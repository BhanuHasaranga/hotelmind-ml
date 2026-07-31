import pandas as pd


def add_lag_features(
    df: pd.DataFrame,
    col: str,
    lags: list[int] = [1, 7, 30],
    group_col: str | None = None,
) -> pd.DataFrame:
    out = df.copy()
    for lag in lags:
        name = f"{col}_lag_{lag}"
        if group_col:
            out[name] = out.groupby(group_col)[col].shift(lag)
        else:
            out[name] = out[col].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    col: str,
    windows: list[int] = [7, 30],
    group_col: str | None = None,
) -> pd.DataFrame:
    out = df.copy()
    for window in windows:
        name = f"{col}_rolling_mean_{window}"
        if group_col:
            out[name] = (
                out.groupby(group_col)[col]
                .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
            )
        else:
            out[name] = out[col].shift(1).rolling(window, min_periods=1).mean()
    return out
