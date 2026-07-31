import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


def handle_missing_values(
    df: pd.DataFrame,
    strategy: str = "median",
    columns: list[str] | None = None,
) -> pd.DataFrame:
    out = df.copy()
    cols = columns or out.select_dtypes(include="number").columns.tolist()
    if strategy == "median":
        for c in cols:
            out[c] = out[c].fillna(out[c].median())
    elif strategy == "ffill":
        out[cols] = out[cols].ffill().bfill()
    else:
        raise ValueError(f"Unknown missing-value strategy: {strategy}")
    return out


def encode_categoricals(
    df: pd.DataFrame, cols: list[str]
) -> tuple[pd.DataFrame, OrdinalEncoder]:
    out = df.copy()
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    out[cols] = encoder.fit_transform(out[cols].astype(str))
    return out, encoder


def apply_categorical_encoder(
    df: pd.DataFrame, cols: list[str], encoder: OrdinalEncoder
) -> pd.DataFrame:
    out = df.copy()
    out[cols] = encoder.transform(out[cols].astype(str))
    return out


def scale_features(
    df: pd.DataFrame, cols: list[str]
) -> tuple[pd.DataFrame, StandardScaler]:
    out = df.copy()
    scaler = StandardScaler()
    out[cols] = scaler.fit_transform(out[cols])
    return out, scaler


def apply_scaler(df: pd.DataFrame, cols: list[str], scaler: StandardScaler) -> pd.DataFrame:
    out = df.copy()
    out[cols] = scaler.transform(out[cols])
    return out
