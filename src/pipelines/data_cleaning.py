"""Data cleaning pipeline for the canonical hotel_bookings.csv dataset.

Implements every rule approved in reports/data_discovery/cleaning_plan.md.
Each step is a small, pure function returning (df, step_stats) so cleaning
rules are independently unit-testable and every step's before/after/changed
counts are captured without relying on log-scraping.

CLI: python -m src.pipelines.data_cleaning
"""

import json
import time
from pathlib import Path

import pandas as pd

from src.config.settings import settings
from src.utils.logging import get_logger

logger = get_logger(__name__)

CANONICAL_CSV_RELATIVE = Path("hotel_booking_demand/original/hotel_bookings.csv")

EXPECTED_COLUMNS = [
    "hotel", "is_canceled", "lead_time", "arrival_date_year", "arrival_date_month",
    "arrival_date_week_number", "arrival_date_day_of_month", "stays_in_weekend_nights",
    "stays_in_week_nights", "adults", "children", "babies", "meal", "country",
    "market_segment", "distribution_channel", "is_repeated_guest", "previous_cancellations",
    "previous_bookings_not_canceled", "reserved_room_type", "assigned_room_type",
    "booking_changes", "deposit_type", "agent", "company", "days_in_waiting_list",
    "customer_type", "adr", "required_car_parking_spaces", "total_of_special_requests",
    "reservation_status", "reservation_status_date",
]

MONTH_NAME_TO_NUM = {
    "January": 1, "February": 2, "March": 3, "April": 4, "May": 5, "June": 6,
    "July": 7, "August": 8, "September": 9, "October": 10, "November": 11, "December": 12,
}

UNDEFINED_PLACEHOLDER_COLUMNS = ["meal", "market_segment", "distribution_channel"]

CATEGORICAL_COLUMNS = [
    "hotel", "meal", "country", "market_segment", "distribution_channel",
    "reserved_room_type", "assigned_room_type", "deposit_type", "customer_type",
    "reservation_status",
]

NULLABLE_INT_COLUMNS = [
    "is_canceled", "is_repeated_guest", "required_car_parking_spaces",
    "total_of_special_requests", "booking_changes", "previous_cancellations",
    "previous_bookings_not_canceled", "days_in_waiting_list", "children",
    "adults", "babies", "stays_in_weekend_nights", "stays_in_week_nights",
    "agent", "company",
]


def load_raw(path: Path | None = None) -> pd.DataFrame:
    csv_path = path or (settings.data_raw_dir_path / CANONICAL_CSV_RELATIVE)
    df = pd.read_csv(csv_path)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    extra = [c for c in df.columns if c not in EXPECTED_COLUMNS]
    if missing or extra:
        raise ValueError(
            f"hotel_bookings.csv schema mismatch. Missing columns: {missing}. "
            f"Unexpected columns: {extra}."
        )

    logger.info("Loaded raw dataset: %d rows, %d columns", len(df), len(df.columns))
    return df


def dedupe_full_row(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows_before = len(df)
    out = df.drop_duplicates(subset=EXPECTED_COLUMNS, keep="first")
    rows_after = len(out)
    stats = {
        "rows_before": rows_before,
        "rows_after": rows_after,
        "values_changed": rows_before - rows_after,
        "reason": "dropped exact-duplicate rows across all 32 source columns, keeping first occurrence",
    }
    logger.info("dedupe_full_row: %s", stats)
    return out, stats


def replace_undefined_with_null(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    rows_before = len(out)
    per_column = {}
    for col in UNDEFINED_PLACEHOLDER_COLUMNS:
        mask = out[col] == "Undefined"
        per_column[col] = int(mask.sum())
        out.loc[mask, col] = pd.NA
    stats = {
        "rows_before": rows_before,
        "rows_after": len(out),
        "values_changed": sum(per_column.values()),
        "reason": "replaced literal 'Undefined' placeholder with NULL",
        "per_column": per_column,
    }
    logger.info("replace_undefined_with_null: %s", stats)
    return out, stats


def impute_missing_children(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    mask = out["children"].isna()
    changed = int(mask.sum())
    out.loc[mask, "children"] = 0
    stats = {
        "rows_before": len(out),
        "rows_after": len(out),
        "values_changed": changed,
        "reason": "imputed missing children as 0",
    }
    logger.info("impute_missing_children: %s", stats)
    return out, stats


def impute_missing_country(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    mask = out["country"].isna()
    changed = int(mask.sum())
    out.loc[mask, "country"] = "UNK"
    stats = {
        "rows_before": len(out),
        "rows_after": len(out),
        "values_changed": changed,
        "reason": "imputed missing country as 'UNK'",
    }
    logger.info("impute_missing_country: %s", stats)
    return out, stats


def fix_country_code_cn(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    mask = out["country"] == "CN"
    changed = int(mask.sum())
    out.loc[mask, "country"] = "CHN"
    stats = {
        "rows_before": len(out),
        "rows_after": len(out),
        "values_changed": changed,
        "reason": "normalized 'CN' to ISO-3 'CHN'",
    }
    logger.info("fix_country_code_cn: %s", stats)
    return out, stats


def clean_adr(df: pd.DataFrame, winsor_pct: float = 0.995) -> tuple[pd.DataFrame, dict]:
    rows_before = len(df)
    out = df[df["adr"] >= 0].copy()
    dropped = rows_before - len(out)

    cap = float(out["adr"].quantile(winsor_pct))
    capped_mask = out["adr"] > cap
    capped_count = int(capped_mask.sum())
    out.loc[capped_mask, "adr"] = cap

    stats = {
        "rows_before": rows_before,
        "rows_after": len(out),
        "values_changed": capped_count,
        "reason": (
            f"dropped {dropped} rows with adr < 0, then winsorized "
            f"{capped_count} rows above the {winsor_pct:.1%} percentile "
            f"(cap={cap:.2f})"
        ),
        "rows_dropped_negative": dropped,
        "winsor_cap": cap,
    }
    logger.info("clean_adr: %s", stats)
    return out, stats


def flag_adults_outliers(df: pd.DataFrame, threshold: int = 10) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    flag = out["adults"] > threshold
    out["is_adults_outlier"] = flag
    stats = {
        "rows_before": len(out),
        "rows_after": len(out),
        "values_changed": int(flag.sum()),
        "reason": f"flagged adults > {threshold} as is_adults_outlier; raw adults values left unmodified",
    }
    logger.info("flag_adults_outliers: %s", stats)
    return out, stats


def check_cancellation_consistency(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    mismatch = (df["reservation_status"] == "Canceled") != (df["is_canceled"] == 1)
    mismatch_count = int(mismatch.sum())
    sample = df.index[mismatch][:5].tolist()
    stats = {
        "rows_before": len(df),
        "rows_after": len(df),
        "values_changed": 0,
        "reason": "cross-validated reservation_status=='Canceled' against is_canceled==1 (reporting only, no rows modified)",
        "mismatch_count": mismatch_count,
        "sample_mismatch_indices": sample,
    }
    logger.info("check_cancellation_consistency: %s", stats)
    return df, stats


def normalize_categoricals(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    per_column = {}
    for col in CATEGORICAL_COLUMNS:
        if col not in out.columns:
            continue
        original = out[col].astype("string")
        stripped = original.str.strip()
        changed = int((original != stripped).fillna(False).sum())
        out[col] = stripped
        per_column[col] = changed
    stats = {
        "rows_before": len(out),
        "rows_after": len(out),
        "values_changed": sum(per_column.values()),
        "reason": "trimmed whitespace on categorical string columns",
        "per_column": per_column,
    }
    logger.info("normalize_categoricals: %s", stats)
    return out, stats


def build_arrival_date(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    month_num = out["arrival_date_month"].map(MONTH_NAME_TO_NUM)
    date_parts = pd.DataFrame({
        "year": out["arrival_date_year"],
        "month": month_num,
        "day": out["arrival_date_day_of_month"],
    })
    out["arrival_date"] = pd.to_datetime(date_parts, errors="coerce")
    unparseable = int(out["arrival_date"].isna().sum())
    stats = {
        "rows_before": len(out),
        "rows_after": len(out),
        "values_changed": len(out),
        "reason": "constructed arrival_date from arrival_date_year/month(name)/day",
        "unparseable_dates": unparseable,
    }
    if unparseable:
        logger.warning("build_arrival_date: %d rows have unparseable arrival dates", unparseable)
    logger.info("build_arrival_date: %s", stats)
    return out, stats


def convert_dtypes(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    before_dtypes = {c: str(out[c].dtype) for c in out.columns}

    for col in NULLABLE_INT_COLUMNS:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("Int64")

    out["adr"] = out["adr"].astype("float64")

    for col in CATEGORICAL_COLUMNS:
        out[col] = out[col].astype("category")

    out["reservation_status_date"] = pd.to_datetime(out["reservation_status_date"], errors="coerce")

    after_dtypes = {c: str(out[c].dtype) for c in out.columns}
    changed_cols = {c: (before_dtypes[c], after_dtypes[c]) for c in out.columns if before_dtypes.get(c) != after_dtypes.get(c)}

    stats = {
        "rows_before": len(out),
        "rows_after": len(out),
        "values_changed": len(changed_cols),
        "reason": "converted columns to nullable Int64 / float64 / category / datetime64 dtypes",
        "dtype_changes": changed_cols,
    }
    logger.info("convert_dtypes: converted %d columns", len(changed_cols))
    return out, stats


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    steps = [
        ("dedupe_full_row", dedupe_full_row),
        ("replace_undefined_with_null", replace_undefined_with_null),
        ("impute_missing_children", impute_missing_children),
        ("impute_missing_country", impute_missing_country),
        ("fix_country_code_cn", fix_country_code_cn),
        ("clean_adr", clean_adr),
        ("flag_adults_outliers", flag_adults_outliers),
        ("check_cancellation_consistency", check_cancellation_consistency),
        ("normalize_categoricals", normalize_categoricals),
        ("build_arrival_date", build_arrival_date),
        ("convert_dtypes", convert_dtypes),
    ]

    overall_stats = {}
    out = df
    for name, fn in steps:
        out, step_stats = fn(out)
        overall_stats[name] = step_stats

    return out, overall_stats


def save(df: pd.DataFrame) -> tuple[Path, Path]:
    parquet_path = settings.data_processed_dir_path / "hotel_bookings_clean.parquet"
    csv_path = settings.data_processed_dir_path / "hotel_bookings_clean.csv"
    df.to_parquet(parquet_path, index=False)
    df.to_csv(csv_path, index=False)
    logger.info("Saved cleaned dataset to %s and %s", parquet_path, csv_path)
    return parquet_path, csv_path


def run() -> dict:
    start = time.perf_counter()
    warnings = []
    errors = []

    try:
        raw_df = load_raw()
        loaded_rows = len(raw_df)

        cleaned_df, stats = clean(raw_df)
        cleaned_rows = len(cleaned_df)
        rejected_rows = loaded_rows - cleaned_rows

        save(cleaned_df)

        for step_name, step_stats in stats.items():
            unparseable = step_stats.get("unparseable_dates")
            if unparseable:
                warnings.append(f"{step_name}: {unparseable} unparseable arrival dates")
            mismatch = step_stats.get("mismatch_count")
            if mismatch:
                warnings.append(f"{step_name}: {mismatch} reservation_status/is_canceled mismatches")
    except Exception as exc:
        errors.append(str(exc))
        logger.exception("data_cleaning pipeline failed")
        raise
    finally:
        elapsed = time.perf_counter() - start

    summary = {
        "loaded_rows": loaded_rows,
        "cleaned_rows": cleaned_rows,
        "rejected_rows": rejected_rows,
        "execution_seconds": elapsed,
        "warnings": warnings,
        "errors": errors,
        "steps": stats,
    }

    logger.info(
        "data_cleaning complete: loaded=%d cleaned=%d rejected=%d elapsed=%.2fs",
        loaded_rows, cleaned_rows, rejected_rows, elapsed,
    )

    stats_path = settings.warehouse_loading_reports_dir_path / "_cleaning_stats.json"
    stats_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    return summary


if __name__ == "__main__":
    run()
