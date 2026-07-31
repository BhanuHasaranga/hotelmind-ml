# Loading Summary

## Cleaning steps

| Step | Rows before | Rows after | Values changed | Reason |
|---|---|---|---|---|
| dedupe_full_row | 119390 | 87396 | 31994 | dropped exact-duplicate rows across all 32 source columns, keeping first occurrence |
| replace_undefined_with_null | 87396 | 87396 | 499 | replaced literal 'Undefined' placeholder with NULL |
| impute_missing_children | 87396 | 87396 | 4 | imputed missing children as 0 |
| impute_missing_country | 87396 | 87396 | 452 | imputed missing country as 'UNK' |
| fix_country_code_cn | 87396 | 87396 | 1093 | normalized 'CN' to ISO-3 'CHN' |
| clean_adr | 87396 | 87395 | 435 | dropped 1 rows with adr < 0, then winsorized 435 rows above the 99.5% percentile (cap=285.00) |
| flag_adults_outliers | 87395 | 87395 | 12 | flagged adults > 10 as is_adults_outlier; raw adults values left unmodified |
| check_cancellation_consistency | 87395 | 87395 | 0 | cross-validated reservation_status=='Canceled' against is_canceled==1 (reporting only, no rows modified) |
| normalize_categoricals | 87395 | 87395 | 0 | trimmed whitespace on categorical string columns |
| build_arrival_date | 87395 | 87395 | 87395 | constructed arrival_date from arrival_date_year/month(name)/day |
| convert_dtypes | 87395 | 87395 | 26 | converted columns to nullable Int64 / float64 / category / datetime64 dtypes |

## Warehouse table row counts

| Table | Rows |
|---|---|
| dim_date | 806 |
| dim_hotel | 2 |
| dim_room_type | 4 |
| dim_guest | 5672 |
| fact_booking | 87395 |

## Execution times (seconds)

| Stage | Seconds |
|---|---|
| load_cleaned | 0.08 |
| build_all | 12.56 |
| save_local | 0.61 |
| validate | 0.02 |

## Warnings

- check_cancellation_consistency: 1014 reservation_status/is_canceled mismatches

## Errors

- None
