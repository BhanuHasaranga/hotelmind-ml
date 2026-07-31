# Key Generation Strategy

All keys are deterministic: identical input always produces an identical key, 
via `src.pipelines.keygen`. No randomness or UUIDs are used anywhere.

| Key | Algorithm |
|---|---|
| date_key | int(YYYYMMDD) from the calendar date. |
| hotel_key / branch_key | Fixed dict lookup: {'Resort Hotel': 1, 'City Hotel': 2}. |
| room_type_key | Letter code -> tier id via ROOM_TYPE_CODE_MAP, else 0 ('Unmapped'). |
| guest_key | SHA-256 hash of (country, market_segment, distribution_channel, customer_type, is_repeated_guest, agent, company), truncated to 63 bits. |
| surrogate_key / reservation_id | SHA-256 hash of all cleaned-row values plus a stable post-sort row index. |

## Unmapped room-type codes encountered

I, K, P

## Guest identity clustering

- Distinct guest_keys: 5672
- Total booking rows: 87395
- Average rows per guest_key cluster: 15.41
