# Mapping & Assumption Summary

## Hotel / branch identity

hotel_key and branch_key both come from a fixed lookup (`Resort Hotel` -> 1, `City Hotel` -> 2). The canonical dataset has no branch granularity below hotel, so branch_key reuses hotel_key.

## Room type mapping

The raw dataset's reserved/assigned_room_type columns are single-letter codes (A, B, C, ...) with no authoritative mapping to the seed room_type_dim.csv names (Standard/Deluxe/Suite). An ordinal-tier assumption is used (A-C -> Standard, D-G -> Deluxe, H/L -> Suite); any other code falls into an explicit 'Unmapped' bucket (room_type_id=0) rather than being dropped. This mapping should be reviewed by a domain stakeholder before being treated as ground truth.

## Guest identity

The anonymized source data has no true guest identity. guest_key is a hash of (country, market_segment, distribution_channel, customer_type, is_repeated_guest, agent, company) — this identifies a 'guest profile cluster', not a real unique person. Many distinct real guests will collide into the same guest_key; this is an accepted limitation, not a bug. Party composition (adults/children/babies) was deliberately excluded from the hash since it varies trip-to-trip for the same guest and would fragment rather than cluster.

## Reservation identity

reservation_id/surrogate_key is a hash of all cleaned-row values plus a stable post-sort row index (fact rows are sorted by arrival_date/hotel/lead_time/adr with a stable mergesort before key assignment), guarding against two bookings that share identical values across every column.

## total_amount derivation

total_amount = nights * adr. No paid_amount/outstanding_amount data exists anywhere in the canonical dataset; those fields are simply not populated by this loader rather than fabricated as 0 or a misleading NULL.

## Out-of-scope tables

fact_restaurant_sale and fact_staff_attendance are not populated — no data in any of the source CSVs supports them. They remain untouched, as instructed, and are not created as empty/stub files.
