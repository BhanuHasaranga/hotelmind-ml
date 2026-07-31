# Validation Report

## Row counts / primary key uniqueness

| Table | Row count | PK unique |
|---|---|---|
| dim_date | 806 | True |
| dim_hotel | 2 | True |
| dim_room_type | 4 | True |
| dim_guest | 5672 | True |
| fact_booking | 87395 | True |

## NULL violations

| Table | Column | Null count |
|---|---|---|
| fact_booking | surrogate_key | 0 |
| fact_booking | room_key | 0 |
| fact_booking | branch_key | 0 |
| fact_booking | guest_key | 0 |
| fact_booking | check_in_date_key | 0 |

## Foreign key integrity

| FK | Orphan count | Sample orphans |
|---|---|---|
| fact_booking.room_key -> dim_room_type.room_type_key | 0 | [] |
| fact_booking.branch_key -> dim_hotel.hotel_key | 0 | [] |
| fact_booking.guest_key -> dim_guest.guest_key | 0 | [] |
| fact_booking.check_in_date_key -> dim_date.date_key | 0 | [] |
| fact_booking.check_out_date_key -> dim_date.date_key | 0 | [] |

## Overall result: PASS
