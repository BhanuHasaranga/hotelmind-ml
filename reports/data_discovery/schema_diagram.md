# Schema Diagram — Raw Datasets → Warehouse

## Raw dataset family relationships

```mermaid
flowchart LR
    subgraph FamilyA["Family A: Hotel Booking Demand (canonical candidate)"]
        A0[hotel_bookings.csv\nhotel_booking_demand]
        A1[hotel_bookings.csv\nhotel_booking_demand_indexed]
        A2[hotel_booking.csv\nhotel_booking_demand_pii]
        A3[hotel_bookings_cleaned.csv]
        A4[hotel_bookings_updated_2024.csv]
        A5[bookings_reduced_columns.csv]
        A0 --> A1
        A0 --> A2
        A0 --> A3
        A0 --> A4
        A0 --> A5
    end

    subgraph FamilyB["Family B: Hotel Reservations (independent)"]
        B0[Hotel Reservations.csv]
    end

    subgraph Seeds["Synthetic seeds (already curated)"]
        S0[room_type_dim.csv]
        S1[events_holiday_calendar.csv]
    end
```

## Family A canonical file → Phase 3 warehouse (target grain: `fact_booking`)

```mermaid
flowchart TD
    CSV["hotel_bookings.csv\n(hotel_booking_demand)\nGrain: 1 row per reservation"]

    CSV --> FB["fact_booking\nGrain: reservation_id"]
    CSV --> DG["dim_guest\nGrain: guest_id"]
    CSV --> DR["dim_room\nGrain: room_id"]
    CSV --> DH["dim_hotel / dim_branch\nGrain: hotel_id / branch_id"]
    CSV --> DD["dim_date\nGrain: date"]

    FB --> MO["mart_occupancy_daily\nbranch_id x date"]
    FB --> MR["mart_revenue_daily\nbranch_id x date"]
    DG --> MG["guest.sql query\n(lifetime value features)"]

    MO --> Occ["Occupancy ML module"]
    MR --> Pricing["Pricing ML module"]
    MG --> Churn["Churn ML module"]
```

## Warehouse star schema (existing, from `hotelmind-data` dbt project)

```mermaid
erDiagram
    dim_guest ||--o{ fact_booking : "guest_key"
    dim_room ||--o{ fact_booking : "room_key"
    dim_branch ||--o{ fact_booking : "branch_key"
    dim_date ||--o{ fact_booking : "check_in_date_key"
    dim_date ||--o{ fact_booking : "check_out_date_key"

    fact_booking ||--o{ mart_revenue_daily : "aggregated by branch_id x date"
    fact_occupancy_daily ||--o{ mart_occupancy_daily : "aggregated"
    fact_restaurant_sale ||--o{ mart_restaurant_daily : "aggregated"
    fact_staff_attendance ||--o{ mart_staff_daily : "aggregated"

    dim_guest {
        string guest_key PK
        string guest_id
        string full_name
        string nationality
        int lifetime_bookings
        numeric lifetime_spend
        date first_stay_date
        date last_stay_date
    }
    fact_booking {
        string surrogate_key PK
        string reservation_id
        string reservation_status
        string room_key FK
        string branch_key FK
        string guest_key FK
        int check_in_date_key FK
        int check_out_date_key FK
        int nights
        int adults
        int children
        numeric total_amount
        numeric avg_daily_rate
        bool is_terminal
        bool is_completed
    }
```

**Gap:** the Kaggle CSVs have no `hotel_id`/`branch_id`, no `room_id`, no
`guest_id`, and no stable `reservation_id` — see `warehouse_mapping.md` for
how each is synthesized or left unmapped.
