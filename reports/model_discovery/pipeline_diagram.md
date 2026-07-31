# Pipeline Diagram — Warehouse Parquet → Features → Models → API

## Data lineage: warehouse parquet → derived/synthetic → feature datasets

```mermaid
flowchart LR
    subgraph Phase3["Phase 3 outputs (input to Phase 4, read-only)"]
        FB[fact_booking.parquet]
        DD[dim_date.parquet]
        DH[dim_hotel.parquet]
        DG[dim_guest.parquet]
        DR[dim_room_type.parquet]
    end

    subgraph Derived["Derived in-process (no Postgres)"]
        AGG["occupancy_aggregation.py\nbuild_daily_occupancy()"]
        SYN["synthetic_data.py\ngenerate_restaurant_daily()\ngenerate_staffing_daily()"]
    end

    subgraph Features["data/features/*.parquet"]
        OCCF[occupancy_features.parquet]
        PRCF[pricing_features.parquet]
        RESF[restaurant_features.parquet]
        STAF[staff_features.parquet]
        CHNF[churn_features.parquet]
    end

    FB --> AGG
    DD --> AGG
    DH --> AGG
    AGG --> SYN
    AGG --> OCCF
    AGG --> PRCF
    DR --> PRCF
    SYN --> RESF
    SYN --> STAF
    DG --> CHNF
    FB --> CHNF
```

## Model → prediction → API flow

```mermaid
flowchart TD
    OCCF[occupancy_features.parquet] --> OccP["OccupancyPipeline\n(BasePipeline)"]
    PRCF[pricing_features.parquet] --> PrcP["PricingPipeline"]
    RESF[restaurant_features.parquet] --> RestP["RestaurantPipeline\n(per-meal loop)"]
    STAF[staff_features.parquet] --> StfP["StaffingPipeline"]
    CHNF[churn_features.parquet] --> ChnP["ChurnPipeline"]

    OccP --> OccM["OccupancyXGBoostModel\nOccupancyProphetModel"]
    PrcP --> PrcM["PricingXGBoostModel"]
    RestP --> RestM["RestaurantDemandModel x3\n(breakfast/lunch/dinner)"]
    StfP --> StfM["StaffingRegressionModel"]
    ChnP --> ChnM["ChurnRandomForestModel\nChurnXGBoostModel"]

    OccM --> Pkl["models/*.pkl (joblib)"]
    PrcM --> Pkl
    RestM --> Pkl
    StfM --> Pkl
    ChnM --> Pkl

    Pkl --> Pred["src/prediction/predict_*.py"]
    Pred --> API["api/main.py (FastAPI)"]
    API --> EP["POST /predict/{occupancy,pricing,restaurant,staff,churn}"]
```

## Star schema this phase reads from (unchanged from Phase 3)

```mermaid
erDiagram
    dim_guest ||--o{ fact_booking : "guest_key"
    dim_room_type ||--o{ fact_booking : "room_key"
    dim_hotel ||--o{ fact_booking : "branch_key"
    dim_date ||--o{ fact_booking : "check_in_date_key"
    dim_date ||--o{ fact_booking : "check_out_date_key"

    fact_booking {
        int surrogate_key PK
        int reservation_id
        string reservation_status
        int room_key FK
        int branch_key FK
        int guest_key FK
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
    dim_guest {
        int guest_key PK
        int guest_id
        int lifetime_bookings
        numeric lifetime_spend
        date first_stay_date
        date last_stay_date
    }
```

**Note:** no `fact_occupancy_daily`, `mart_occupancy_daily`,
`mart_revenue_daily`, `mart_restaurant_daily`, or `mart_staff_daily` table
exists anywhere in this project's warehouse parquet set — the Phase-3-era
scaffold assumed these, but they were never populated (see
`reports/final_phase4/known_limitations.md`). `occupancy_aggregation.py`
substitutes for the occupancy/revenue marts; `synthetic_data.py` substitutes
for the restaurant/staff marts.
