# Phase 4 Pipeline

The requested "Hotel System → Warehouse → Feature Engineering → ML Models →
Prediction API" flow, as actually implemented in this repository.

```mermaid
flowchart TD
    subgraph Warehouse["Phase 3 Warehouse (data/warehouse/*.parquet)"]
        FB[fact_booking.parquet]
        DD[dim_date.parquet]
        DH[dim_hotel.parquet]
        DG[dim_guest.parquet]
        DR[dim_room_type.parquet]
    end

    subgraph FeatEng["Feature Engineering (src/pipelines/feature_engineering.py)"]
        Agg["occupancy_aggregation.py\nderives daily occupancy_pct/\nrevenue from fact_booking"]
        Syn["synthetic_data.py\ngenerates Restaurant/Staffing\nseeds from real occupancy signal"]
        Occ[build_occupancy_features]
        Prc[build_pricing_features]
        Res[build_restaurant_features]
        Stf[build_staff_features]
        Chn[build_churn_features]
    end

    FB --> Agg
    DD --> Agg
    DH --> Agg
    Agg --> Syn
    Agg --> Occ
    Agg --> Prc
    DR --> Prc
    Syn --> Res
    Syn --> Stf
    DG --> Chn
    FB --> Chn

    subgraph FeatureData["data/features/*.parquet"]
        OccF[occupancy_features.parquet]
        PrcF[pricing_features.parquet]
        ResF[restaurant_features.parquet]
        StfF[staff_features.parquet]
        ChnF[churn_features.parquet]
    end

    Occ --> OccF
    Prc --> PrcF
    Res --> ResF
    Stf --> StfF
    Chn --> ChnF

    subgraph MLModels["ML Models (models/*.pkl)"]
        OccM["OccupancyXGBoostModel\nOccupancyProphetModel"]
        PrcM[PricingXGBoostModel]
        ResM["RestaurantDemandModel x3"]
        StfM[StaffingRegressionModel]
        ChnM["ChurnRandomForestModel\nChurnXGBoostModel"]
    end

    OccF --> OccM
    PrcF --> PrcM
    ResF --> ResM
    StfF --> StfM
    ChnF --> ChnM

    subgraph API["Prediction API (api/main.py)"]
        EP["POST /predict/{occupancy,pricing,\nrestaurant,staff,churn}"]
    end

    OccM --> EP
    PrcM --> EP
    ResM --> EP
    StfM --> EP
    ChnM --> EP
```

## Notes

- **No `mart_occupancy_daily`/`mart_revenue_daily`/`mart_restaurant_daily`/
  `mart_staff_daily` tables exist anywhere** — the boxes labeled
  `occupancy_aggregation.py` and `synthetic_data.py` substitute for them,
  computed entirely in-process from parquet. See
  `reports/model_discovery/pipeline_diagram.md` for the original discovery
  of this gap.
- `feature_engineering.py` is a **standalone orchestration layer** — it
  calls the same `src/features/*.py` functions each domain pipeline uses
  internally, and additionally persists the result to
  `data/features/*.parquet` for inspection. Pipelines never call into
  `feature_engineering.py`, and vice versa — both call into `src/features/`.
- Full column-level detail: `reports/model_discovery/feature_mapping.md`.
