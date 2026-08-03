# Prediction Flow

How a single API request is served, from HTTP request to trained-model
output.

```mermaid
sequenceDiagram
    participant C as Client (curl / Swagger UI)
    participant API as FastAPI router\n(api/routers/*.py)
    participant Pred as predict_*.py\n(src/prediction/)
    participant Model as BaseMLModel subclass
    participant Disk as models/*.pkl (joblib)

    C->>API: POST /predict/<domain>\n{JSON body}
    API->>API: Pydantic validates request\n(api/schemas.py)
    alt validation fails
        API-->>C: 422 Unprocessable Entity
    end

    API->>Pred: call predict_*()/forecast_*()/recommend_*()\nwith validated fields
    Pred->>Model: instantiate BaseMLModel subclass
    Pred->>Disk: model.load(path)
    alt model file missing
        Disk-->>Pred: FileNotFoundError
        Pred-->>API: propagate
        API-->>C: 503 "<Module> model(s) not trained yet"
    end
    Disk-->>Model: joblib.load() -> {model, encoders, scaler, feature_names}

    Pred->>Pred: build feature row(s) from request\n(calendar features, encoding, scaling\nas needed per domain)
    Pred->>Model: model.predict(X) / predict_proba(X) / predict_with_interval(X)
    Model-->>Pred: prediction array

    alt churn: guest_id not found
        Pred-->>API: ValueError
        API-->>C: 404 "guest_id <id> not found"
    end

    alt any other exception
        Pred-->>API: Exception
        API-->>C: 500 "<exception message>"\n(full traceback logged server-side)
    end

    Pred-->>API: result dict
    API->>API: wrap in Pydantic response model\n(api/schemas.py)
    API-->>C: 200 OK + JSON response
```

## Per-domain differences

| Domain | Prediction function | Extra I/O at request time |
|---|---|---|
| Occupancy | `predict_occupancy.py::forecast_occupancy` | reads `model.history["ds"].max()` from the loaded Prophet model to anchor the forecast start date — no parquet read |
| Pricing | `predict_pricing.py::recommend_price` | reads `data/raw/room_type_dim.csv` to resolve `room_type_id` |
| Restaurant | `predict_restaurant.py::forecast_restaurant_demand` | none — all 3 meal models loaded and predicted in sequence |
| Staffing | `predict_staffing.py::recommend_staffing` | none |
| Churn | `predict_churn.py::predict_churn` | reads `data/warehouse/dim_guest.parquet` + `fact_booking.parquet` to look up the guest and compute `total_nights`/`recency_days` |

No domain queries a live database at prediction time — the only I/O beyond
the model `.pkl` file itself is reading local parquet/CSV seed files.

## Startup behavior

`api/main.py`'s `lifespan` context calls `api/model_cache.py::warm_cache()`
on server start, which currently only **logs** whether `models/` exists —
it does not eagerly load any model into memory. The first request to each
endpoint pays the full `joblib.load()` cost; subsequent requests to the same
endpoint still reload from disk each time (no in-memory caching is
implemented — see `reports/final_phase4/known_limitations.md` for related
caveats, and treat this as a known future-work item, not a bug to fix in
this milestone).
