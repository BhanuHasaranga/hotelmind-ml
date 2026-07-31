# HotelMind ML — Phase 4 (Machine Learning)

Phase 4 of the HotelMind AI portfolio project. Builds five ML modules —
occupancy forecasting, dynamic pricing, restaurant demand, staff optimization,
and customer churn — reading **only** from local parquet files: the cleaned
Phase 3 dataset (`data/processed/hotel_bookings_clean.parquet`) and the
warehouse dimension/fact tables (`data/warehouse/{dim_date,dim_hotel,
dim_guest,dim_room_type,fact_booking}.parquet`). No live Postgres connection
is required to train, predict, or run the API.

Occupancy and revenue are not pre-aggregated anywhere in the warehouse, so a
new module (`src/features/occupancy_aggregation.py`) derives a daily
occupancy/revenue "mart" directly from `fact_booking` in pandas. Restaurant
and Staffing have no real underlying data anywhere in this project, so their
training data is a small, clearly-labeled **synthetic** seed
(`src/pipelines/synthetic_data.py`), driven by the real derived occupancy
signal rather than pure noise. See
[reports/final_phase4/known_limitations.md](reports/final_phase4/known_limitations.md)
for the full list of assumptions this implies.

Out of scope for this phase: LLMs, LangChain, RAG, MLflow, Kafka, AWS,
Terraform, Docker changes, monitoring. Those belong to later phases.

## Documentation

- [doc/architecture.md](doc/architecture.md) — pipeline design, folder structure, model storage convention, scaffolded-only components
- [doc/running.md](doc/running.md) — setup, training, prediction API, tests, evaluation
- [doc/assumptions.md](doc/assumptions.md) — every synthetic/derived field and why it exists
- [reports/final_phase4/phase4_summary.md](reports/final_phase4/phase4_summary.md) — what was built, deliverable checklist
- [reports/final_phase4/training_results.md](reports/final_phase4/training_results.md) — every trained model's real metrics
- [reports/final_phase4/api_examples.md](reports/final_phase4/api_examples.md) — sample request/response per endpoint
- [reports/final_phase4/known_limitations.md](reports/final_phase4/known_limitations.md) — synthetic data, date-anchoring, and other caveats

## Quickstart

```bash
cd hotelmind-ml
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt

# 1. Build the standalone feature datasets (data/features/*.parquet)
python -m src.pipelines.feature_engineering

# 2. Train every module (writes models/*.pkl + reports/latest_<module>.json)
python -m src.training.train_occupancy --branch-id 1 --start-date 2015-07-01 --end-date 2017-09-13
python -m src.training.train_pricing --branch-id 1 --start-date 2015-07-01 --end-date 2017-09-13
python -m src.training.train_restaurant --branch-id 1 --start-date 2015-07-01 --end-date 2017-09-13
python -m src.training.train_staffing --branch-id 1 --start-date 2015-07-01 --end-date 2017-09-13
python -m src.training.train_churn

# 3. Generate the occupancy forecast report (reports/models/occupancy_forecast.csv)
python -m src.pipelines.generate_occupancy_report

# 4. Run the prediction API
uvicorn api.main:app --reload

# 5. Tests
pytest tests/ -v
```

`--start-date`/`--end-date` default to `2023-01-01`..today in each training
CLI, which does **not** overlap the canonical dataset's real date range
(2015-07-01 to 2017-09-13) — pass explicit dates matching your data, or
training will run over zero rows. See
[reports/final_phase4/known_limitations.md](reports/final_phase4/known_limitations.md).

See [doc/running.md](doc/running.md) for full details on every module's training command and all five prediction endpoints.
