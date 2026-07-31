# HotelMind ML — Phase 4 (Machine Learning)

Phase 4 of the HotelMind AI portfolio project. Builds five ML modules —
occupancy forecasting, dynamic pricing, restaurant demand, staff optimization,
and customer churn — reading **only** from the Phase 3 warehouse marts
(`mart_occupancy_daily`, `mart_revenue_daily`, `mart_restaurant_daily`,
`mart_staff_daily`) and dimension/fact tables (`dim_guest`, `fact_booking`) in
the `hotelmind_warehouse` Postgres schema produced by `hotelmind-data`.

Out of scope for this phase: LLMs, LangChain, RAG, MLflow, Kafka, AWS,
Terraform, Docker changes, monitoring. Those belong to later phases.

## Documentation

- [doc/architecture.md](doc/architecture.md) — pipeline design, folder structure, model storage convention, scaffolded-only components
- [doc/running.md](doc/running.md) — setup, training, prediction API, tests, evaluation
- [doc/assumptions.md](doc/assumptions.md) — every synthetic/derived field and why it exists

## Quickstart

```bash
cd hotelmind-ml
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
cp .env.example .env           # then fill in WAREHOUSE_DB_* credentials

python -m src.training.train_occupancy --branch-id 1
uvicorn api.main:app --reload
pytest tests/ -v
```

See [doc/running.md](doc/running.md) for full details on every module's training command and all five prediction endpoints.
