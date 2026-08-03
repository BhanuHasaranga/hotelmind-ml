# Phase 6 Acceptance Report — MLOps Platform

## Summary

Phase 6 (MLOps) of the HotelMind AI portfolio project is **complete**. All 20
milestones (M1-M20) from the approved plan are implemented. Everything that
can be exercised on this host (Python 3.14, Windows, no Docker daemon
reachable in this sandbox) has been run for real and verified; everything
that fundamentally requires Docker (MLflow/Evidently runtime execution,
Airflow scheduling, the full `docker compose up -d` stack) has been
structurally/syntax-validated but not launched — this is expected and
explicitly scoped in "Known limitations" below.

## Milestone checklist

| Milestone | Description | Status |
|---|---|---|
| M1 | Folder structure + `MLOpsSettings` | ✅ Complete |
| M2 | MLflow integration (`MLflowTracker`, `promote.py`) | ✅ Complete (code+mocked-tests; real mlflow execution is Docker-only) |
| M3 | Model registry (`ModelRegistry`, `model_factory`, `predict_*.py` updated) | ✅ Complete, fully tested on this host |
| M4 | Dataset versioning | ✅ Complete, fully tested on this host |
| M5 | MLOps pipeline classes (mixin + 5 concrete wrappers) | ✅ Complete, orchestration tested with mocks |
| M6 | Airflow DAGs (4 DAGs) | ✅ Complete; compile-checked, never scheduled/executed by a running scheduler |
| M7 | GitHub Actions CI | ✅ Complete (prior task; not re-verified this session) |
| M8 | Prometheus monitoring | ✅ Complete, fully tested on this host (prometheus_client installed) |
| M9 | Data drift detection (Evidently) | ✅ Complete (code+mocked-tests; real evidently execution is Docker-only) |
| M10 | Prediction drift (pure Python) | ✅ Complete, fully tested on this host |
| M11 | Model evaluation dashboard | ✅ Complete, fully tested on this host |
| M12 | Automatic retraining (`check_and_retrain`) | ✅ Complete, fully tested with mocks |
| M13 | Rollback CLI (`rollback.py`) | ✅ Complete, tested + run live against the seeded registry |
| M14 | API response metadata (`meta` field) | ✅ Complete, verified live via `tests/test_api_mlops.py` |
| M15 | Centralized logging | ✅ Complete, fully tested on this host |
| M16 | Tests (>90% coverage on `src/mlops`) | ✅ Complete — **95.70%** achieved |
| M17 | Docker Compose | ✅ Complete; `docker-compose.yml` written, never actually launched in this sandbox |
| M18 | Documentation (`docs/phase6.md`) | ✅ Complete (this session) |
| M19 | Code quality (cross-cutting) | ✅ Complete — Repository/Factory patterns, DI via constructor params, full type hints/docstrings, specific exceptions preserved |
| M20 | Seed script + acceptance report | ✅ Complete (this session) |

## Coverage on `src/mlops`

```
pytest tests/mlops/ tests/test_api_mlops.py --cov=src/mlops --cov-report=term-missing --cov-fail-under=90
```

**Result: 95.70% (744 statements, 32 missed), 101 tests passed.** Threshold
(90%) met with margin.

```
Name                                               Stmts   Miss  Cover   Missing
--------------------------------------------------------------------------------
src\mlops\__init__.py                                  0      0   100%
src\mlops\config\__init__.py                           0      0   100%
src\mlops\config\mlops_settings.py                    59      0   100%
src\mlops\deployment\__init__.py                       0      0   100%
src\mlops\metrics\__init__.py                          0      0   100%
src\mlops\metrics\evaluation_dashboard.py             37      3    92%   45-47
src\mlops\monitoring\__init__.py                       0      0   100%
src\mlops\monitoring\drift_detector.py                42      1    98%   55
src\mlops\monitoring\metrics_middleware.py            26      2    92%   48-49
src\mlops\monitoring\prediction_drift.py              90      5    94%   82-83, 87-88, 113
src\mlops\observability\__init__.py                    0      0   100%
src\mlops\observability\logging_config.py             26      1    96%   28
src\mlops\pipelines\__init__.py                        0      0   100%
src\mlops\pipelines\churn_mlops_pipeline.py            5      0   100%
src\mlops\pipelines\mlops_pipeline_mixin.py           51      0   100%
src\mlops\pipelines\occupancy_mlops_pipeline.py        5      0   100%
src\mlops\pipelines\pricing_mlops_pipeline.py          5      0   100%
src\mlops\pipelines\restaurant_mlops_pipeline.py      44      0   100%
src\mlops\pipelines\retrain_trigger.py                71      8    89%   49-50, 56, 66, 70, 104-105, 136
src\mlops\pipelines\staffing_mlops_pipeline.py         5      0   100%
src\mlops\registry\__init__.py                         0      0   100%
src\mlops\registry\model_factory.py                   15      2    87%   27-28
src\mlops\registry\model_registry.py                 166      7    96%   189, 202-205, 257-258
src\mlops\tracking\__init__.py                         0      0   100%
src\mlops\tracking\mlflow_tracker.py                  61      3    95%   39-41
src\mlops\validation\__init__.py                       0      0   100%
src\mlops\validation\dataset_version.py               36      0   100%
--------------------------------------------------------------------------------
TOTAL                                                744     32    96%
```

Remaining gaps and why they're impractical to close further on this host:

- `retrain_trigger.py` 49-50/56/66/70/104-105/136: rare `except Exception`
  guard branches inside `_accuracy_drop_breached` (e.g.
  `production_record.version == latest_record.version`) and the
  `MLflowTracker` construction line inside the retrain-success path when
  `getattr(pipeline, "mlflow_experiment_name", ...)` — all reachable, but
  each additional branch requires a new mocked scenario for marginal
  additional confidence; the module's core decision logic (drift breach,
  accuracy breach, promote/no-promote, exception handling) is already
  covered by 7 dedicated tests.
- `model_registry.py` 189/202-205/257-258: an `archive_model` branch for a
  version that isn't currently STAGING/PRODUCTION (the "already-archived,
  re-archived" edge case) and one defensive `except Exception` inside
  `load_production`'s registry-read guard — both defensive/rare paths.
- `model_factory.py` 27-28: the `ValueError` raise for an unknown model name
  — trivial to hit but not currently asserted; low value to add given the
  factory's simplicity.
- `mlflow_tracker.py` 39-41: `_git_commit()`'s exception fallback branch
  (this repo is genuinely not a git repository, so the `subprocess.run`
  non-zero-return-code path is exercised, but the `except Exception` path
  around a `subprocess` call itself failing to launch is not).
- `evaluation_dashboard.py` 45-47: `_dashboards_dir()`'s own `mkdir` line,
  shadowed in tests by monkeypatching the whole function for isolation.
- `drift_detector.py` 55: the fallback branch of `generate_data_drift_report`
  that searches for `share_of_drifted_columns` when `DataDriftTable` isn't
  found in Evidently's own report dict shape — inherently Evidently-internal
  and only meaningfully exercisable against a real Evidently report object
  (Docker-only on this host).
- `logging_config.py` 28: one line inside `get_logger("")`'s call chain from
  `src/utils/logging.py`, outside this module's own logic.
- `metrics_middleware.py` 48-49: `update_resource_gauges()`'s `except
  Exception: pass` branch — would require actually breaking `psutil` mid-call
  to hit, not a meaningful test to add.
- `prediction_drift.py` 82-83/87-88/113: `_predictions_dir()`'s own
  `mkdir`/return line inside functions that are otherwise directly tested,
  plus one guard branch when zero timestamped records parse successfully out
  of a non-empty log (distinct from the "insufficient windows" case already
  covered).

None of these gaps involve the `mlflow`/`evidently` real-execution paths that
are the primary, expected, and explicitly-scoped host limitation (see below)
— they are ordinary defensive-code branches.

## New ports/services (from `docker-compose.yml`)

| service | image/build | host port | container port |
|---|---|---|---|
| backend | build: `.` (`Dockerfile`, `python:3.12-slim`) | 8110 | 8000 |
| mlflow | `ghcr.io/mlflow/mlflow:v2.17.2` | 5500 | 5000 |
| mlflow-db | `postgres:16-alpine` | 5544 | 5432 |
| airflow-webserver | build: `airflow/Dockerfile` | 8180 | 8080 |
| airflow-scheduler | build: `airflow/Dockerfile` | (none exposed) | — |
| airflow-init | build: `airflow/Dockerfile` | (one-shot, no ports) | — |
| airflow-db | `postgres:16-alpine` | 5545 | 5432 |
| prometheus | `prom/prometheus:latest` | 9190 | 9090 |
| grafana | `grafana/grafana:latest` | 3300 | 3000 |

All distinct from the sibling `hotelmind-data` stack's ports (8080, 5433,
5434, 9000, 9001), confirmed by inspection of both `docker-compose.yml` files.

## Registry state: before and after seeding

**Before this session's seed run**: `models/registry/`, `models/staging/`,
`models/production/`, `models/archived/` each contained only their
`.gitkeep` placeholder — confirmed clean via directory listing before running
`scripts/seed_registry_from_legacy.py`, i.e. no stray entries from any prior
task's manual verification.

**After**: all 9 legacy models registered as registry version 1 and promoted
to Production:

```
=== AFTER seed_registry_from_legacy.py ===
models/registry/churn_random_forest/1/model.pkl + index.json
models/registry/churn_xgboost/1/model.pkl + index.json
models/registry/occupancy_prophet/1/model.pkl + index.json
models/registry/occupancy_xgboost/1/model.pkl + index.json
models/registry/pricing_xgboost/1/model.pkl + index.json
models/registry/restaurant_breakfast/1/model.pkl + index.json
models/registry/restaurant_dinner/1/model.pkl + index.json
models/registry/restaurant_lunch/1/model.pkl + index.json
models/registry/staffing_regression/1/model.pkl + index.json

models/production/churn_random_forest.pkl
models/production/churn_xgboost.pkl
models/production/occupancy_prophet.pkl
models/production/occupancy_xgboost.pkl
models/production/pricing_xgboost.pkl
models/production/restaurant_breakfast.pkl
models/production/restaurant_dinner.pkl
models/production/restaurant_lunch.pkl
models/production/staffing_regression.pkl
```

Re-running the script confirmed idempotency — all 9 models were reported
`skipped (already registered, 1 version(s))`, zero duplicate versions
created.

## `scripts/seed_registry_from_legacy.py` — real run output

```
$ python scripts/seed_registry_from_legacy.py
Importing plotly failed. Interactive plots will not work.
2026-08-03 20:19:56,896 | INFO | ... Registered model=occupancy_prophet version=1 stage=staging
2026-08-03 20:19:56,911 | INFO | ... Promoted model=occupancy_prophet version=1 to stage=production
...(all 9 models, same pattern)...

Seed registry from legacy .pkl files -- summary
======================================================================
model_name            status
----------------------------------------------------------------------
occupancy_prophet     registered + promoted to production
occupancy_xgboost     registered + promoted to production
pricing_xgboost       registered + promoted to production
restaurant_breakfast  registered + promoted to production
restaurant_lunch      registered + promoted to production
restaurant_dinner     registered + promoted to production
staffing_regression   registered + promoted to production
churn_random_forest   registered + promoted to production
churn_xgboost         registered + promoted to production
======================================================================
Registered+promoted: 9   Skipped: 0   Total: 9
```

(The `Importing plotly failed` line is Prophet's own harmless startup log
noise, pre-existing since Phase 4, unrelated to Phase 6.)

## `rollback.py` / `promote.py` — real transcripts against the seeded registry

```
$ python rollback.py --model occupancy
Rollback failed for model_name='occupancy_prophet': No archived version available to roll back to for model_name='occupancy_prophet'
```

```
$ python promote.py --model occupancy
No Staging version found for model_name='occupancy_prophet'. Nothing to promote.
```

Both outputs are the **correct, honest** result for a freshly-seeded registry
where every model has exactly one version, already in Production: there is
nothing archived to roll back to, and nothing sitting in Staging to promote.
Both CLIs exit non-zero (`1`) and print a clear explanatory message rather
than silently no-op'ing — this exact behavior (including the version-count
edge cases) is covered by `tests/mlops/test_rollback_cli.py` and
`tests/mlops/test_promote_cli.py` using an isolated `tmp_path` registry with
multiple versions, where the full promote/rollback/archive state-machine is
exercised successfully.

## Full test suite

```
$ python -m pytest tests/ -v
...
====================== 404 passed, 22 warnings in 17.9s ======================
```

**404/404 tests passing** — every Phase 1-5 test (`tests/test_*.py`,
`tests/genai/`) plus all new Phase 6 tests (`tests/mlops/*`,
`tests/test_api_mlops.py`), run against the now-seeded real registry, with
zero failures and zero behavioral regressions. The 22 warnings are
pre-existing (Pydantic `ArbitraryTypeWarning` from a GenAI test, and NumPy
`DeprecationWarning`s from joblib's pickle codepath) and unrelated to Phase 6
changes.

## Known limitations

1. **MLflow/Evidently could not be verified end-to-end on this host.**
   Python 3.14 has no prebuilt wheels for `mlflow` or `evidently`, and
   building their transitive dependencies from source requires `cmake`,
   which is not available in this sandbox. `mlflow_tracker.py`,
   `drift_detector.py`'s report-generation function, and any code path that
   transitively imports either package (the MLOps pipeline mixin, the
   retrain trigger, the Airflow DAG task callables) cannot be *executed*
   against the real libraries here. Every such module was instead tested by
   stubbing `mlflow`/`evidently` in `sys.modules` with `unittest.mock.MagicMock`
   and asserting the correct sequence and arguments of calls
   (`mlflow.log_params`, `mlflow.log_metrics`, `Report(...).run(...).save_html(...)`,
   etc.) — this verifies *the code's own logic* is correct, but does not
   prove the real libraries accept the exact call signatures used, or that
   version-specific behavior (e.g. Evidently's `report.as_dict()` shape
   across versions) matches what `drift_detector.py` assumes. Real
   verification requires running inside the `python:3.12-slim`-based Docker
   images this project ships (`Dockerfile`, `airflow/Dockerfile`), where
   `requirements.txt`/`requirements-training.txt` pin `mlflow>=2.17,<3.0` and
   `evidently>=0.4,<0.5` against a supported Python version.

2. **The Docker Compose stack was never actually launched in this sandbox.**
   No Docker daemon was reachable in this environment. `docker-compose.yml`
   was written and is believed correct based on careful reading of each
   image's documented configuration (MLflow server flags, Airflow's
   `LocalExecutor` + `SQL_ALCHEMY_CONN` env vars, Postgres healthchecks,
   Prometheus/Grafana provisioning mounts), and `docker compose config`
   syntax validation was performed in a prior session (per the plan's CI job
   `docker-build`), but the 7-service stack has never been brought up
   together, health-checked, or smoke-tested end-to-end
   (`curl localhost:8110/health`, MLflow UI reachability, Airflow UI login,
   Grafana dashboard rendering, Prometheus scrape success). A human with
   Docker Desktop running should perform this as the next concrete step —
   see "What to do next" below.

3. **Airflow DAGs were compile-checked, never scheduled/executed by a live
   scheduler.** All 4 DAG files (`daily_training.py`, `weekly_full_retrain.py`,
   `monthly_model_validation.py`, `nightly_data_validation.py`) import
   cleanly as plain Python modules (the `from airflow import DAG` /
   `from airflow.operators.python import PythonOperator` imports and DAG
   definitions were structurally reviewed), but no Airflow scheduler process
   has ever parsed them, no DAG run has ever been triggered (manually or on
   schedule), and the task callables' actual behavior when run inside the
   Airflow worker's process (with `PYTHONPATH=/opt/airflow/project` resolving
   `from src.mlops...` imports) has not been observed. This requires the
   Docker Compose stack (limitation 2) to be running.

4. **Coverage gap in `src/mlops`**: 95.70% achieved against a 90% threshold,
   with the remaining ~4.3 percentage points in defensive/rare exception
   branches (detailed line-by-line in the Coverage section above) — none of
   which touch the mlflow/evidently real-execution boundary (limitation 1),
   they are ordinary Python exception-handling paths that would require
   contrived failure injection for marginal additional confidence.

## What a human should do next

1. Start Docker Desktop, then from `hotelmind-ml/`: `docker compose up -d`.
   Confirm all 7 services report healthy (`docker compose ps`).
2. `docker compose exec backend python scripts/seed_registry_from_legacy.py`
   (or run it on the host first, then bring the stack up — the registry is a
   bind-mounted flat-file store, so either order works) if not already seeded.
3. Smoke-test: `curl localhost:8110/health`, `curl localhost:8110/metrics`,
   open `http://localhost:5500` (MLflow), `http://localhost:8180` (Airflow,
   `admin`/`admin`), `http://localhost:3300` (Grafana, `admin`/`admin`),
   `http://localhost:9190` (Prometheus).
4. `docker compose exec backend python scripts/train_all.py --mlops` and
   confirm a new MLflow run appears in the UI with params/metrics/artifacts,
   and a new Staging version appears under `models/registry/`.
5. In the Airflow UI, manually trigger `nightly_data_validation` and
   `daily_training` once each; confirm drift HTML reports land in
   `reports/drift/` and a fresh dashboard JSON lands in `dashboards/`.
6. Only after step 4 confirms a real MLflow-tracked run exists, exercise
   `promote.py`/`rollback.py` against a registry with more than one version
   per model to see a real (non-degenerate) promotion/rollback transcript.

## Production-readiness declaration

**Phase 6 is complete for its stated scope**: a local-first, Docker-only
MLOps platform layered non-destructively over the existing Phase 1-5
prediction service — registry, tracking, versioning, orchestration,
monitoring, and drift detection all implemented, code-reviewed for
correctness via 101 dedicated tests (95.70% coverage) plus a clean 404/404
full-suite run, and the one-time registry seed script executed for real
against the actual `models/` directory with a confirmed clean-before /
correctly-populated-after state transition.

It is **not yet operationally proven** in the sense of a running, scraped,
scheduled, end-to-end deployment — that final verification pass requires a
host with a working Docker daemon and is the explicit, correctly-scoped
hand-off to a human reviewer (see "What a human should do next").
