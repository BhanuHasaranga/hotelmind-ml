"""Prometheus instrumentation for the FastAPI app: auto request metrics via
prometheus_fastapi_instrumentator, plus custom prediction/resource metrics.
"""

import psutil
from prometheus_client import Counter, Gauge, Histogram
from prometheus_fastapi_instrumentator import Instrumentator

PREDICTION_COUNTER = Counter(
    "hotelmind_predictions_total", "Predictions served", ["domain", "model_version"]
)
PREDICTION_LATENCY = Histogram(
    "hotelmind_prediction_latency_seconds", "Prediction latency seconds", ["domain"]
)
PREDICTION_ERRORS = Counter("hotelmind_prediction_errors_total", "Prediction errors", ["domain"])
MODEL_VERSION_GAUGE = Gauge(
    "hotelmind_model_version", "Currently loaded production model version", ["model_name"]
)
CPU_USAGE_GAUGE = Gauge("hotelmind_process_cpu_percent", "Process CPU percent")
RAM_USAGE_GAUGE = Gauge("hotelmind_process_ram_mb", "Process RSS megabytes")

_process = psutil.Process()


def instrument_app(app) -> None:
    """Wrap the FastAPI app with automatic request/latency instrumentation and
    expose a /metrics endpoint."""
    Instrumentator().instrument(app).expose(app, endpoint="/metrics")


def record_prediction(domain: str, model_version: str, latency_s: float, error: bool = False) -> None:
    """Record a single prediction's latency and outcome. Never raises."""
    try:
        PREDICTION_LATENCY.labels(domain=domain).observe(latency_s)
        if error:
            PREDICTION_ERRORS.labels(domain=domain).inc()
        else:
            PREDICTION_COUNTER.labels(domain=domain, model_version=str(model_version)).inc()
    except Exception:
        pass


def update_resource_gauges() -> None:
    """Refresh process CPU/RAM gauges. Never raises."""
    try:
        CPU_USAGE_GAUGE.set(_process.cpu_percent())
        RAM_USAGE_GAUGE.set(_process.memory_info().rss / (1024 * 1024))
    except Exception:
        pass
