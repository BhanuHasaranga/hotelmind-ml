from src.utils.logging import get_logger

logger = get_logger(__name__)

_cache: dict[str, object] = {}


def warm_cache() -> None:
    """Pre-touch model directories at startup so missing-model errors surface
    immediately in logs rather than on the first request. Actual model
    objects are loaded lazily per-request by the predict_* functions, since
    each domain module already owns its own load() call and artifact schema
    (e.g. Prophet needs a different load path than XGBoost/joblib payloads).
    """
    from src.config.settings import settings

    model_dir = settings.model_dir_path
    logger.info("Model directory: %s (exists=%s)", model_dir, model_dir.exists())
