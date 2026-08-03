"""Centralized logging: adds a rotating JSON file handler alongside the existing
console handler configured by src/utils/logging.py, gated behind
mlops_settings.MLOPS_JSON_LOGGING (default False -> zero-risk no-op).
"""

import json
import logging
import logging.handlers

from src.mlops.config.mlops_settings import mlops_settings
from src.utils.logging import get_logger

# Marker attribute set on handlers we attach, so repeated calls are idempotent.
_HANDLER_MARKER = "_hotelmind_mlops_json_handler"


class _JsonFormatter(logging.Formatter):
    """Minimal stdlib-only JSON formatter (no python-json-logger dependency)."""

    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "time": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload)


def configure_mlops_logging(log_name: str, log_file: str) -> None:
    """Attach a RotatingFileHandler with JSON formatting to the root logger for
    `log_file` under mlops_settings.logs_dir_path, if MLOPS_JSON_LOGGING is enabled.

    Safe to call multiple times (e.g. from multiple entrypoints/tests) -- duplicate
    handlers for the same log_file are not attached twice. No-op when the feature
    flag is disabled (the default), so existing console-only behavior is unaffected.
    """
    if not mlops_settings.MLOPS_JSON_LOGGING:
        return

    # Ensure src.utils.logging's console handler has already been configured.
    get_logger("")

    root_logger = logging.getLogger()
    target_path = mlops_settings.logs_dir_path / log_file

    for handler in root_logger.handlers:
        if getattr(handler, _HANDLER_MARKER, None) == str(target_path):
            return  # already attached

    file_handler = logging.handlers.RotatingFileHandler(
        target_path, maxBytes=10_000_000, backupCount=5
    )
    file_handler.setFormatter(_JsonFormatter())
    setattr(file_handler, _HANDLER_MARKER, str(target_path))
    root_logger.addHandler(file_handler)
    root_logger.info("MLOps JSON logging enabled for log_name=%s file=%s", log_name, target_path)
