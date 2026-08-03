"""Tests for src/mlops/observability/logging_config.py::configure_mlops_logging."""

import logging

import pytest

from src.mlops.observability.logging_config import _HANDLER_MARKER, configure_mlops_logging


@pytest.fixture(autouse=True)
def isolated_logs_dir(tmp_path, monkeypatch):
    from src.mlops.config import mlops_settings as settings_module

    logs_dir = tmp_path / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(
        type(settings_module.mlops_settings),
        "logs_dir_path",
        property(lambda self: logs_dir),
    )
    return logs_dir


@pytest.fixture(autouse=True)
def clean_root_handlers():
    """Remove any handlers our own tests attach so tests don't leak state
    into each other or the rest of the suite."""
    root = logging.getLogger()
    before = list(root.handlers)
    yield
    for handler in list(root.handlers):
        if getattr(handler, _HANDLER_MARKER, None) is not None and handler not in before:
            root.removeHandler(handler)
            handler.close()


def test_configure_mlops_logging_noop_when_flag_disabled(monkeypatch):
    from src.mlops.config import mlops_settings as settings_module

    monkeypatch.setattr(settings_module.mlops_settings, "MLOPS_JSON_LOGGING", False)

    root = logging.getLogger()
    before_count = len(root.handlers)
    configure_mlops_logging("test", "test.log")
    assert len(root.handlers) == before_count


def test_configure_mlops_logging_attaches_handler_when_enabled(monkeypatch, isolated_logs_dir):
    from src.mlops.config import mlops_settings as settings_module

    monkeypatch.setattr(settings_module.mlops_settings, "MLOPS_JSON_LOGGING", True)

    root = logging.getLogger()
    before_count = len(root.handlers)
    configure_mlops_logging("test", "test_enabled.log")
    after_count = len(root.handlers)

    assert after_count == before_count + 1
    assert (isolated_logs_dir / "test_enabled.log").exists()


def test_configure_mlops_logging_idempotent_no_duplicate_handlers(monkeypatch, isolated_logs_dir):
    from src.mlops.config import mlops_settings as settings_module

    monkeypatch.setattr(settings_module.mlops_settings, "MLOPS_JSON_LOGGING", True)

    root = logging.getLogger()
    before_count = len(root.handlers)
    configure_mlops_logging("test", "test_idempotent.log")
    configure_mlops_logging("test", "test_idempotent.log")
    configure_mlops_logging("test", "test_idempotent.log")
    after_count = len(root.handlers)

    assert after_count == before_count + 1  # only one handler attached, not three


def test_json_formatter_produces_valid_json(monkeypatch, isolated_logs_dir):
    import json

    from src.mlops.config import mlops_settings as settings_module

    monkeypatch.setattr(settings_module.mlops_settings, "MLOPS_JSON_LOGGING", True)
    configure_mlops_logging("test", "test_format.log")

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    logger = logging.getLogger("test.logging_config.format")
    logger.setLevel(logging.INFO)
    logger.info("hello world")

    log_path = isolated_logs_dir / "test_format.log"
    lines = [line for line in log_path.read_text().splitlines() if line.strip()]
    assert lines
    last = json.loads(lines[-1])
    assert last["message"] == "hello world"
    assert last["level"] == "INFO"
