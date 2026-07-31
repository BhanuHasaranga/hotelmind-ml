import pytest

import src.prediction.predict_churn as predict_churn_module


def test_predict_churn_does_not_touch_database(monkeypatch):
    """Guards against reintroducing a live-DB call at prediction time --
    Phase 4 requires predictions to come only from warehouse parquet."""

    def _fail_if_called(*args, **kwargs):
        raise AssertionError("predict_churn must not call get_connection()/run_query()")

    monkeypatch.setattr("src.database.postgres.get_connection", _fail_if_called, raising=False)

    with pytest.raises(ValueError):
        predict_churn_module.predict_churn("__nonexistent_guest_id__")
