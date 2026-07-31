from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.database import writer


def _mock_conn(exists: bool) -> MagicMock:
    conn = MagicMock()
    cursor_cm = MagicMock()
    cursor = MagicMock()
    cursor.fetchone.return_value = (exists,)
    cursor_cm.__enter__.return_value = cursor
    conn.cursor.return_value = cursor_cm
    return conn


def test_table_exists_true_false():
    conn_true = _mock_conn(True)
    assert writer.table_exists(conn_true, "dim_hotel") is True

    conn_false = _mock_conn(False)
    assert writer.table_exists(conn_false, "dim_hotel") is False


def test_bulk_upsert_raises_warehouse_write_error_when_table_missing():
    conn = _mock_conn(False)
    df = pd.DataFrame({"pk": [1], "val": ["a"]})
    with pytest.raises(writer.WarehouseWriteError):
        writer.bulk_upsert(conn, "missing_table", df, ["pk"])


def test_bulk_upsert_calls_execute_values_with_expected_row_count():
    conn = _mock_conn(True)
    df = pd.DataFrame({"pk": [1, 2, 3], "val": ["a", "b", "c"]})

    with patch("src.database.writer.execute_values") as mock_execute_values:
        rows_written = writer.bulk_upsert(conn, "dim_hotel", df, ["pk"])

    assert rows_written == 3
    mock_execute_values.assert_called_once()
    _, args, _ = mock_execute_values.mock_calls[0]
    values_arg = args[2]
    assert len(values_arg) == 3
