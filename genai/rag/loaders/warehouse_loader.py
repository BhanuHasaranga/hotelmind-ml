"""Loads warehouse mart data (via `genai.data_access.warehouse_source`) into
indexable text documents, e.g. "Branch 3 occupancy on 2026-07-01 was 82.4%."
"""

from genai.data_access.warehouse_source import get_mart
from src.utils.logging import get_logger

logger = get_logger(__name__)

_MART_TEMPLATES = {
    "mart_occupancy_daily": lambda r: (
        f"Branch {r.get('branch_id')} occupancy on {r.get('date')} was "
        f"{r.get('occupancy_pct')}% ({r.get('occupied_rooms')}/{r.get('total_rooms')} rooms occupied)."
    ),
    "mart_revenue_daily": lambda r: (
        f"Branch {r.get('branch_id')} total revenue on {r.get('date')} was {r.get('total_revenue')} "
        f"(RevPAR {r.get('revpar')}, ADR {r.get('avg_daily_rate')})."
    ),
    "mart_restaurant_daily": lambda r: (
        f"Branch {r.get('branch_id')} restaurant revenue on {r.get('date')} was {r.get('total_revenue')} "
        f"across {r.get('total_orders')} orders (avg order value {r.get('avg_order_value')})."
    ),
    "mart_staff_daily": lambda r: (
        f"Branch {r.get('branch_id')} {r.get('department_name')} on {r.get('date')} had "
        f"{r.get('present_employees')}/{r.get('scheduled_employees')} staff present "
        f"({r.get('attendance_rate_pct')}% attendance)."
    ),
}


def load_warehouse_documents(mart_name: str, max_rows: int = 200) -> list[dict]:
    if mart_name not in _MART_TEMPLATES:
        raise ValueError(f"No warehouse-loader template for mart {mart_name!r}")

    df = get_mart(mart_name)
    df = df.tail(max_rows)
    template = _MART_TEMPLATES[mart_name]

    docs = []
    for _, row in df.iterrows():
        text = template(row.to_dict())
        docs.append(
            {
                "text": text,
                "metadata": {
                    "source": mart_name,
                    "doc_type": "warehouse",
                    "branch_id": row.get("branch_id"),
                    "date": str(row.get("date")),
                },
            }
        )
    return docs


def load_all_warehouse_documents(max_rows_per_mart: int = 200) -> list[dict]:
    docs = []
    for mart_name in _MART_TEMPLATES:
        try:
            docs.extend(load_warehouse_documents(mart_name, max_rows=max_rows_per_mart))
        except Exception as exc:
            logger.warning("Skipping mart %s for RAG indexing: %s", mart_name, exc)
    return docs
