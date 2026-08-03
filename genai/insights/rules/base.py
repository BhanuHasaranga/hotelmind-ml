"""Shared types for insight rules."""

from dataclasses import dataclass, field


@dataclass
class Finding:
    category: str
    title: str
    metric: str
    metric_delta: float | None
    severity: str  # "low" | "medium" | "high" | "critical"
    branch_id: object | None = None
    supporting_data: dict = field(default_factory=dict)
    citation: str = ""
