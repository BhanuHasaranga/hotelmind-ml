"""Lightweight markdown report writers for Phase 4 feature/model/documentation
deliverables. Mirrors the plain-function/f-string-table pattern used in
src/pipelines/warehouse_reports.py from the Phase 3 milestone -- no external
templating library.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config.settings import settings


def _write(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def write_feature_dictionary(feature_dfs: dict[str, pd.DataFrame]) -> Path:
    lines = ["# Feature Dictionary", ""]
    for domain, df in feature_dfs.items():
        lines.append(f"## {domain}")
        lines.append("")
        lines.append("| Column | Dtype |")
        lines.append("|---|---|")
        for col, dtype in df.dtypes.items():
            lines.append(f"| {col} | {dtype} |")
        lines.append("")
    return _write(settings.features_reports_dir_path / "feature_dictionary.md", "\n".join(lines))


def write_feature_statistics(feature_dfs: dict[str, pd.DataFrame]) -> Path:
    lines = ["# Feature Statistics", ""]
    for domain, df in feature_dfs.items():
        lines.append(f"## {domain}")
        lines.append("")
        lines.append(f"Rows: {len(df)}")
        lines.append("")
        numeric = df.select_dtypes(include="number")
        if not numeric.empty:
            desc = numeric.describe().T[["mean", "std", "min", "max"]].round(3)
            lines.append("| Column | Mean | Std | Min | Max |")
            lines.append("|---|---|---|---|---|")
            for col, row in desc.iterrows():
                lines.append(f"| {col} | {row['mean']} | {row['std']} | {row['min']} | {row['max']} |")
        lines.append("")
    return _write(settings.features_reports_dir_path / "feature_statistics.md", "\n".join(lines))


def write_correlation_report(feature_dfs: dict[str, pd.DataFrame]) -> Path:
    lines = ["# Correlation Report", ""]
    for domain, df in feature_dfs.items():
        numeric = df.select_dtypes(include="number")
        if numeric.shape[1] < 2:
            continue
        corr = numeric.corr(numeric_only=True)
        lines.append(f"## {domain}")
        lines.append("")
        lines.append("Top 10 absolute pairwise correlations (excluding self-correlation):")
        lines.append("")
        mask = np.eye(len(corr), dtype=bool)
        pairs = corr.mask(mask)
        stacked = pairs.stack().abs().sort_values(ascending=False)
        seen = set()
        lines.append("| Feature A | Feature B | |corr| |")
        lines.append("|---|---|---|")
        count = 0
        for (a, b), value in stacked.items():
            key = frozenset((a, b))
            if key in seen:
                continue
            seen.add(key)
            lines.append(f"| {a} | {b} | {value:.3f} |")
            count += 1
            if count >= 10:
                break
        lines.append("")
    return _write(settings.features_reports_dir_path / "correlation_report.md", "\n".join(lines))


def _load_latest_reports() -> dict[str, dict]:
    reports = {}
    for path in settings.reports_dir_path.glob("latest_*.json"):
        module = path.stem.removeprefix("latest_")
        reports[module] = json.loads(path.read_text(encoding="utf-8"))
    return reports


_PRIMARY_METRIC = {
    "occupancy": ("mape", True),   # (metric, lower_is_better)
    "pricing": ("mape", True),
    "restaurant": ("mape", True),
    "staffing": ("mape", True),
    "churn": ("roc_auc", False),
}


def write_comparison_and_leaderboard() -> tuple[Path, Path]:
    reports = _load_latest_reports()

    comparison_lines = ["# Model Comparison", ""]
    leaderboard_rows: list[tuple[str, str, float, str]] = []

    for module, payload in sorted(reports.items()):
        metrics_by_model = payload.get("metrics", {})
        metric_name, lower_is_better = _PRIMARY_METRIC.get(module, ("mae", True))

        comparison_lines.append(f"## {module}")
        comparison_lines.append("")
        all_metric_keys: list[str] = []
        for m in metrics_by_model.values():
            for k in m:
                if k not in all_metric_keys:
                    all_metric_keys.append(k)
        header = "| Model | " + " | ".join(all_metric_keys) + " |"
        sep = "|---|" + "---|" * len(all_metric_keys)
        comparison_lines.append(header)
        comparison_lines.append(sep)

        best_model, best_value = None, None
        for model_key, model_metrics in metrics_by_model.items():
            row_values = [str(round(model_metrics.get(k, float("nan")), 4)) for k in all_metric_keys]
            comparison_lines.append(f"| {model_key} | " + " | ".join(row_values) + " |")

            value = model_metrics.get(metric_name)
            if value is None:
                continue
            if best_value is None or (value < best_value if lower_is_better else value > best_value):
                best_model, best_value = model_key, value

        comparison_lines.append("")
        if best_model is not None:
            leaderboard_rows.append((module, best_model, best_value, metric_name))

    comparison_path = _write(
        settings.models_reports_dir_path / "comparison.md", "\n".join(comparison_lines)
    )

    leaderboard_lines = ["# Model Leaderboard", "", "Best model per task, ranked by its primary metric:", ""]
    leaderboard_lines.append("| Task | Best Model | Metric | Value |")
    leaderboard_lines.append("|---|---|---|---|")
    for module, model, value, metric_name in leaderboard_rows:
        leaderboard_lines.append(f"| {module} | {model} | {metric_name} | {round(value, 4)} |")
    leaderboard_lines.append("")
    leaderboard_path = _write(
        settings.models_reports_dir_path / "leaderboard.md", "\n".join(leaderboard_lines)
    )

    return comparison_path, leaderboard_path
